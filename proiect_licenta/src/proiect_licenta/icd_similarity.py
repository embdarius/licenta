"""Graded (near-miss) similarity metrics for Stage-2 exact-ICD evaluation.

Evaluation-only. These helpers score the resolver's ranked code lists with
partial credit for clinically-close misses, complementing the strict exact-code
recall in benchmark_icd_resolution.py. Three engines share one grader interface:
TF-IDF title cosine (lexical), Gemini title cosine (semantic, cached offline),
and ICD-tree distance (shared prefix / chapter).

Graded credit takes the max similarity over the top-k predicted codes:

    graded@k(row) = max over c in topk(row) of sim(true_code_row, c)

Since sim(x, x) == 1, graded@k >= strict recall@k always; the gap is the partial
credit from near misses. Both title engines compare code titles on each side
(candidate title vs true-code title), so an exact hit scores 1.0. Codes absent
from the index fall back to the row's raw icd_title.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .icd_resolution import rollup_icd  # noqa: F401  (reused by the tree engine)


def _clean(titles):
    """Coerce a title iterable to a list of non-None strings."""
    return [("" if t is None else str(t)) for t in titles]


# Graded aggregation (shared by every engine)
def graded_max_over_topk(sim_all: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    """Per-row max similarity over the top-k ranked candidates.

    Parameters
    ----------
    sim_all : (n, m) similarity of each row's true code vs every candidate.
    order   : (n, m) candidate ranking (argsort of the resolver score, best first).
    k       : top-k cutoff.

    Returns (n,) graded credit in [0, 1].
    """
    sim_all = np.asarray(sim_all, dtype=np.float32)
    topk = np.asarray(order)[:, :k]
    return np.take_along_axis(sim_all, topk, axis=1).max(axis=1)


# TF-IDF title engine
def fit_title_tfidf(all_titles) -> TfidfVectorizer:
    """Fit a small word-level TF-IDF over the set of distinct ICD titles.

    Uni+bigrams so a shared head word ("hepatitis A" vs "hepatitis B") drives a
    high cosine while distinct tails pull it below 1. Fit on the FULL title set
    (candidates of both granularities + the test true titles) so the candidate
    and true vectors live in one shared vocabulary space.

    The token pattern keeps SINGLE-character tokens (``\\b\\w+\\b`` instead of
    sklearn's default ``\\w\\w+``). ICD titles distinguish variants precisely by
    those one-char tails - "hepatitis A" vs "hepatitis B", "type 1" vs "type 2" -
    and dropping them would collapse such near misses to an identical (1.0)
    vector, destroying the very signal this metric exists to capture.
    """
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1,
                          token_pattern=r"(?u)\b\w+\b")
    vec.fit(_clean(all_titles))
    return vec


def tfidf_vectors(titles, vec: TfidfVectorizer) -> np.ndarray:
    """Transform titles to dense L2-normalized rows (so cosine == dot)."""
    M = normalize(vec.transform(_clean(titles)))
    return np.asarray(M.todense(), dtype=np.float32)


# Grader: title-vector engines (TF-IDF now, Gemini in Phase B share this class)
class TitleGrader:
    """A title-cosine grader parameterized by a vectorizer callable.

    ``vectorize(titles) -> (n, D) L2-normalized ndarray`` (or returns ``None`` to
    signal the engine is unavailable, e.g. no Gemini cache + no API key). Prepared
    once per granularity via :meth:`prepare`, after which :meth:`oracle_sim_all`
    (batched, for the oracle tables) and :meth:`row_max_sim` (per-row, for the
    end-to-end union/flat tallies) provide the similarities the benchmark needs.
    """

    def __init__(self, name: str, vectorize):
        self.name = name
        self._vectorize = vectorize
        self._probe = None  # cached availability probe
        self.codes = None
        self.col = None
        self.C = None      # (M, D) candidate title vectors, in self.codes order
        self.true = None   # (n_test, D) true-title vectors, in df_test row order

    @property
    def available(self) -> bool:
        if self._probe is None:
            self._probe = self._vectorize([""]) is not None
        return bool(self._probe)

    def prepare(self, ctx: dict) -> None:
        """Build the candidate matrix + per-row true vectors for one granularity.

        ``ctx["code_to_title"]`` maps every candidate code (at this granularity)
        to its representative title; ``ctx["true_titles"]`` is the df_test rows'
        genuine ``icd_title`` in row order. (Other ctx keys are for the tree
        engine; ignored here.)
        """
        code_to_title = ctx["code_to_title"]
        self.codes = list(code_to_title)
        self.col = {c: i for i, c in enumerate(self.codes)}
        self.C = self._vectorize([code_to_title[c] for c in self.codes])
        self.true = self._vectorize(list(ctx["true_titles"]))

    def oracle_sim_all(self, row_idx: np.ndarray, entry_codes) -> np.ndarray:
        """(n_rows, m) cosine of those rows' true titles vs an entry's candidates."""
        cols = [self.col[c] for c in entry_codes]
        cand = self.C[cols]                       # (m, D)
        return np.asarray(self.true[row_idx] @ cand.T, dtype=np.float32)

    def row_max_sim(self, r: int, pred_codes) -> float:
        """Max cosine of row r's true title vs a small set of predicted codes."""
        cols = [self.col[c] for c in pred_codes if c in self.col]
        if not cols:
            return 0.0
        return float((self.C[cols] @ self.true[r]).max())


# Gemini title-embedding engine (Phase B) - semantic, cached, backend-agnostic
GEMINI_EMBED_MODEL = "gemini-embedding-001"
_GEMINI_BATCH = 100
_GEMINI_RATE_PER_MIN = 2000   # stay safely under the paid-tier 3000/min quota
_GEMINI_MAX_RETRIES = 8


def _gemini_api_key():
    import os
    try:  # mirror llm_config: populate from .env without overriding os.environ
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _make_gemini_embedder():
    """Return ``embed(batch) -> list[list[float]]`` using the google-genai SDK.

    Tries the new ``google.genai`` client first, then the legacy
    ``google.generativeai`` module. Raises if no key / SDK is available.
    """
    key = _gemini_api_key()
    if not key:
        raise RuntimeError("no GEMINI_API_KEY / GOOGLE_API_KEY in environment")
    try:  # new unified SDK
        from google import genai

        client = genai.Client(api_key=key)

        def embed(batch):
            resp = client.models.embed_content(
                model=GEMINI_EMBED_MODEL, contents=[t or " " for t in batch])
            return [list(e.values) for e in resp.embeddings]
        return embed
    except ImportError:
        pass
    import google.generativeai as genai  # legacy SDK fallback

    genai.configure(api_key=key)

    def embed(batch):  # legacy SDK: one content at a time
        out = []
        for t in batch:
            resp = genai.embed_content(model="models/text-embedding-004", content=(t or " "))
            out.append(list(resp["embedding"]))
        return out
    return embed


def _is_rate_limit(err) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()


def _retry_delay_seconds(err, default=30.0) -> float:
    import re
    s = str(err)
    m = re.search(r"retry in ([\d.]+)s", s) or re.search(r"retryDelay'?:\s*'?(\d+)", s)
    return (float(m.group(1)) + 1.0) if m else default


def build_or_load_title_embeddings(titles, cache_path, verbose=True):
    """Return a ``{title -> float32 vector}`` cache, building/extending it on miss.

    Loads any existing cache (joblib), embeds only the not-yet-cached titles, and
    saves the union back. Built once (needs a Gemini key at build time);
    thereafter every run - under any LLM backend - reads the cache and makes zero
    API calls. Embedding is **rate-limited** (paced under the per-minute quota)
    and **resumable**: each batch is saved incrementally and HTTP 429s are honored
    with the server's retry delay, so a kill/retry never loses progress. Returns
    ``None`` if the engine is unavailable (no key AND no cache), so the benchmark
    cleanly skips this one metric while the offline metrics still run.
    """
    import time
    import joblib

    cache = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = dict(joblib.load(cache_path))
        except Exception:
            cache = {}
    wanted = list({("" if t is None else str(t)) for t in titles})
    missing = [t for t in wanted if t not in cache]
    if not missing:
        return cache

    try:
        embed = _make_gemini_embedder()
    except Exception as e:
        if not cache:
            print(f"  [gemini] title-embedding engine unavailable ({e}); skipping.")
            return None
        print(f"  [gemini] cannot embed {len(missing)} new titles ({e}); "
              f"using {len(cache)} cached.")
        return cache

    def _save():
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(cache, cache_path)

    min_interval = _GEMINI_BATCH / (_GEMINI_RATE_PER_MIN / 60.0)  # sec/batch pacing
    if verbose:
        print(f"  [gemini] embedding {len(missing):,} new titles "
              f"({len(cache):,} cached) at <={_GEMINI_RATE_PER_MIN}/min...")
    n_batches = (len(missing) + _GEMINI_BATCH - 1) // _GEMINI_BATCH
    for bi in range(n_batches):
        batch = missing[bi * _GEMINI_BATCH:(bi + 1) * _GEMINI_BATCH]
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                t0 = time.time()
                vecs = embed(batch)
                for t, v in zip(batch, vecs):
                    cache[t] = np.asarray(v, dtype=np.float32)
                if bi % 20 == 19:
                    _save()  # periodic checkpoint
                time.sleep(max(0.0, min_interval - (time.time() - t0)))
                break
            except Exception as e:
                if _is_rate_limit(e) and attempt < _GEMINI_MAX_RETRIES - 1:
                    delay = _retry_delay_seconds(e)
                    _save()  # checkpoint before the wait
                    if verbose:
                        print(f"  [gemini] 429 rate limit; waiting {delay:.0f}s "
                              f"(batch {bi + 1}/{n_batches}, {len(cache):,} cached)...")
                    time.sleep(delay)
                    continue
                # non-rate-limit error, or retries exhausted
                _save()
                if not cache:
                    print(f"  [gemini] embedding failed ({e}); skipping.")
                    return None
                print(f"  [gemini] stopped early ({e}); using {len(cache):,} cached.")
                return cache
    _save()
    if verbose:
        print(f"  [gemini] cache ready: {len(cache):,} title vectors -> {cache_path}")
    return cache


def gemini_vectors(titles, cache):
    """Look titles up in the embedding cache -> (n, D) L2-normalized rows.

    Missing titles (not embedded) map to a zero row (cosine 0 with everything).
    """
    if not cache:
        return None
    dim = len(next(iter(cache.values())))
    M = np.zeros((len(titles), dim), dtype=np.float32)
    for i, t in enumerate(_clean(titles)):
        v = cache.get(t)
        if v is not None:
            M[i] = v
    return normalize(M)


# ICD-tree engine (Phase C) - hierarchical credit by shared prefix / chapter
W_ROLLUP = 0.6   # credit for sharing the 3-char rubric (full granularity only)
W_CHAPTER = 0.3  # credit for sharing the ICD chapter

# Unified chapter labels so an ICD-9 code and its ICD-10 equivalent land in the
# same bucket (e.g. ICD-9 390-459 and ICD-10 I00-I99 both -> "circulatory").
_ICD10_LETTER = {
    "A": "infectious", "B": "infectious", "C": "neoplasms", "E": "endocrine",
    "F": "mental", "G": "nervous", "I": "circulatory", "J": "respiratory",
    "K": "digestive", "L": "skin", "M": "musculoskeletal", "N": "genitourinary",
    "O": "pregnancy", "P": "perinatal", "Q": "congenital", "R": "symptoms",
    "S": "injury", "T": "injury", "U": "special", "Z": "factors",
}


def icd_chapter(code, version) -> str:
    """Map an ICD code to a unified clinical chapter label (version-aware).

    ICD-10 chapters follow the letter (with the D and H letters split by number);
    ICD-9 follows the classic numeric ranges, with V codes -> "factors" and E
    codes -> "external" so they align with the ICD-10 Z / V-Y chapters.
    """
    c = str(code).strip().upper()
    if not c:
        return ""
    if str(version).strip() == "9":
        if c.startswith("E"):
            return "external"
        if c.startswith("V"):
            return "factors"
        try:
            n = int(c[:3])
        except ValueError:
            return "other"
        bounds = [
            (139, "infectious"), (239, "neoplasms"), (279, "endocrine"),
            (289, "blood"), (319, "mental"), (389, "nervous"),
            (459, "circulatory"), (519, "respiratory"), (579, "digestive"),
            (629, "genitourinary"), (679, "pregnancy"), (709, "skin"),
            (739, "musculoskeletal"), (759, "congenital"), (779, "perinatal"),
            (799, "symptoms"), (999, "injury"),
        ]
        for hi, name in bounds:
            if n <= hi:
                return name
        return "other"
    # ICD-10
    L = c[0]
    try:
        n = int(c[1:3])
    except ValueError:
        n = 0
    if L == "D":
        return "neoplasms" if n <= 49 else "blood"
    if L == "H":
        return "eye" if n <= 59 else "ear"
    if L in ("V", "W", "X", "Y"):
        return "external"
    return _ICD10_LETTER.get(L, f"other:{L}")


class IcdTreeGrader:
    """Hierarchical-credit grader: shared full code (1.0) > 3-char rubric
    (``W_ROLLUP``) > ICD chapter (``W_CHAPTER``) > nothing (0).

    Same interface as :class:`TitleGrader` (prepare / oracle_sim_all /
    row_max_sim) so the benchmark iterates it identically. Compares CODES, not
    titles; reuses ``rollup_icd`` and :func:`icd_chapter`. ``available`` is always
    True (no external dependency).
    """

    name = "tree"
    available = True

    def __init__(self):
        self.col = None
        self.cand_roll = None     # (M,) candidate rollup keys
        self.cand_chap = None     # (M,) candidate chapter keys
        self.cand_full = None     # (M,) candidate full keys (granularity code)
        self.true_full = None     # (n,) per-row true keys
        self.true_roll = None
        self.true_chap = None

    @staticmethod
    def _keys(codes, versions):
        full = np.asarray([str(c) for c in codes], dtype=object)
        roll = np.asarray([rollup_icd(c, v) for c, v in zip(codes, versions)], dtype=object)
        chap = np.asarray([icd_chapter(c, v) for c, v in zip(codes, versions)], dtype=object)
        return full, roll, chap

    def prepare(self, ctx: dict) -> None:
        code_to_version = ctx["code_to_version"]
        codes = list(code_to_version)
        self.col = {c: i for i, c in enumerate(codes)}
        self.cand_full, self.cand_roll, self.cand_chap = self._keys(
            codes, [code_to_version[c] for c in codes])
        self.true_full, self.true_roll, self.true_chap = self._keys(
            ctx["true_codes"], ctx["true_versions"])

    def _sim_vec(self, ti, cand_idx):
        """Tiered similarity of true row `ti` against candidate positions cand_idx."""
        cf, cr, cc = self.cand_full[cand_idx], self.cand_roll[cand_idx], self.cand_chap[cand_idx]
        out = np.where(
            cf == self.true_full[ti], 1.0,
            np.where(cr == self.true_roll[ti], W_ROLLUP,
                     np.where(cc == self.true_chap[ti], W_CHAPTER, 0.0)))
        return out.astype(np.float32)

    def oracle_sim_all(self, row_idx: np.ndarray, entry_codes) -> np.ndarray:
        cand_idx = np.array([self.col[c] for c in entry_codes])
        return np.vstack([self._sim_vec(ti, cand_idx) for ti in row_idx])

    def row_max_sim(self, r: int, pred_codes) -> float:
        cand_idx = np.array([self.col[c] for c in pred_codes if c in self.col])
        if cand_idx.size == 0:
            return 0.0
        return float(self._sim_vec(r, cand_idx).max())
