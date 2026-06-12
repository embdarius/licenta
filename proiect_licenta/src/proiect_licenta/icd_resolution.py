"""Stage-2 exact-ICD resolution within a predicted diagnosis category.

The Doctor v3 diagnosis model predicts a diagnosis *category* (one of 13
ICD-chapter groups). This module adds a second stage that, given a category,
ranks the exact ICD diagnoses inside it and surfaces the top-k codes.

Design (see plan + docs/agents/doctor-agent.md):

  - Candidate pool per category = the ICD codes observed in that category in
    the Doctor v3 **training split** (no test leakage). Built at two
    granularities: 3-char rollup (ICD-10 rubric / ICD-9 3-digit — the headline
    target) and full code (secondary/stretch).
  - Each candidate carries:
      * a **prototype centroid** = mean of the L2-normalized TF-IDF chief-
        complaint vectors of the training stays that received that code
        (symptom->symptom matching, sidesteps the complaint-vs-ICD-title gap);
      * a **prevalence prior** = the code's frequency within its category.
  - Ranking blends the two:
        score = alpha * minmax(cosine) + (1 - alpha) * minmax(prevalence)
    with cosine = centroid . query (both L2-normalized, so a plain dot). alpha
    is tuned offline on a held-out train slice and stored in the resolver.

This module is dependency-light (reuses the existing TF-IDF vectorizer +
numpy/sklearn already in the stack). It does NOT load any MIMIC data itself —
the builder (`training/train_icd_resolver.py`) feeds it the train rows, and the
benchmark / runtime feed it query vectors.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import normalize

# Filenames under DOCTOR_V3_ICD_RESOLVER_DIR.
RESOLVER_FILENAME = "icd_resolver.joblib"
RESOLVER_META_FILENAME = "metadata.json"

GRANULARITIES = ("rollup", "full")

# Physiology features for the vitals-conditioned similarity term (Stage-2 v2).
# 6 continuous vitals + 7 binary abnormal flags, all z-scored together so a
# Euclidean "nearest physiological prototype" comparison is scale-consistent.
# Column names match _clean_vitals (train_nurse) and doctor_tool_v3.
PHYSIO_CONT = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
PHYSIO_FLAGS = ["fever", "tachycardia", "bradycardia", "tachypnea",
                "hypoxia", "hypertension", "hypotension"]
PHYSIO_COLS = PHYSIO_CONT + PHYSIO_FLAGS


# ---------------------------------------------------------------------------
# ICD code rollup
# ---------------------------------------------------------------------------
def rollup_icd(code, version) -> str:
    """Roll an ICD code up to its 3-char category (the headline target).

    ICD-9 (version == "9"):
      - E-codes (external causes) use a 4-char rubric: ``E932`` <- ``E9320``.
      - Numeric and V-codes use the first 3 chars: ``458`` <- ``4589``,
        ``070`` <- ``07070`` (leading zeros preserved — pass codes as str),
        ``V76`` <- ``V7644``.
    ICD-10 (everything else): the first 3 chars are the category rubric,
      e.g. ``I21`` <- ``I2109``.
    """
    c = str(code).strip().upper()
    if not c:
        return c
    if str(version).strip() == "9" and c.startswith("E"):
        return c[:4]
    return c[:3]


# ---------------------------------------------------------------------------
# Query vectorization
# ---------------------------------------------------------------------------
def vectorize_queries(texts, tfidf):
    """TF-IDF transform + L2 row-normalize a list of normalized complaint texts.

    Returns a sparse CSR matrix (n, V). The caller is expected to have already
    passed ``texts`` through ``preprocessing.normalize_complaint_text`` (the
    same normalization used to fit the vectorizer and to build the centroids).
    """
    return normalize(tfidf.transform(texts))  # L2 per row


def vectorize_query(text, tfidf) -> np.ndarray:
    """Vectorize a single complaint into a dense L2-normalized vector (V,)."""
    return np.asarray(vectorize_queries([text], tfidf).todense()).ravel()


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
def build_index(vectors_l2, codes, titles, categories, physio=None) -> dict:
    """Build a per-category candidate index from train-split rows.

    Parameters
    ----------
    vectors_l2 : sparse (n, V) — L2-normalized TF-IDF complaint vectors.
    codes      : length-n sequence of ICD code keys (already rolled up to the
                 desired granularity by the caller).
    titles     : length-n sequence of human-readable ICD titles.
    categories : length-n sequence of diagnosis-group labels.
    physio     : optional (n, P) standardized physiology matrix (z-scored vitals
                 + flags, see `physio_matrix`). When given, a per-code vitals
                 centroid is stored for the vitals-conditioned similarity term.

    Returns
    -------
    dict: category_label -> {
        "codes":          list[str]        (m candidates),
        "titles":         list[str]        (representative title per code),
        "prevalence":     float32 (m,)     (count / category_total),
        "centroids":      float32 (m, V)   (L2-normalized text prototype per code),
        "vital_centroids":float32 (m, P) or None  (mean standardized physiology),
        "n_total":        int,
    }
    """
    n = vectors_l2.shape[0]
    meta = pd.DataFrame({
        "row": np.arange(n),
        "code": [str(c) for c in codes],
        "title": list(titles),
        "cat": list(categories),
    })

    index: dict = {}
    for cat, sub in meta.groupby("cat", sort=True):
        total = len(sub)
        code_list, title_list, prev_list, centroids, vital_centroids = [], [], [], [], []
        for code, cg in sub.groupby("code", sort=True):
            rows = cg["row"].to_numpy()
            centroid = np.asarray(vectors_l2[rows].mean(axis=0)).ravel()
            nrm = np.linalg.norm(centroid)
            if nrm > 0:
                centroid = centroid / nrm
            code_list.append(str(code))
            # Most common title for this code (codes can carry minor title
            # variants; rollups merge several titles).
            mode = cg["title"].mode()
            title_list.append(str(mode.iloc[0]) if len(mode) else "")
            prev_list.append(len(cg) / total)
            centroids.append(centroid.astype(np.float32))
            if physio is not None:
                vital_centroids.append(physio[rows].mean(axis=0).astype(np.float32))

        index[cat] = {
            "codes": code_list,
            "titles": title_list,
            "prevalence": np.asarray(prev_list, dtype=np.float32),
            "centroids": np.vstack(centroids).astype(np.float32),
            "vital_centroids": (
                np.vstack(vital_centroids).astype(np.float32)
                if physio is not None else None
            ),
            "n_total": int(total),
        }
    return index


# ---------------------------------------------------------------------------
# Physiology (vitals) standardization + similarity (Stage-2 v2)
# ---------------------------------------------------------------------------
def build_standardizer(df, cols=None) -> dict:
    """Fit a z-score standardizer over `cols` (default PHYSIO_COLS) from rows.

    Continuous vitals, binary flags, vital deltas, and rhythm one-hots are
    standardized together so the Euclidean "nearest physiological prototype"
    comparison treats every dimension on the same scale. Returns {cols, mean,
    std} (std floored away from 0 to avoid divide-by-zero on a constant column).
    The caller chooses `cols` (e.g. snapshot-only, or snapshot + rhythm +
    deltas); inference reads them back from the saved standardizer.
    """
    cols = list(cols) if cols is not None else list(PHYSIO_COLS)
    X = df[cols].astype(float)
    mean = X.mean().to_numpy().astype(np.float32)
    std = X.std(ddof=0).to_numpy().astype(np.float32)
    std = np.where(std < 1e-8, np.float32(1.0), std)
    return {"cols": cols, "mean": mean, "std": std}


def physio_matrix(df, standardizer) -> np.ndarray:
    """Standardize a DataFrame's PHYSIO_COLS into an (n, P) z-scored matrix."""
    X = df[standardizer["cols"]].astype(float).to_numpy()
    return ((X - standardizer["mean"]) / standardizer["std"]).astype(np.float32)


def physio_vector(values: dict, standardizer) -> np.ndarray:
    """Standardize a single patient's physiology dict into a (P,) vector.

    Missing entries fall back to the training mean (→ z-score 0, "average"),
    matching how the doctor model imputes unknown vitals.
    """
    cols, mean, std = standardizer["cols"], standardizer["mean"], standardizer["std"]
    x = np.empty(len(cols), dtype=np.float32)
    for i, c in enumerate(cols):
        v = values.get(c, None)
        x[i] = float(v) if v is not None else float(mean[i])
    return ((x - mean) / std).astype(np.float32)


def _neg_euclidean(queries, centroids) -> np.ndarray:
    """Negative Euclidean distance — higher = closer. Supports (P,) or (N,P)."""
    q = np.asarray(queries, dtype=np.float32)
    c = np.asarray(centroids, dtype=np.float32)
    if q.ndim == 1:
        return -np.sqrt(np.maximum(((c - q) ** 2).sum(axis=1), 0.0))
    q2 = (q * q).sum(axis=1)[:, None]
    c2 = (c * c).sum(axis=1)[None, :]
    d2 = np.maximum(q2 + c2 - 2.0 * (q @ c.T), 0.0)
    return -np.sqrt(d2)


def blend3(text_cos, vitals_sim, prevalence, weights) -> np.ndarray:
    """Three-term blend: w_text·text + w_vit·vitals + w_prev·prevalence.

    Each term is min-max normalized within the candidate set first. Supports a
    single query (1-D `text_cos`/`vitals_sim`) or a batch (2-D, shape (N, m)).
    `weights` is a dict with keys text/vit/prev. `vitals_sim` may be None when
    w_vit == 0 (text-only / prevalence-only baselines).
    """
    wt, wv, wp = weights["text"], weights.get("vit", 0.0), weights["prev"]
    prev_n = _minmax(prevalence)
    text_cos = np.asarray(text_cos, dtype=np.float32)
    if text_cos.ndim == 1:
        t = _minmax(text_cos)
        v = _minmax(vitals_sim) if (wv and vitals_sim is not None) else 0.0
        return wt * t + wv * v + wp * prev_n
    t = _minmax_rows(text_cos)
    v = _minmax_rows(vitals_sim) if (wv and vitals_sim is not None) else 0.0
    return wt * t + wv * (v if np.isscalar(v) else v) + wp * prev_n[None, :]


def attach_centroids(index: dict, matrix, codes, categories, key, row_mask=None):
    """Attach a per-code centroid set under `key`, in entry['codes'] order.

    `matrix` is an (n, P) feature matrix (e.g. standardized PMH vectors) aligned
    with `codes`/`categories`. When `row_mask` is given (bool (n,)), only those
    rows contribute to the means — used to build PMH centroids from has-history
    rows only. Codes with no contributing rows get a zero (neutral) centroid.
    """
    n = matrix.shape[0]
    P = matrix.shape[1]
    meta = pd.DataFrame({
        "row": np.arange(n),
        "code": [str(c) for c in codes],
        "cat": list(categories),
    })
    if row_mask is not None:
        meta = meta[np.asarray(row_mask, dtype=bool)]
    for cat, entry in index.items():
        sub = meta[meta["cat"] == cat]
        code_to_rows = {c: g["row"].to_numpy() for c, g in sub.groupby("code")}
        cents = np.zeros((len(entry["codes"]), P), dtype=np.float32)
        for i, code in enumerate(entry["codes"]):
            rows = code_to_rows.get(str(code))
            if rows is not None and len(rows):
                cents[i] = matrix[rows].mean(axis=0)
        entry[key] = cents


def gated_score(text_cos, vital_sim, pmh_sim, prevalence,
                weights_pmh, weights_nopmh, has_pmh):
    """Blend terms, applying the PMH term only to patients with history.

    Works for a single query (1-D term arrays, scalar `has_pmh`) or a batch
    (2-D (N, m) term arrays, `has_pmh` shape (N,)). `vital_sim`/`pmh_sim` may be
    None (term dropped). Patients with `has_pmh` use `weights_pmh` (4-term);
    the rest use `weights_nopmh` (3-term). Each term is min-max'd within the
    candidate set first.
    """
    text_cos = np.asarray(text_cos, dtype=np.float32)
    is_batch = text_cos.ndim == 2
    mm = _minmax_rows if is_batch else _minmax
    Tn = mm(text_cos)
    Vn = mm(vital_sim) if vital_sim is not None else 0.0
    Pn = _minmax(prevalence)
    if is_batch:
        Pn = Pn[None, :]

    def _combine(w):
        return (w["text"] * Tn + w.get("vit", 0.0) * Vn + w["prev"] * Pn)

    score_no = _combine(weights_nopmh)
    if pmh_sim is None:
        return score_no
    Mn = mm(pmh_sim)
    score_pmh = _combine(weights_pmh) + weights_pmh.get("pmh", 0.0) * Mn
    if is_batch:
        return np.where(np.asarray(has_pmh, dtype=bool)[:, None], score_pmh, score_no)
    return score_pmh if has_pmh else score_no


def rank_within_category_v3(text_q_l2, physio_q, pmh_q, has_pmh, cat_entry,
                            weights_pmh, weights_nopmh, k):
    """Single-query ranking with the gated PMH term (and the vitals term)."""
    centroids = cat_entry["centroids"]
    text_cos = centroids @ np.asarray(text_q_l2, dtype=np.float32).ravel()
    vital_sim = None
    if physio_q is not None and cat_entry.get("vital_centroids") is not None:
        vital_sim = _neg_euclidean(physio_q, cat_entry["vital_centroids"])
    pmh_sim = None
    if has_pmh and pmh_q is not None and cat_entry.get("pmh_centroids") is not None:
        pmh_sim = _neg_euclidean(pmh_q, cat_entry["pmh_centroids"])
    score = gated_score(text_cos, vital_sim, pmh_sim, cat_entry["prevalence"],
                        weights_pmh, weights_nopmh, bool(has_pmh))
    order = np.argsort(score)[::-1][:k]
    return [
        (cat_entry["codes"][i], cat_entry["titles"][i], float(score[i]),
         float(text_cos[i]), float(cat_entry["prevalence"][i]))
        for i in order
    ]


def rank_within_category_v2(text_q_l2, physio_q, cat_entry, weights, k):
    """Single-query ranking with the optional vitals term.

    `physio_q` is a (P,) standardized patient physiology vector (or None).
    Returns (code, title, score, text_cos, prevalence) tuples, best first.
    """
    centroids = cat_entry["centroids"]
    text_cos = centroids @ np.asarray(text_q_l2, dtype=np.float32).ravel()
    vc = cat_entry.get("vital_centroids")
    if physio_q is not None and vc is not None and weights.get("vit", 0.0) > 0:
        vitals_sim = _neg_euclidean(physio_q, vc)
    else:
        vitals_sim = None
    score = blend3(text_cos, vitals_sim, cat_entry["prevalence"], weights)
    order = np.argsort(score)[::-1][:k]
    return [
        (cat_entry["codes"][i], cat_entry["titles"][i], float(score[i]),
         float(text_cos[i]), float(cat_entry["prevalence"][i]))
        for i in order
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _minmax(a: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1-D array to [0, 1]; flat arrays map to 0.5."""
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.full_like(a, 0.5, dtype=np.float32)
    return (a - lo) / (hi - lo)


def _minmax_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise min-max normalize a 2-D array (each row to [0, 1])."""
    a = np.asarray(a, dtype=np.float32)
    lo = a.min(axis=1, keepdims=True)
    hi = a.max(axis=1, keepdims=True)
    denom = hi - lo
    out = np.where(denom < 1e-12, np.float32(0.5), (a - lo) / np.where(denom < 1e-12, 1.0, denom))
    return out.astype(np.float32)


def blend_scores(cos, prevalence, alpha: float) -> np.ndarray:
    """Blend cosine + prevalence into a final score.

    Works for a single query (``cos`` shape ``(m,)``) or a batch (``cos`` shape
    ``(N, m)``). ``prevalence`` is always ``(m,)``.
    """
    cos = np.asarray(cos, dtype=np.float32)
    prev_n = _minmax(prevalence)
    if cos.ndim == 1:
        return alpha * _minmax(cos) + (1.0 - alpha) * prev_n
    return alpha * _minmax_rows(cos) + (1.0 - alpha) * prev_n[None, :]


def score_category_batch(queries_l2, cat_entry, alpha: float) -> np.ndarray:
    """Score every query against one category's candidates: returns (N, m).

    ``queries_l2`` is a sparse (N, V) L2-normalized matrix; the dot with the
    L2-normalized centroids is the cosine similarity.
    """
    cos = (queries_l2 @ cat_entry["centroids"].T)  # (N, m), dense ndarray
    cos = np.asarray(cos, dtype=np.float32)
    return blend_scores(cos, cat_entry["prevalence"], alpha)


def rank_within_category(query_l2: np.ndarray, cat_entry, alpha: float, k: int):
    """Rank a single query's candidates within one category.

    Returns a list of ``(code, title, score, cosine, prevalence)`` tuples,
    highest score first, length ``min(k, m)``.
    """
    query_l2 = np.asarray(query_l2, dtype=np.float32).ravel()
    cos = cat_entry["centroids"] @ query_l2  # (m,)
    score = blend_scores(cos, cat_entry["prevalence"], alpha)
    order = np.argsort(score)[::-1][:k]
    return [
        (
            cat_entry["codes"][i],
            cat_entry["titles"][i],
            float(score[i]),
            float(cos[i]),
            float(cat_entry["prevalence"][i]),
        )
        for i in order
    ]


def resolve_exact_diagnoses(
    category_probs,
    query_l2: np.ndarray,
    granularity_index: dict,
    alpha: float,
    k_per_cat: int = 5,
    k_flat: int = 10,
) -> dict:
    """Resolve exact diagnoses for one patient across their top categories.

    Parameters
    ----------
    category_probs : iterable of ``(category_label, p_category)`` — typically
        the Doctor v3 top-5 categories with their softmax probabilities.
    query_l2 : dense (V,) L2-normalized TF-IDF complaint vector.
    granularity_index : one granularity's index (``resolver["granularities"][g]``).
    alpha : blend weight (use the resolver's stored alpha).

    Returns ``{"per_category": [...], "flat_top": [...]}`` where the flat list
    is ranked by ``p_category * stage2_score`` across all categories.
    """
    per_category, flat = [], []
    for cat, p in category_probs:
        entry = granularity_index.get(cat)
        if entry is None:
            continue
        ranked = rank_within_category(query_l2, entry, alpha, k_per_cat)
        per_category.append({
            "category": cat,
            "p_category": float(p),
            "codes": [
                {"code": code, "title": title, "score": score,
                 "cosine": cos, "prevalence": prev}
                for (code, title, score, cos, prev) in ranked
            ],
        })
        for (code, title, score, cos, prev) in ranked:
            flat.append({
                "category": cat, "code": code, "title": title,
                "combined": float(p) * score, "score": score,
            })
    flat.sort(key=lambda d: d["combined"], reverse=True)
    return {"per_category": per_category, "flat_top": flat[:k_flat]}


def resolve_exact_diagnoses_v3(
    category_probs, query_l2, physio_q, pmh_q, has_pmh,
    granularity_index, weights_pmh, weights_nopmh,
    k_per_cat: int = 5, k_flat: int = 10,
) -> dict:
    """Resolve exact diagnoses with the gated PMH term + vitals term.

    `pmh_q` is a (P,) standardized PMH vector and `has_pmh` whether the patient
    has prior history (PMH term applies only then). Falls back to the v2/v1
    behavior when those are absent.
    """
    per_category, flat = [], []
    for cat, p in category_probs:
        entry = granularity_index.get(cat)
        if entry is None:
            continue
        ranked = rank_within_category_v3(
            query_l2, physio_q, pmh_q, has_pmh, entry,
            weights_pmh, weights_nopmh, k_per_cat)
        per_category.append({
            "category": cat,
            "p_category": float(p),
            "codes": [
                {"code": code, "title": title, "score": score,
                 "text_cos": tc, "prevalence": prev}
                for (code, title, score, tc, prev) in ranked
            ],
        })
        for (code, title, score, tc, prev) in ranked:
            flat.append({
                "category": cat, "code": code, "title": title,
                "combined": float(p) * score, "score": score,
            })
    flat.sort(key=lambda d: d["combined"], reverse=True)
    return {"per_category": per_category, "flat_top": flat[:k_flat]}


def resolve_exact_diagnoses_v2(
    category_probs,
    query_l2: np.ndarray,
    physio_q,
    granularity_index: dict,
    weights: dict,
    k_per_cat: int = 5,
    k_flat: int = 10,
) -> dict:
    """Like `resolve_exact_diagnoses` but with the vitals-conditioned term.

    `physio_q` is a (P,) standardized patient physiology vector (or None to
    fall back to text+prevalence). `weights` is the 3-term blend {text,vit,prev}.
    """
    per_category, flat = [], []
    for cat, p in category_probs:
        entry = granularity_index.get(cat)
        if entry is None:
            continue
        ranked = rank_within_category_v2(query_l2, physio_q, entry, weights, k_per_cat)
        per_category.append({
            "category": cat,
            "p_category": float(p),
            "codes": [
                {"code": code, "title": title, "score": score,
                 "text_cos": tc, "prevalence": prev}
                for (code, title, score, tc, prev) in ranked
            ],
        })
        for (code, title, score, tc, prev) in ranked:
            flat.append({
                "category": cat, "code": code, "title": title,
                "combined": float(p) * score, "score": score,
            })
    flat.sort(key=lambda d: d["combined"], reverse=True)
    return {"per_category": per_category, "flat_top": flat[:k_flat]}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def make_resolver(granularities: dict, alpha: float, vocab_size: int,
                  tfidf_path: str, n_train: int, weights: dict | None = None,
                  standardizer: dict | None = None,
                  weights_pmh: dict | None = None,
                  pmh_standardizer: dict | None = None,
                  extra: dict | None = None) -> dict:
    """Assemble the on-disk resolver object.

    `alpha` is the legacy 2-term text-vs-prevalence weight (kept so the
    backward-compatible `rank_within_category` path keeps working). `weights`
    is the tuned 3-term blend {text, vit, prev} and `standardizer` the
    physiology z-score fit (vitals-conditioned v2 path). `weights_pmh` is the
    tuned 4-term blend {text, vit, pmh, prev} applied to patients with history,
    and `pmh_standardizer` the PMH z-score fit (gated v3 path).
    """
    resolver = {
        "alpha": float(alpha),
        "weights": weights,                  # 3-term, no-PMH patients
        "weights_pmh": weights_pmh,          # 4-term, has-PMH patients
        "standardizer": standardizer,        # physiology z-score fit
        "pmh_standardizer": pmh_standardizer,  # PMH z-score fit
        "granularities": granularities,   # {"rollup": {...}, "full": {...}}
        "vocab_size": int(vocab_size),
        "tfidf_path": str(tfidf_path),
        "n_train": int(n_train),
        "built_at": datetime.now().isoformat(),
    }
    if extra:
        resolver.update(extra)
    return resolver


def save_resolver(resolver: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(resolver, out_dir / RESOLVER_FILENAME)


def load_resolver(out_dir) -> dict:
    return joblib.load(out_dir / RESOLVER_FILENAME)
