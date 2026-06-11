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
def build_index(vectors_l2, codes, titles, categories) -> dict:
    """Build a per-category candidate index from train-split rows.

    Parameters
    ----------
    vectors_l2 : sparse (n, V) — L2-normalized TF-IDF complaint vectors.
    codes      : length-n sequence of ICD code keys (already rolled up to the
                 desired granularity by the caller).
    titles     : length-n sequence of human-readable ICD titles.
    categories : length-n sequence of diagnosis-group labels.

    Returns
    -------
    dict: category_label -> {
        "codes":      list[str]        (m candidates),
        "titles":     list[str]        (representative title per code),
        "prevalence": float32 (m,)     (count / category_total),
        "centroids":  float32 (m, V)   (L2-normalized prototype per code),
        "n_total":    int,
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
        code_list, title_list, prev_list, centroids = [], [], [], []
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

        index[cat] = {
            "codes": code_list,
            "titles": title_list,
            "prevalence": np.asarray(prev_list, dtype=np.float32),
            "centroids": np.vstack(centroids).astype(np.float32),
            "n_total": int(total),
        }
    return index


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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def make_resolver(granularities: dict, alpha: float, vocab_size: int,
                  tfidf_path: str, n_train: int, extra: dict | None = None) -> dict:
    """Assemble the on-disk resolver object."""
    resolver = {
        "alpha": float(alpha),
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
