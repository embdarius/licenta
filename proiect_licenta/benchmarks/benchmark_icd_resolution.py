"""Benchmark: Stage-2 exact-ICD resolution within predicted categories (Doctor v3).

Reuses the Doctor v3 with-nurse test split and diagnosis model, then evaluates
the Stage-2 resolver (built by `uv run train_icd_resolver`) at two granularities
(3-char rollup = headline, full code = secondary) and three blend variants:

  - blend       : the resolver's tuned alpha (cosine + prevalence)
  - prevalence  : alpha = 0 (frequency-only baseline)
  - cosine      : alpha = 1 (prototype-cosine only)

Metrics reported per (granularity, variant):
  - Oracle-category : within the TRUE category, is the true code in top-5/top-10?
                      (isolates Stage-2 from Stage-1 category errors)
  - End-to-end      : across the top-5 PREDICTED categories, is the true code in
                      the union of (top-5 cats x top-5 codes), or the flat top-10
                      ranked by P(category) x score?
  - Conditional     : end-to-end recall among rows whose true category is in the
                      predicted top-5 (removes the Stage-1 category ceiling).

Run: ``uv run python benchmarks/benchmark_icd_resolution.py``
"""
# Force UTF-8 stdout/stderr BEFORE crewai is imported (it wraps the streams).
# The v3 loader's PMH step prints non-cp1252 chars (e.g. "≥"), which crash a
# piped Windows console (cp1252). Mirrors the guard in main.py. errors="replace"
# guarantees a stray char never aborts the run.
import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import (
    TRIAGE_V1_DIR, DIAGNOSIS_CSV, DOCTOR_V3_DIR, DOCTOR_V3_ICD_RESOLVER_DIR,
)
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.training.train_nurse_v3 import (
    load_and_clean_data as load_v3_data,
    build_features as build_v3_with_nurse_features,
)
from proiect_licenta import icd_resolution as icdr

TEST_SIZE = 0.2
SPLIT_SEED = 42
K_PER_CAT = 5
K_FLAT = 10
TOP_CATS = 5


def print_section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _attach_icd_version(df: pd.DataFrame) -> pd.DataFrame:
    """Re-load primary diagnosis with string-safe code/version/title (+ rollup)."""
    diag = pd.read_csv(
        DIAGNOSIS_CSV,
        dtype={"icd_code": str, "icd_version": str, "icd_title": str},
    )
    diag = diag[diag["seq_num"] == 1][
        ["stay_id", "icd_code", "icd_version", "icd_title"]
    ].drop_duplicates("stay_id")
    df = df.drop(columns=["icd_code", "icd_title"], errors="ignore")
    df = df.merge(diag, on="stay_id", how="left")
    df["icd_code"] = df["icd_code"].fillna("").str.strip().str.upper()
    df["icd_version"] = df["icd_version"].fillna("").str.strip()
    df["icd_title"] = df["icd_title"].fillna("")
    df["rollup_code"] = [
        icdr.rollup_icd(c, v) for c, v in zip(df["icd_code"], df["icd_version"])
    ]
    return df


def _oracle_recall(Q_test, df_test, true_cat_idx, gindex, diagnosis_labels,
                   code_col, alpha):
    """Within the TRUE category, recall@5 / recall@10 of the true code."""
    n = len(df_test)
    hit5 = hit10 = scored = 0
    true_codes = df_test[code_col].to_numpy()
    for c_idx, cat in enumerate(diagnosis_labels):
        entry = gindex.get(cat)
        if entry is None:
            continue
        rows = np.where(true_cat_idx == c_idx)[0]
        if len(rows) == 0:
            continue
        cos = np.asarray(Q_test[rows] @ entry["centroids"].T, dtype=np.float32)
        score = icdr.blend_scores(cos, entry["prevalence"], alpha)
        order = np.argsort(-score, axis=1)
        codes_arr = np.asarray(entry["codes"])
        top5 = codes_arr[order[:, :K_PER_CAT]]
        top10 = codes_arr[order[:, :K_FLAT]]
        for j, r in enumerate(rows):
            tc = true_codes[r]
            scored += 1
            if tc in set(top5[j].tolist()):
                hit5 += 1
            if tc in set(top10[j].tolist()):
                hit10 += 1
    return hit5 / scored, hit10 / scored, scored


def _end_to_end_recall(Q_test, df_test, proba, top5_cat_idx, gindex,
                       diagnosis_labels, code_col, alpha, true_cat_idx):
    """End-to-end recall: union(top-5 cats x top-5 codes) and flat top-10.

    Also returns the conditional recall (rows whose true category is in the
    predicted top-5), which removes the Stage-1 category-recall ceiling.
    """
    n = len(df_test)
    true_codes = df_test[code_col].to_numpy()
    union_sets = [set() for _ in range(n)]
    flat_lists = [[] for _ in range(n)]

    for c_idx, cat in enumerate(diagnosis_labels):
        entry = gindex.get(cat)
        if entry is None:
            continue
        rows = np.where((top5_cat_idx == c_idx).any(axis=1))[0]
        if len(rows) == 0:
            continue
        cos = np.asarray(Q_test[rows] @ entry["centroids"].T, dtype=np.float32)
        score = icdr.blend_scores(cos, entry["prevalence"], alpha)
        order = np.argsort(-score, axis=1)[:, :K_PER_CAT]
        codes_arr = np.asarray(entry["codes"])
        p_rows = proba[rows, c_idx]
        for j, r in enumerate(rows):
            cand = codes_arr[order[j]]
            scs = score[j, order[j]]
            for cc, sc in zip(cand.tolist(), scs.tolist()):
                union_sets[r].add(cc)
                flat_lists[r].append((p_rows[j] * sc, cc))

    cat_correct = (top5_cat_idx == true_cat_idx[:, None]).any(axis=1)
    union_hit = flat_hit = 0
    cond_union_hit = cond_total = 0
    for r in range(n):
        tc = true_codes[r]
        in_union = tc in union_sets[r]
        flat_top = [c for _, c in sorted(flat_lists[r], reverse=True)[:K_FLAT]]
        in_flat = tc in set(flat_top)
        union_hit += in_union
        flat_hit += in_flat
        if cat_correct[r]:
            cond_total += 1
            cond_union_hit += in_union
    return (
        union_hit / n,
        flat_hit / n,
        (cond_union_hit / cond_total if cond_total else 0.0),
        cat_correct.mean(),
    )


def main():
    print("\n" + "#" * 72)
    print("  STAGE-2 EXACT-ICD RESOLUTION BENCHMARK (Doctor v3)")
    print("#" * 72)

    # 1. Load data + reproduce v3 split.
    df = load_v3_data()
    df = _attach_icd_version(df)
    df["complaint_text_norm"] = df["chiefcomplaint"].apply(normalize_complaint_text)

    print_section("BUILDING v3 WITH-NURSE FEATURES (for the diagnosis model)")
    features = build_v3_with_nurse_features(df)

    meta = json.loads((DOCTOR_V3_DIR / "metadata.json").read_text())
    diagnosis_labels = meta["diagnosis_labels"]
    diag_map = {l: i for i, l in enumerate(sorted(df["diagnosis_group"].unique()))}
    # Both are sorted(unique) by construction, so proba column i <-> diagnosis_labels[i].
    assert list(diag_map.keys()) == diagnosis_labels, (
        "diagnosis label space mismatch between data and v3 metadata"
    )
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)

    idx_all = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y_diag,
    )
    X_test = features.iloc[test_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    print(f"  Test set: {len(df_test):,} stays")

    # 2. Diagnosis model -> category soft-probs -> top-5 categories.
    print_section("LOADING DIAGNOSIS MODEL + RESOLVER")
    diag_model = joblib.load(DOCTOR_V3_DIR / "diagnosis_model.joblib")
    proba = diag_model.predict_proba(X_test)  # (N, 13)
    # Reorder proba columns to diagnosis_labels order if needed: the model's
    # class index i corresponds to diag_map label order == sorted labels, which
    # is exactly diagnosis_labels in v3 metadata. They match by construction.
    top5_cat_idx = np.argsort(-proba, axis=1)[:, :TOP_CATS]

    true_cat_idx = df_test["diagnosis_group"].map(diag_map).to_numpy()

    resolver = icdr.load_resolver(DOCTOR_V3_ICD_RESOLVER_DIR)
    alpha_blend = resolver["alpha"]
    print(f"  Resolver alpha (tuned) = {alpha_blend:.2f}  | n_train={resolver['n_train']:,}")

    tfidf = joblib.load(TRIAGE_V1_DIR / "tfidf_vectorizer.joblib")
    Q_test = icdr.vectorize_queries(df_test["complaint_text_norm"].tolist(), tfidf)

    # Category recall sanity (Stage-1 ceiling).
    cat_top5 = (top5_cat_idx == true_cat_idx[:, None]).any(axis=1).mean()
    cat_top1 = (np.argmax(proba, axis=1) == true_cat_idx).mean()
    print(f"  Stage-1 category recall: top-1={cat_top1:.4f}  top-5={cat_top5:.4f}")

    variants = {"prevalence": 0.0, "cosine": 1.0, "blend": alpha_blend}

    # 3. Evaluate each granularity.
    for g in icdr.GRANULARITIES:
        gindex = resolver["granularities"][g]
        code_col = "rollup_code" if g == "rollup" else "icd_code"
        label = "3-char rollup (PRIMARY)" if g == "rollup" else "full code (secondary)"
        print_section(f"GRANULARITY: {label}")

        n_codes_total = sum(len(e["codes"]) for e in gindex.values())
        print(f"  Candidate pool: {n_codes_total:,} codes across {len(gindex)} categories")

        print(f"\n  {'Variant':<12s} {'alpha':>6s} | "
              f"{'Oracle@5':>9s} {'Oracle@10':>10s} | "
              f"{'E2E union':>10s} {'E2E flat10':>11s} {'Cond@union':>11s}")
        print(f"  {'-'*78}")
        for name, alpha in variants.items():
            o5, o10, _ = _oracle_recall(
                Q_test, df_test, true_cat_idx, gindex, diagnosis_labels,
                code_col, alpha,
            )
            ee_union, ee_flat, cond, _ = _end_to_end_recall(
                Q_test, df_test, proba, top5_cat_idx, gindex,
                diagnosis_labels, code_col, alpha, true_cat_idx,
            )
            print(f"  {name:<12s} {alpha:>6.2f} | "
                  f"{o5:>9.4f} {o10:>10.4f} | "
                  f"{ee_union:>10.4f} {ee_flat:>11.4f} {cond:>11.4f}")

    # 4. Spot-check a couple of cases.
    print_section("SPOT-CHECK (blend, rollup) — first 3 admitted test stays")
    gindex = resolver["granularities"]["rollup"]
    for r in range(min(3, len(df_test))):
        complaint = df_test["chiefcomplaint"].iloc[r]
        true_cat = df_test["diagnosis_group"].iloc[r]
        true_roll = df_test["rollup_code"].iloc[r]
        true_title = df_test["icd_title"].iloc[r]
        cat_probs = [
            (diagnosis_labels[c], float(proba[r, c])) for c in top5_cat_idx[r]
        ]
        q = np.asarray(Q_test[r].todense()).ravel()
        res = icdr.resolve_exact_diagnoses(
            cat_probs, q, gindex, alpha_blend, k_per_cat=K_PER_CAT, k_flat=K_FLAT,
        )
        print(f"\n  Complaint: {complaint}")
        print(f"  TRUE: [{true_cat}] {true_roll}  {true_title[:50]}")
        print(f"  Flat top-{K_FLAT}:")
        for d in res["flat_top"]:
            mark = " <== TRUE" if d["code"] == true_roll else ""
            print(f"    [{d['category']:<28s}] {d['code']:<6s} "
                  f"{d['combined']:.3f}  {d['title'][:42]}{mark}")

    print("\n" + "#" * 72)
    print("  BENCHMARK COMPLETE")
    print("#" * 72 + "\n")


if __name__ == "__main__":
    main()
