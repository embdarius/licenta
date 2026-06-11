"""Benchmark: Stage-2 exact-ICD resolution — Doctor v3_base (pre-nurse) vs v3-nurse.

Both Doctor v3 diagnosis models share one test split, and the Stage-2 resolver
(built by `uv run train_icd_resolver`) is model-agnostic — its candidate codes +
prototype centroids come from the shared training split, not from either
diagnosis model. So the **oracle** metrics (which use the true category) are
identical for both models and reported once; only the **end-to-end** metrics
differ, isolating how much the nurse-collected data improves exact-ICD recall.

Evaluated at two granularities (3-char rollup = headline, full code = secondary)
and three blend variants:

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
    TRIAGE_V1_DIR, DIAGNOSIS_CSV, DOCTOR_V3_DIR, DOCTOR_V3_BASE_DIR,
    DOCTOR_V3_ICD_RESOLVER_DIR,
)
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.training.train_nurse_v3 import (
    load_and_clean_data as load_v3_data,
    build_features as build_v3_with_nurse_features,
)
from proiect_licenta.training.train_doctor_v3 import (
    build_features as build_v3_base_features,
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


def _print_recall_table(label, variants, row_fn):
    """Print a small per-variant table. `row_fn(name, alpha)` -> tuple of floats."""
    cols = row_fn("__header__", None)  # list of (header, width)
    head = "".join(f"{h:>{w}s}" for h, w in cols)
    print(f"\n  [{label}]  {'Variant':<12s}{head}")
    for name, alpha in variants.items():
        vals = row_fn(name, alpha)
        body = "".join(f"{v:>{w}.4f}" for v, (_, w) in zip(vals, cols))
        print(f"               {name:<12s}{body}")


def main():
    print("\n" + "#" * 72)
    print("  STAGE-2 EXACT-ICD BENCHMARK — Doctor v3_base (pre-nurse) vs v3 with-nurse")
    print("#" * 72)

    # 1. Load once + reproduce the SHARED v3_base / v3-nurse split.
    df = load_v3_data()
    df = _attach_icd_version(df)
    df["complaint_text_norm"] = df["chiefcomplaint"].apply(normalize_complaint_text)

    print_section("BUILDING FEATURES (v3_base + v3 with-nurse; one shared test split)")
    features_base = build_v3_base_features(df)
    features_nurse = build_v3_with_nurse_features(df)

    meta_nurse = json.loads((DOCTOR_V3_DIR / "metadata.json").read_text())
    meta_base = json.loads((DOCTOR_V3_BASE_DIR / "metadata.json").read_text())
    diagnosis_labels = meta_nurse["diagnosis_labels"]
    assert meta_base["diagnosis_labels"] == diagnosis_labels, (
        "v3_base and v3-nurse have different diagnosis label spaces"
    )
    diag_map = {l: i for i, l in enumerate(sorted(df["diagnosis_group"].unique()))}
    assert list(diag_map.keys()) == diagnosis_labels, (
        "diagnosis label space mismatch between data and v3 metadata"
    )
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)

    # v3_base and v3-nurse share this split (same seed + same y_diag stratify),
    # so the resolver (built on the v3 train split) is held out for both.
    idx_all = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y_diag,
    )
    Xb_test = features_base.iloc[test_idx].reset_index(drop=True)
    Xn_test = features_nurse.iloc[test_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    true_cat_idx = df_test["diagnosis_group"].map(diag_map).to_numpy()
    print(f"  Test set: {len(df_test):,} stays (shared by v3_base and v3 with-nurse)")

    # 2. Shared resolver + query vectors.
    resolver = icdr.load_resolver(DOCTOR_V3_ICD_RESOLVER_DIR)
    alpha_blend = resolver["alpha"]
    tfidf = joblib.load(TRIAGE_V1_DIR / "tfidf_vectorizer.joblib")
    Q_test = icdr.vectorize_queries(df_test["complaint_text_norm"].tolist(), tfidf)
    variants = {"prevalence": 0.0, "cosine": 1.0, "blend": alpha_blend}
    print(f"  Resolver alpha (tuned) = {alpha_blend:.2f} | n_train={resolver['n_train']:,}")
    print("  (Stage-2 index is shared — its candidates + centroids do NOT depend on the")
    print("   diagnosis model; only Stage-1's predicted categories differ between models.)")

    # 3. Stage-2 ORACLE ceiling — model-INDEPENDENT (uses the TRUE category, so it
    #    is identical for v3_base and v3-nurse; reported once).
    print_section("STAGE-2 ORACLE CEILING (shared — within the true category)")
    for g in icdr.GRANULARITIES:
        gindex = resolver["granularities"][g]
        code_col = "rollup_code" if g == "rollup" else "icd_code"
        label = "3-char rollup (PRIMARY)" if g == "rollup" else "full code (secondary)"

        def _orow(name, alpha, _g=gindex, _c=code_col):
            if name == "__header__":
                return [("Oracle@5", 11), ("Oracle@10", 11)]
            o5, o10, _ = _oracle_recall(
                Q_test, df_test, true_cat_idx, _g, diagnosis_labels, _c, alpha)
            return (o5, o10)
        _print_recall_table(label, variants, _orow)

    # 4. END-TO-END per diagnosis model — this is where nurse data matters.
    models = [
        ("v3_base (pre-nurse)", DOCTOR_V3_BASE_DIR, Xb_test),
        ("v3 with-nurse", DOCTOR_V3_DIR, Xn_test),
    ]
    e2e_blend = {}             # (model_name, g) -> (union, flat, cond)
    nurse_proba = nurse_top5 = None
    for model_name, model_dir, X_test in models:
        model = joblib.load(model_dir / "diagnosis_model.joblib")
        proba = model.predict_proba(X_test)
        top5 = np.argsort(-proba, axis=1)[:, :TOP_CATS]
        if model_dir == DOCTOR_V3_DIR:
            nurse_proba, nurse_top5 = proba, top5
        cat_top1 = (np.argmax(proba, axis=1) == true_cat_idx).mean()
        cat_top5 = (top5 == true_cat_idx[:, None]).any(axis=1).mean()
        print_section(f"END-TO-END — {model_name}")
        print(f"  Stage-1 category recall: top-1={cat_top1:.4f}  top-5={cat_top5:.4f}")
        for g in icdr.GRANULARITIES:
            gindex = resolver["granularities"][g]
            code_col = "rollup_code" if g == "rollup" else "icd_code"
            label = "3-char rollup (PRIMARY)" if g == "rollup" else "full code (secondary)"

            def _erow(name, alpha, _g=gindex, _c=code_col, _p=proba, _t=top5, _gn=g):
                if name == "__header__":
                    return [("E2E union", 11), ("E2E flat10", 12), ("Cond@union", 12)]
                u, f, c, _ = _end_to_end_recall(
                    Q_test, df_test, _p, _t, _g, diagnosis_labels, _c, alpha,
                    true_cat_idx)
                if name == "blend":
                    e2e_blend[(model_name, _gn)] = (u, f, c)
                return (u, f, c)
            _print_recall_table(label, variants, _erow)

    # 5. Before/after-nurse summary (blend).
    print_section(f"BEFORE/AFTER NURSE — exact-ICD end-to-end lift (blend, alpha={alpha_blend:.2f})")
    base_name, nurse_name = models[0][0], models[1][0]
    for g in icdr.GRANULARITIES:
        label = "3-char rollup" if g == "rollup" else "full code"
        ub, fb, cb = e2e_blend[(base_name, g)]
        un, fn, cn = e2e_blend[(nurse_name, g)]
        print(f"\n  [{label}]  {'Metric':<14s} {'v3_base':>9s} {'v3_nurse':>9s} {'delta':>9s}")
        for mlabel, vb, vn in [("E2E union", ub, un),
                               ("E2E flat-10", fb, fn),
                               ("Cond@union", cb, cn)]:
            print(f"               {mlabel:<14s} {vb:>9.4f} {vn:>9.4f} {vn - vb:>+9.4f}")

    # 6. Spot-check (v3 with-nurse, blend, rollup).
    proba, top5_cat_idx = nurse_proba, nurse_top5
    print_section("SPOT-CHECK (v3 with-nurse, blend, rollup) — first 3 test stays")
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
