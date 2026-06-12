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


def _score_rows(Q_rows, physio_rows, pmh_rows, has_pmh_rows, entry,
                weights, weights_nopmh=None):
    """Blended scores for a row subset against one category's candidates.

    Unifies all variants via `gated_score`: text cosine always, vitals when a
    weight uses it, and the gated PMH term when `weights['pmh'] > 0` and PMH
    data is present (patients with `has_pmh_rows` get `weights`, the rest get
    `weights_nopmh`). Returns (n_rows, m)."""
    text_cos = np.asarray(Q_rows @ entry["centroids"].T, dtype=np.float32)
    wn = weights_nopmh if weights_nopmh is not None else weights
    use_vit = weights.get("vit", 0.0) > 0 or wn.get("vit", 0.0) > 0
    vit = None
    if use_vit and physio_rows is not None and entry.get("vital_centroids") is not None:
        vit = icdr._neg_euclidean(physio_rows, entry["vital_centroids"])
    pmh_sim = None
    if weights.get("pmh", 0.0) > 0 and pmh_rows is not None \
            and entry.get("pmh_centroids") is not None:
        pmh_sim = icdr._neg_euclidean(pmh_rows, entry["pmh_centroids"])
    hp = has_pmh_rows if has_pmh_rows is not None else False
    return icdr.gated_score(text_cos, vit, pmh_sim, entry["prevalence"],
                            weights, wn, hp)


def _subset_rows(rows, physio_test, pmh_test, has_pmh, subset_mask):
    """Filter `rows` by `subset_mask` and return aligned physio/pmh/has_pmh."""
    if subset_mask is not None:
        rows = rows[subset_mask[rows]]
    physio_rows = physio_test[rows] if physio_test is not None else None
    pmh_rows = pmh_test[rows] if pmh_test is not None else None
    hp_rows = has_pmh[rows] if has_pmh is not None else None
    return rows, physio_rows, pmh_rows, hp_rows


def _oracle_recall(Q_test, physio_test, pmh_test, has_pmh, df_test, true_cat_idx,
                   gindex, diagnosis_labels, code_col, weights,
                   weights_nopmh=None, subset_mask=None):
    """Within the TRUE category, recall@5 / recall@10 of the true code."""
    hit5 = hit10 = scored = 0
    true_codes = df_test[code_col].to_numpy()
    for c_idx, cat in enumerate(diagnosis_labels):
        entry = gindex.get(cat)
        if entry is None:
            continue
        rows = np.where(true_cat_idx == c_idx)[0]
        rows, physio_rows, pmh_rows, hp_rows = _subset_rows(
            rows, physio_test, pmh_test, has_pmh, subset_mask)
        if len(rows) == 0:
            continue
        score = _score_rows(Q_test[rows], physio_rows, pmh_rows, hp_rows,
                            entry, weights, weights_nopmh)
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
    if scored == 0:
        return 0.0, 0.0, 0
    return hit5 / scored, hit10 / scored, scored


def _end_to_end_recall(Q_test, physio_test, pmh_test, has_pmh, df_test, proba,
                       top5_cat_idx, gindex, diagnosis_labels, code_col, weights,
                       true_cat_idx, weights_nopmh=None, subset_mask=None):
    """End-to-end recall: union(top-5 cats x top-5 codes) and flat top-10.

    Also returns the conditional recall (rows whose true category is in the
    predicted top-5), which removes the Stage-1 category-recall ceiling.
    `subset_mask` restricts the final tally (e.g. to has-PMH rows).
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
        physio_rows = physio_test[rows] if physio_test is not None else None
        pmh_rows = pmh_test[rows] if pmh_test is not None else None
        hp_rows = has_pmh[rows] if has_pmh is not None else None
        score = _score_rows(Q_test[rows], physio_rows, pmh_rows, hp_rows,
                            entry, weights, weights_nopmh)
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
    keep = np.ones(n, dtype=bool) if subset_mask is None else subset_mask
    union_hit = flat_hit = tally = 0
    cond_union_hit = cond_total = 0
    for r in range(n):
        if not keep[r]:
            continue
        tally += 1
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
        union_hit / tally,
        flat_hit / tally,
        (cond_union_hit / cond_total if cond_total else 0.0),
        cat_correct[keep].mean(),
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

    # 2. Shared resolver + query vectors + standardized physiology.
    resolver = icdr.load_resolver(DOCTOR_V3_ICD_RESOLVER_DIR)
    tfidf = joblib.load(TRIAGE_V1_DIR / "tfidf_vectorizer.joblib")
    Q_test = icdr.vectorize_queries(df_test["complaint_text_norm"].tolist(), tfidf)
    standardizer = resolver.get("standardizer")
    physio_test = (icdr.physio_matrix(df_test, standardizer)
                   if standardizer is not None else None)
    # PMH (gated history term): standardized PMH vectors + has-history mask.
    pmh_standardizer = resolver.get("pmh_standardizer")
    pmh_test = (icdr.physio_matrix(df_test, pmh_standardizer)
                if pmh_standardizer is not None else None)
    has_pmh = (df_test["no_history"] == 0).to_numpy() if "no_history" in df_test else None

    # Variants as 3-term weight dicts {text, vit, prev}. "blend" = tuned text+
    # prevalence (no vitals); "blend+vitals" = tuned 3-term (the headline).
    w_full = resolver["weights"]
    w_novit = resolver.get("weights_novit") or {
        "text": resolver["alpha"], "vit": 0.0, "prev": 1.0 - resolver["alpha"]}
    variants = {
        "prevalence":   {"text": 0.0, "vit": 0.0, "prev": 1.0},
        "cosine":       {"text": 1.0, "vit": 0.0, "prev": 0.0},
        "blend":        w_novit,
        "blend+vitals": w_full,
    }
    HEADLINE = "blend+vitals"
    print(f"  n_train={resolver['n_train']:,} | tuned weights (text/vit/prev): "
          f"blend={w_novit['text']:.1f}/{w_novit['vit']:.1f}/{w_novit['prev']:.1f}  "
          f"blend+vitals={w_full['text']:.1f}/{w_full['vit']:.1f}/{w_full['prev']:.1f}")
    print("  (Stage-2 index is shared — candidates/centroids do NOT depend on the diagnosis")
    print("   model; vitals are the patient's snapshot triage vitals, z-scored.)")

    # 3. Stage-2 ORACLE ceiling — model-INDEPENDENT (uses the TRUE category, so it
    #    is identical for v3_base and v3-nurse; reported once).
    print_section("STAGE-2 ORACLE CEILING (shared — within the true category)")
    for g in icdr.GRANULARITIES:
        gindex = resolver["granularities"][g]
        code_col = "rollup_code" if g == "rollup" else "icd_code"
        label = "3-char rollup (PRIMARY)" if g == "rollup" else "full code (secondary)"

        def _orow(name, w, _g=gindex, _c=code_col):
            if name == "__header__":
                return [("Oracle@5", 11), ("Oracle@10", 11)]
            o5, o10, _ = _oracle_recall(
                Q_test, physio_test, None, None, df_test, true_cat_idx, _g,
                diagnosis_labels, _c, w)
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

            def _erow(name, w, _g=gindex, _c=code_col, _p=proba, _t=top5, _gn=g):
                if name == "__header__":
                    return [("E2E union", 11), ("E2E flat10", 12), ("Cond@union", 12)]
                u, f, c, _ = _end_to_end_recall(
                    Q_test, physio_test, None, None, df_test, _p, _t, _g,
                    diagnosis_labels, _c, w, true_cat_idx)
                if name == HEADLINE:
                    e2e_blend[(model_name, _gn)] = (u, f, c)
                return (u, f, c)
            _print_recall_table(label, variants, _erow)

    # 5. Before/after-nurse summary (headline = blend+vitals).
    print_section(f"BEFORE/AFTER NURSE — exact-ICD end-to-end lift ({HEADLINE})")
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

    # 5b. Gated PMH term — only patients WITH prior history get the 4th term.
    w_pmh = resolver.get("weights_pmh")
    if w_pmh is not None and pmh_test is not None and has_pmh is not None:
        n_has, n_no = int(has_pmh.sum()), int((~has_pmh).sum())
        print_section(f"PMH-GATED TERM — has-PMH {n_has:,} / no-PMH {n_no:,} "
                      f"({100*has_pmh.mean():.0f}% have history)")
        print(f"  no-PMH weights (text/vit/prev) = "
              f"{w_full['text']:.1f}/{w_full['vit']:.1f}/{w_full['prev']:.1f}  |  "
              f"has-PMH (text/vit/pmh/prev) = {w_pmh['text']:.1f}/{w_pmh['vit']:.1f}/"
              f"{w_pmh['pmh']:.1f}/{w_pmh['prev']:.1f}")
        print("  (baseline = blend+vitals; +PMH = gated 4-term. no-PMH subset should be ~unchanged.)")
        for g in icdr.GRANULARITIES:
            gindex = resolver["granularities"][g]
            code_col = "rollup_code" if g == "rollup" else "icd_code"
            label = "3-char rollup" if g == "rollup" else "full code"
            print(f"\n  [{label}]  {'subset':<10s} {'metric':<10s} "
                  f"{'baseline':>10s} {'+PMH':>10s} {'delta':>9s}")
            for sub_name, mask in [("all", None), ("has-PMH", has_pmh), ("no-PMH", ~has_pmh)]:
                o5b, _, _ = _oracle_recall(
                    Q_test, physio_test, None, None, df_test, true_cat_idx, gindex,
                    diagnosis_labels, code_col, w_full, subset_mask=mask)
                o5g, _, _ = _oracle_recall(
                    Q_test, physio_test, pmh_test, has_pmh, df_test, true_cat_idx,
                    gindex, diagnosis_labels, code_col, w_pmh,
                    weights_nopmh=w_full, subset_mask=mask)
                print(f"             {sub_name:<10s} {'oracle@5':<10s} "
                      f"{o5b:>10.4f} {o5g:>10.4f} {o5g - o5b:>+9.4f}")
            for sub_name, mask in [("all", None), ("has-PMH", has_pmh)]:
                ub, _, _, _ = _end_to_end_recall(
                    Q_test, physio_test, None, None, df_test, nurse_proba,
                    nurse_top5, gindex, diagnosis_labels, code_col, w_full,
                    true_cat_idx, subset_mask=mask)
                ug, _, _, _ = _end_to_end_recall(
                    Q_test, physio_test, pmh_test, has_pmh, df_test, nurse_proba,
                    nurse_top5, gindex, diagnosis_labels, code_col, w_pmh,
                    true_cat_idx, weights_nopmh=w_full, subset_mask=mask)
                print(f"             {sub_name:<10s} {'E2E union':<10s} "
                      f"{ub:>10.4f} {ug:>10.4f} {ug - ub:>+9.4f}")

    # 6. Spot-check (v3 with-nurse, blend+vitals, rollup).
    proba, top5_cat_idx = nurse_proba, nurse_top5
    print_section("SPOT-CHECK (v3 with-nurse, blend+vitals, rollup) — first 3 test stays")
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
        physio_q = physio_test[r] if physio_test is not None else None
        res = icdr.resolve_exact_diagnoses_v2(
            cat_probs, q, physio_q, gindex, w_full,
            k_per_cat=K_PER_CAT, k_flat=K_FLAT,
        )
        active_flags = [c for c in icdr.PHYSIO_FLAGS if float(df_test[c].iloc[r]) == 1]
        print(f"\n  Complaint: {complaint}  | flags: {active_flags or ['none']}")
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
