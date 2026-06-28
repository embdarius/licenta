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
from datetime import datetime
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
    DOCTOR_V3_ICD_RESOLVER_DIR, ARTIFACTS_DIR,
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
from proiect_licenta import icd_similarity as icds

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
                   weights_nopmh=None, subset_mask=None, graders=None):
    """Within the TRUE category: strict recall@5/@10 + graded@5/@10 per grader.

    Returns a dict ``{"r5", "r10", "scored", "graded": {name: (g5, g10)}}`` where
    the graded scores give partial credit (max similarity over the top-k) for
    clinically-close misses. ``graded@k >= r{k}`` always (sim of a code with
    itself is 1). Strict numbers are byte-identical to the pre-graded version.
    """
    hit1 = hit3 = hit5 = hit10 = scored = 0
    rr_sum = 0.0                              # sum of reciprocal ranks of the true code
    graders = graders or []
    gsum = {g.name: [0.0, 0.0] for g in graders}
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
        # Additive @1/@3 + MRR of the true code (full ranking). Vectorized over
        # this category's rows; strict @5/@10 below stay byte-identical to the
        # original per-row set membership check.
        ranked = codes_arr[order]
        tc_col = true_codes[rows]
        match = ranked == tc_col[:, None]
        hit1 += int(match[:, :1].any(axis=1).sum())
        hit3 += int(match[:, :3].any(axis=1).sum())
        has = match.any(axis=1)
        pos = match.argmax(axis=1)
        rr_sum += float(np.where(has, 1.0 / (pos + 1), 0.0).sum())
        for j, r in enumerate(rows):
            tc = true_codes[r]
            scored += 1
            if tc in set(top5[j].tolist()):
                hit5 += 1
            if tc in set(top10[j].tolist()):
                hit10 += 1
        for g in graders:
            sim_all = g.oracle_sim_all(rows, entry["codes"])
            gsum[g.name][0] += float(icds.graded_max_over_topk(sim_all, order, K_PER_CAT).sum())
            gsum[g.name][1] += float(icds.graded_max_over_topk(sim_all, order, K_FLAT).sum())
    if scored == 0:
        return {"r1": 0.0, "r3": 0.0, "r5": 0.0, "r10": 0.0, "mrr": 0.0,
                "scored": 0, "graded": {g.name: (0.0, 0.0) for g in graders}}
    return {
        "r1": hit1 / scored, "r3": hit3 / scored,
        "r5": hit5 / scored, "r10": hit10 / scored,
        "mrr": rr_sum / scored, "scored": scored,
        "graded": {n: (s[0] / scored, s[1] / scored) for n, s in gsum.items()},
    }


def _end_to_end_recall(Q_test, physio_test, pmh_test, has_pmh, df_test, proba,
                       top5_cat_idx, gindex, diagnosis_labels, code_col, weights,
                       true_cat_idx, weights_nopmh=None, subset_mask=None,
                       graders=None):
    """End-to-end recall: union(top-5 cats x top-5 codes) and flat top-10.

    Returns a dict with the strict union/flat/conditional recall + category
    accuracy, plus per-grader graded (union, flat, cond) — the max similarity of
    the true code to the codes in the predicted set, giving partial credit for
    near misses. The conditional recall restricts to rows whose true category is
    in the predicted top-5 (removes the Stage-1 ceiling). `subset_mask` restricts
    the final tally (e.g. to has-PMH rows).
    """
    n = len(df_test)
    graders = graders or []
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
    flat_rr_sum = 0.0                         # MRR of the true code over the flat ranking
    g_union = {g.name: 0.0 for g in graders}
    g_flat = {g.name: 0.0 for g in graders}
    g_cond = {g.name: 0.0 for g in graders}
    for r in range(n):
        if not keep[r]:
            continue
        tally += 1
        tc = true_codes[r]
        union_codes = union_sets[r]
        in_union = tc in union_codes
        # Full de-duplicated flat ranking (best score per code) for MRR; the
        # top-K_FLAT slice of it is the strict flat@10 set.
        ranked_codes, seen = [], set()
        for _, c in sorted(flat_lists[r], reverse=True):
            if c not in seen:
                seen.add(c)
                ranked_codes.append(c)
        flat_top = ranked_codes[:K_FLAT]
        in_flat = tc in set(flat_top)
        if tc in seen:
            flat_rr_sum += 1.0 / (ranked_codes.index(tc) + 1)
        union_hit += in_union
        flat_hit += in_flat
        if cat_correct[r]:
            cond_total += 1
            cond_union_hit += in_union
        for g in graders:
            um = g.row_max_sim(r, union_codes)
            g_union[g.name] += um
            g_flat[g.name] += g.row_max_sim(r, flat_top)
            if cat_correct[r]:
                g_cond[g.name] += um
    return {
        "union": union_hit / tally,
        "flat": flat_hit / tally,
        "flat_mrr": flat_rr_sum / tally,
        "cond": (cond_union_hit / cond_total if cond_total else 0.0),
        "cat_acc": float(cat_correct[keep].mean()),
        "graded": {
            g.name: (g_union[g.name] / tally, g_flat[g.name] / tally,
                     (g_cond[g.name] / cond_total if cond_total else 0.0))
            for g in graders
        },
    }


def _print_recall_table(label, variants, row_fn):
    """Print a small per-variant table. `row_fn(name, alpha)` -> tuple of floats."""
    cols = row_fn("__header__", None)  # list of (header, width)
    head = "".join(f"{h:>{w}s}" for h, w in cols)
    print(f"\n  [{label}]  {'Variant':<12s}{head}")
    for name, alpha in variants.items():
        vals = row_fn(name, alpha)
        body = "".join(f"{v:>{w}.4f}" for v, (_, w) in zip(vals, cols))
        print(f"               {name:<12s}{body}")


def _print_graded_table(label, variants, graders, colspecs, extract):
    """One graded sub-table per available grader (keeps rows narrow).

    `colspecs` is a list of (header, width); `extract(variant_name, grader_name)`
    returns a tuple of floats aligned with `colspecs`. Graded scores give partial
    credit for near misses, so they read >= the matching strict recall.
    """
    for g in graders:
        if not getattr(g, "available", True):
            continue
        head = "".join(f"{h:>{w}s}" for h, w in colspecs)
        print(f"\n  [{label} | graded:{g.name}]  {'Variant':<12s}{head}")
        for name in variants:
            vals = extract(name, g.name)
            body = "".join(f"{v:>{w}.4f}" for v, (_, w) in zip(vals, colspecs))
            print(f"               {name:<12s}{body}")


ORACLE_GRADED_COLS = [("Graded@5", 11), ("Graded@10", 11)]
E2E_GRADED_COLS = [("G-union", 11), ("G-flat10", 12), ("G-cond", 11)]


def _count_invariant_violations(results, tol=1e-9):
    """Count graded@k < strict recall@k cells across oracle + E2E (should be 0).

    A graded metric is the max similarity over the top-k; since a code is
    identical to itself (sim 1.0), any strict hit forces graded == 1 for that
    row, so graded@k can never fall below strict recall@k. A nonzero count here
    means the wiring is wrong.
    """
    v = 0
    for vmap in results["oracle"].values():
        for cell in vmap.values():
            for sk, gk in (("oracle@5", "graded@5"), ("oracle@10", "graded@10")):
                s = cell["strict"][sk]
                for gv in cell["graded"].values():
                    v += gv[gk] + tol < s
    for gmap in results["end_to_end"].values():
        for key, vmap in gmap.items():
            if key == "stage1_category_recall":
                continue
            for cell in vmap.values():
                for k in ("union", "flat", "cond"):
                    s = cell["strict"][k]
                    for gv in cell["graded"].values():
                        v += gv[k] + tol < s
    return int(v)


def run_icd_benchmark(out_path: "Path | None" = None) -> dict:
    """Run the full Stage-2 exact-ICD benchmark and return the structured results
    dict (also written to ``out_path``, defaulting to
    ``artifacts/benchmarks/icd_resolution_graded.json``). Importable so the
    tabular orchestrator can fold these numbers into one master JSON without a
    second data load."""
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

    # 2b. Graded near-miss metrics (evaluation-only; partial credit for clinically
    #     close misses). The strict recall below is byte-for-byte unchanged. Build
    #     ONE title set = every candidate title (both granularities) + the test
    #     rows' true titles, and fit one TF-IDF over the union so candidate and
    #     true vectors share a vocabulary space.
    def _granularity_code_titles(gname):
        m = {}
        for entry in resolver["granularities"][gname].values():
            for code, title in zip(entry["codes"], entry["titles"]):
                m.setdefault(str(code), title)
        return m
    code_titles = {gname: _granularity_code_titles(gname) for gname in icdr.GRANULARITIES}
    true_titles_raw = df_test["icd_title"].tolist()
    distinct_titles = {t for m in code_titles.values() for t in m.values()}
    distinct_titles.update(true_titles_raw)

    # Version per candidate code (for the ICD-tree engine's rollup/chapter keys),
    # taken from the full dataset by majority vote; defaults to ICD-10 if unseen.
    def _granularity_code_versions(gname):
        col = "rollup_code" if gname == "rollup" else "icd_code"
        ver_by_code = (df.groupby(col)["icd_version"]
                       .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
                       .astype(str).to_dict())
        return {code: ver_by_code.get(code, "10") for code in code_titles[gname]}
    code_versions = {gname: _granularity_code_versions(gname) for gname in icdr.GRANULARITIES}

    # Three graded engines (evaluation-only): TF-IDF title cosine (lexical),
    # Gemini-embedding title cosine (semantic, cached + offline after first build),
    # and ICD-tree hierarchical distance. Gemini self-skips if no key + no cache.
    tfidf_titles = icds.fit_title_tfidf(distinct_titles)
    gem_cache = icds.build_or_load_title_embeddings(
        distinct_titles, DOCTOR_V3_ICD_RESOLVER_DIR / "title_embeddings.joblib")
    graders = [
        icds.TitleGrader("tfidf", lambda ts: icds.tfidf_vectors(ts, tfidf_titles)),
        icds.TitleGrader("gemini", lambda ts: icds.gemini_vectors(ts, gem_cache)),
        icds.IcdTreeGrader(),
    ]
    active_graders = [gr for gr in graders if gr.available]

    def _prep_graders(gname):
        code_col = "rollup_code" if gname == "rollup" else "icd_code"
        true_codes = df_test[code_col].astype(str).tolist()
        # True side uses the true code's REPRESENTATIVE title (same map as the
        # candidates), so an exact-code hit scores cosine 1.0 and the invariant
        # graded@k >= strict recall@k holds. Codes absent from the index (never
        # candidates, so they can't strict-hit) fall back to the row's raw title.
        ctitles = code_titles[gname]
        true_titles = [ctitles.get(tc, raw)
                       for tc, raw in zip(true_codes, true_titles_raw)]
        ctx = {
            "code_to_title": ctitles,
            "code_to_version": code_versions[gname],
            "true_titles": true_titles,
            "true_codes": true_codes,
            "true_versions": df_test["icd_version"].astype(str).tolist(),
        }
        for gr in active_graders:
            gr.prepare(ctx)
    print(f"  Graded metrics: {', '.join(gr.name for gr in active_graders)} "
          f"| distinct titles: {len(distinct_titles):,}")

    # Accumulate every reported number into a structured results dict, dumped to
    # JSON at the end so the run is verifiable at any time (mirrors the other
    # artifacts/benchmarks/*.json result files).
    gem_dim = (len(next(iter(gem_cache.values()))) if gem_cache else None)
    results = {
        "_meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "purpose": ("Stage-2 exact-ICD resolution: strict exact-code recall@k "
                        "PLUS 3 evaluation-only GRADED near-miss metrics that give "
                        "partial credit for clinically-close misses. graded@k = max "
                        "similarity of the true code to the top-k predicted codes; "
                        "graded@k >= strict recall@k by construction."),
            "test_set": int(len(df_test)),
            "n_train": int(resolver["n_train"]),
            "split": {"test_size": TEST_SIZE, "random_state": SPLIT_SEED},
            "granularities": list(icdr.GRANULARITIES),
            "variants": {k: v for k, v in variants.items()},
            "graded_engines": [gr.name for gr in active_graders],
            "graded_engine_detail": {
                "tfidf": "lexical cosine of TF-IDF vectors of the ICD titles "
                         "(single-char tokens kept so 'hepatitis A' vs 'B' differ)",
                "gemini": {"model": icds.GEMINI_EMBED_MODEL, "dim": gem_dim,
                           "cache": str(DOCTOR_V3_ICD_RESOLVER_DIR / "title_embeddings.joblib"),
                           "n_cached": (len(gem_cache) if gem_cache else 0),
                           "note": "semantic cosine of cached Gemini title embeddings; "
                                   "high similarity floor -> deltas/ordering matter more "
                                   "than absolute values"},
                "tree": {"w_rollup": icds.W_ROLLUP, "w_chapter": icds.W_CHAPTER,
                         "note": "exact code 1.0 / shared 3-char rollup w_rollup / "
                                 "shared ICD chapter w_chapter / else 0"},
            },
            "distinct_titles": int(len(distinct_titles)),
            "k_per_cat": K_PER_CAT, "k_flat": K_FLAT, "top_cats": TOP_CATS,
        },
        "oracle": {}, "end_to_end": {},
        "before_after_nurse": {}, "pmh_gated": {},
    }

    def _graded_pair(gradedmap, keys):
        return {e: dict(zip(keys, v)) for e, v in gradedmap.items()}

    # 3. Stage-2 ORACLE ceiling — model-INDEPENDENT (uses the TRUE category, so it
    #    is identical for v3_base and v3-nurse; reported once).
    print_section("STAGE-2 ORACLE CEILING (shared — within the true category)")
    for g in icdr.GRANULARITIES:
        gindex = resolver["granularities"][g]
        code_col = "rollup_code" if g == "rollup" else "icd_code"
        label = "3-char rollup (PRIMARY)" if g == "rollup" else "full code (secondary)"
        _prep_graders(g)
        omemo = {}

        def _ores(name, w, _g=gindex, _c=code_col, _m=omemo):
            if name not in _m:
                _m[name] = _oracle_recall(
                    Q_test, physio_test, None, None, df_test, true_cat_idx, _g,
                    diagnosis_labels, _c, w, graders=active_graders)
            return _m[name]

        def _orow(name, w):
            if name == "__header__":
                return [("Oracle@5", 11), ("Oracle@10", 11)]
            r = _ores(name, w)
            return (r["r5"], r["r10"])
        _print_recall_table(label, variants, _orow)
        _print_graded_table(
            label, variants, active_graders, ORACLE_GRADED_COLS,
            lambda nm, gn: _ores(nm, variants[nm])["graded"][gn])
        results["oracle"][g] = {
            name: {
                "strict": {
                    "oracle@1": omemo[name]["r1"], "oracle@3": omemo[name]["r3"],
                    "oracle@5": omemo[name]["r5"], "oracle@10": omemo[name]["r10"],
                    "mrr": omemo[name]["mrr"],
                },
                "graded": _graded_pair(omemo[name]["graded"], ("graded@5", "graded@10")),
            } for name in variants
        }

    # 4. END-TO-END per diagnosis model — this is where nurse data matters.
    models = [
        ("v3_base (pre-nurse)", DOCTOR_V3_BASE_DIR, Xb_test),
        ("v3 with-nurse", DOCTOR_V3_DIR, Xn_test),
    ]
    e2e_blend = {}             # (model_name, g) -> full result dict (incl. graded)
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
            _prep_graders(g)
            ememo = {}

            def _eres(name, w, _g=gindex, _c=code_col, _p=proba, _t=top5, _m=ememo):
                if name not in _m:
                    _m[name] = _end_to_end_recall(
                        Q_test, physio_test, None, None, df_test, _p, _t, _g,
                        diagnosis_labels, _c, w, true_cat_idx, graders=active_graders)
                return _m[name]

            def _erow(name, w, _gn=g):
                if name == "__header__":
                    return [("E2E union", 11), ("E2E flat10", 12), ("Cond@union", 12)]
                r = _eres(name, w)
                if name == HEADLINE:
                    e2e_blend[(model_name, _gn)] = r
                return (r["union"], r["flat"], r["cond"])
            _print_recall_table(label, variants, _erow)
            _print_graded_table(
                label, variants, active_graders, E2E_GRADED_COLS,
                lambda nm, gn: _eres(nm, variants[nm])["graded"][gn])
            mres = results["end_to_end"].setdefault(model_name, {
                "stage1_category_recall": {"top1": float(cat_top1), "top5": float(cat_top5)}})
            mres[g] = {
                name: {
                    "strict": {k: ememo[name][k]
                               for k in ("union", "flat", "flat_mrr", "cond", "cat_acc")},
                    "graded": _graded_pair(ememo[name]["graded"], ("union", "flat", "cond")),
                } for name in variants
            }

    # 5. Before/after-nurse summary (headline = blend+vitals).
    print_section(f"BEFORE/AFTER NURSE — exact-ICD end-to-end lift ({HEADLINE})")
    base_name, nurse_name = models[0][0], models[1][0]
    for g in icdr.GRANULARITIES:
        label = "3-char rollup" if g == "rollup" else "full code"
        rb = e2e_blend[(base_name, g)]
        rn = e2e_blend[(nurse_name, g)]
        print(f"\n  [{label}]  {'Metric':<18s} {'v3_base':>9s} {'v3_nurse':>9s} {'delta':>9s}")
        for mlabel, vb, vn in [("E2E union", rb["union"], rn["union"]),
                               ("E2E flat-10", rb["flat"], rn["flat"]),
                               ("Cond@union", rb["cond"], rn["cond"])]:
            print(f"               {mlabel:<18s} {vb:>9.4f} {vn:>9.4f} {vn - vb:>+9.4f}")
        for gr in active_graders:
            gb, gn_ = rb["graded"][gr.name], rn["graded"][gr.name]
            for mlabel, vb, vn in [(f"G-union:{gr.name}", gb[0], gn_[0]),
                                   (f"G-flat10:{gr.name}", gb[1], gn_[1]),
                                   (f"G-cond:{gr.name}", gb[2], gn_[2])]:
                print(f"               {mlabel:<18s} {vb:>9.4f} {vn:>9.4f} {vn - vb:>+9.4f}")

        def _ba(vb, vn):
            return {"v3_base": vb, "v3_nurse": vn, "delta": vn - vb}
        results["before_after_nurse"][g] = {
            "strict": {m: _ba(rb[m], rn[m]) for m in ("union", "flat", "cond")},
            "graded": {
                gr.name: {m: _ba(rb["graded"][gr.name][i], rn["graded"][gr.name][i])
                          for i, m in enumerate(("union", "flat", "cond"))}
                for gr in active_graders
            },
        }

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
        results["_meta"]["pmh_weights"] = {"no_pmh": w_full, "has_pmh": w_pmh}
        for g in icdr.GRANULARITIES:
            gindex = resolver["granularities"][g]
            code_col = "rollup_code" if g == "rollup" else "icd_code"
            label = "3-char rollup" if g == "rollup" else "full code"
            _prep_graders(g)
            pmh_g = {"oracle@5": {}, "E2E union": {}}
            print(f"\n  [{label}]  {'subset':<10s} {'metric':<14s} "
                  f"{'baseline':>10s} {'+PMH':>10s} {'delta':>9s}")
            for sub_name, mask in [("all", None), ("has-PMH", has_pmh), ("no-PMH", ~has_pmh)]:
                rb = _oracle_recall(
                    Q_test, physio_test, None, None, df_test, true_cat_idx, gindex,
                    diagnosis_labels, code_col, w_full, subset_mask=mask,
                    graders=active_graders)
                rg = _oracle_recall(
                    Q_test, physio_test, pmh_test, has_pmh, df_test, true_cat_idx,
                    gindex, diagnosis_labels, code_col, w_pmh,
                    weights_nopmh=w_full, subset_mask=mask, graders=active_graders)
                print(f"             {sub_name:<10s} {'oracle@5':<14s} "
                      f"{rb['r5']:>10.4f} {rg['r5']:>10.4f} {rg['r5'] - rb['r5']:>+9.4f}")
                for gr in active_graders:
                    b, gg = rb["graded"][gr.name][0], rg["graded"][gr.name][0]
                    print(f"             {sub_name:<10s} {'g@5:' + gr.name:<14s} "
                          f"{b:>10.4f} {gg:>10.4f} {gg - b:>+9.4f}")
                pmh_g["oracle@5"][sub_name] = {
                    "baseline": rb["r5"], "+PMH": rg["r5"], "delta": rg["r5"] - rb["r5"],
                    "graded": {gr.name: {
                        "baseline": rb["graded"][gr.name][0], "+PMH": rg["graded"][gr.name][0],
                        "delta": rg["graded"][gr.name][0] - rb["graded"][gr.name][0]}
                        for gr in active_graders},
                }
            for sub_name, mask in [("all", None), ("has-PMH", has_pmh)]:
                rb = _end_to_end_recall(
                    Q_test, physio_test, None, None, df_test, nurse_proba,
                    nurse_top5, gindex, diagnosis_labels, code_col, w_full,
                    true_cat_idx, subset_mask=mask, graders=active_graders)
                rg = _end_to_end_recall(
                    Q_test, physio_test, pmh_test, has_pmh, df_test, nurse_proba,
                    nurse_top5, gindex, diagnosis_labels, code_col, w_pmh,
                    true_cat_idx, weights_nopmh=w_full, subset_mask=mask,
                    graders=active_graders)
                print(f"             {sub_name:<10s} {'E2E union':<14s} "
                      f"{rb['union']:>10.4f} {rg['union']:>10.4f} {rg['union'] - rb['union']:>+9.4f}")
                for gr in active_graders:
                    b, gg = rb["graded"][gr.name][0], rg["graded"][gr.name][0]
                    print(f"             {sub_name:<10s} {'Gunion:' + gr.name:<14s} "
                          f"{b:>10.4f} {gg:>10.4f} {gg - b:>+9.4f}")
                pmh_g["E2E union"][sub_name] = {
                    "baseline": rb["union"], "+PMH": rg["union"], "delta": rg["union"] - rb["union"],
                    "graded": {gr.name: {
                        "baseline": rb["graded"][gr.name][0], "+PMH": rg["graded"][gr.name][0],
                        "delta": rg["graded"][gr.name][0] - rb["graded"][gr.name][0]}
                        for gr in active_graders},
                }
            results["pmh_gated"][g] = pmh_g

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

    # 7. Persist the full, detailed results to JSON for accurate later verification.
    results["_meta"]["invariant_graded_ge_strict_violations"] = _count_invariant_violations(results)
    if out_path is None:
        out_path = ARTIFACTS_DIR / "benchmarks" / "icd_resolution_graded.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print_section("RESULTS SAVED")
    print(f"  Detailed results -> {out_path}")
    print(f"  Invariant graded@k >= strict recall@k — violations: "
          f"{results['_meta']['invariant_graded_ge_strict_violations']} (expected 0)")

    print("\n" + "#" * 72)
    print("  BENCHMARK COMPLETE")
    print("#" * 72 + "\n")
    return results


def main():
    run_icd_benchmark()


if __name__ == "__main__":
    main()
