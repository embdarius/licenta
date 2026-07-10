"""Offline builder for the Stage-2 exact-ICD resolver (Doctor v3).

Builds per-category candidate indices (prototype centroids + prevalence priors)
from the doctor v3 training split only, at two granularities: 3-char rollup and
full code. Each candidate also carries a vitals centroid. The 3-term blend
weights (text cosine / vitals / prevalence) are grid-searched on a held-out slice
using the oracle-category top-5 rollup hit rate, then the final indices are
rebuilt on the full training split and saved (with the legacy 2-term alpha kept
for the text+prevalence path). Offline/benchmark tooling, not part of the live
crew. Run with `uv run train_icd_resolver`.
"""
import argparse
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from proiect_licenta.paths import (
    TRIAGE_V1_DIR, TRIAGE_CSV, EDSTAYS_CSV, PATIENTS_CSV, SERVICES_CSV,
    DIAGNOSIS_CSV, VITALSIGN_CSV, DIAGNOSES_ICD_CSV, ADMISSIONS_CSV,
    DISCHARGE_NOTES_CSV, DERIVED_DIR, DOCTOR_V3_ICD_RESOLVER_DIR,
)
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.training.train_doctor import (
    DIAGNOSIS_GROUP_MAP, SERVICE_GROUP_MAP, CATCH_ALL_LABEL,
)
from proiect_licenta.training.train_nurse import _clean_vitals
from proiect_licenta.training.train_nurse_v3 import (
    _aggregate_vitalsigns, _fill_longitudinal_vitals, LONG_VITAL_FEATURE_COLS,
)
from proiect_licenta.pmh_vocab import PMH_CATEGORIES
from proiect_licenta.pmh_features import aggregate_pmh, fill_missing_pmh_columns
from proiect_licenta import icd_resolution as icdr

# PMH features for the gated history term. The 13 prior-diagnosis-category
# flags + the same-complaint Jaccard are the directly discriminative signals
# for the *exact* recurrent code; the count/days numerics (recurrence/acuity,
# and the 9999 "no prior" sentinel) are excluded to avoid sentinel pollution of
# the z-score. Centroids are built from has-history rows only.
PMH_RESOLVER_COLS = [f"pmh_{c}" for c in PMH_CATEGORIES] + ["same_complaint_as_prior"]
PMH_CACHE = DERIVED_DIR / "icd_resolver_pmh.parquet"

# Physiology features for the vitals-conditioned term. Snapshot vitals + flags
# (from triage.csv) PLUS the genuinely-new longitudinal signals that the
# snapshot lacks: cardiac rhythm (one-hot buckets + rhythm_irregular) and the
# per-vital trajectory delta. min/max/last and abnormal-reading counts are
# excluded - they mostly triplicate the snapshot value (a snapshot-only patient
# has min==max==last==snapshot), adding redundancy without new information.
_LONG_RHYTHM_DELTA = [c for c in LONG_VITAL_FEATURE_COLS
                      if c.endswith("_delta") or c.startswith("rhythm_")]
PHYSIO_COLS_V2 = icdr.PHYSIO_CONT + icdr.PHYSIO_FLAGS + _LONG_RHYTHM_DELTA

# Same split parameters as train_nurse_v3.main (must match exactly so the test
# split stays held out from the resolver).
TEST_SIZE = 0.2
SPLIT_SEED = 42

TUNE_VAL_FRACTION = 0.10  # held-out slice of TRAIN for weight tuning
TUNE_K = 5                # oracle-category top-5 rollup hit rate


def _weight_simplex(step: int = 10):
    """Enumerate (w_text, w_vit, w_prev) on the simplex in 1/step increments."""
    grid = []
    for i in range(step + 1):
        for j in range(step + 1 - i):
            wt, wv = i / step, j / step
            grid.append({"text": round(wt, 3), "vit": round(wv, 3),
                         "prev": round(1.0 - wt - wv, 3)})
    return grid


WEIGHT_GRID = _weight_simplex(10)  # 66 (text, vit, prev) combinations


def _weight_simplex4(step: int = 10):
    """Enumerate (w_text, w_vit, w_pmh, w_prev) on the simplex in 1/step steps."""
    grid = []
    for i in range(step + 1):
        for j in range(step + 1 - i):
            for k in range(step + 1 - i - j):
                wt, wv, wm = i / step, j / step, k / step
                grid.append({"text": round(wt, 3), "vit": round(wv, 3),
                             "pmh": round(wm, 3),
                             "prev": round(1.0 - wt - wv - wm, 3)})
    return grid


WEIGHT_GRID4 = _weight_simplex4(10)  # 286 (text, vit, pmh, prev) combinations


# Data assembly
# The resolver only needs each train-split stay's complaint text + ICD code +
# diagnosis group. It does NOT need the snapshot/longitudinal vitals, the
# medication flags, or the PMH features - and building those is where
# train_nurse_v3.load_and_clean_data spends ~all of its time (the uncached
# 3.3 GB discharge.csv parse + the vitalsign.csv per-stay loop).
#
# Crucially, none of that slow data changes which rows survive: medrecon,
# vitalsign and PMH are all LEFT-joined AFTER the row set is fixed, so they
# never drop or reorder rows. The row set + order (hence the train/test split)
# is determined entirely by the triage/edstays/diagnosis/services merges and
# the dropna/catch-all/acuity filters below. `_load_light` reproduces exactly
# those row-determining steps (mirroring train_nurse_v3.load_and_clean_data),
# so the split is identical with no leakage - at a fraction of the cost.
#
# Run `uv run train_icd_resolver --verify` to assert (once) that the light
# loader's row order matches the full v3 loader's, then use the fast default.

def _load_light() -> pd.DataFrame:
    """Fast loader: reproduce the v3 row set with string-safe ICD codes.

    Mirrors the row-affecting steps of train_nurse_v3.load_and_clean_data up to
    the acuity filter (the last row-dropping op), skipping medrecon / vitalsign
    / PMH (left joins that preserve rows). Returns a DataFrame with stay_id,
    chiefcomplaint, diagnosis_group, icd_code, icd_version, icd_title,
    rollup_code, complaint_text_norm.
    """
    triage = pd.read_csv(TRIAGE_CSV)
    edstays = pd.read_csv(
        EDSTAYS_CSV,
        usecols=["subject_id", "stay_id", "hadm_id", "intime", "gender",
                 "arrival_transport", "disposition"],
    )
    patients = pd.read_csv(
        PATIENTS_CSV, usecols=["subject_id", "anchor_age", "anchor_year"],
    )
    # String-safe primary diagnosis (preserves ICD-9 leading zeros + version).
    diag = pd.read_csv(
        DIAGNOSIS_CSV,
        dtype={"icd_code": str, "icd_version": str, "icd_title": str},
    )
    diag = diag[diag["seq_num"] == 1][
        ["stay_id", "category", "icd_code", "icd_version", "icd_title"]
    ].drop_duplicates("stay_id")
    services = pd.read_csv(SERVICES_CSV)
    services["transfertime"] = pd.to_datetime(services["transfertime"])
    services_first = (
        services.sort_values("transfertime")
        .groupby("hadm_id").first().reset_index()[["hadm_id", "curr_service"]]
    )

    admitted = edstays[edstays["disposition"] == "ADMITTED"].copy()
    df = triage.merge(admitted, on=["subject_id", "stay_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="left")
    df = df.merge(diag, on="stay_id", how="inner")
    df = df.merge(services_first, on="hadm_id", how="inner")

    # Row-dropping filters (identical to the v3 loader)
    df = df.dropna(subset=["chiefcomplaint", "category", "curr_service"])
    df = df[df["chiefcomplaint"].str.strip() != ""]
    df["diagnosis_group"] = df["category"].map(DIAGNOSIS_GROUP_MAP).fillna("Other")
    df["service_group"] = df["curr_service"].map(SERVICE_GROUP_MAP).fillna("OTHER")
    df = df[df["diagnosis_group"] != CATCH_ALL_LABEL].reset_index(drop=True)
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    df = df[df["acuity"].between(1, 5)].reset_index(drop=True)

    # Derived columns the resolver needs
    df["icd_code"] = df["icd_code"].fillna("").str.strip().str.upper()
    df["icd_version"] = df["icd_version"].fillna("").str.strip()
    df["icd_title"] = df["icd_title"].fillna("")
    df["rollup_code"] = [
        icdr.rollup_icd(c, v) for c, v in zip(df["icd_code"], df["icd_version"])
    ]
    df["complaint_text_norm"] = df["chiefcomplaint"].apply(normalize_complaint_text)

    # Snapshot vitals + abnormal flags (from triage.csv) for the vitals term.
    # `_clean_vitals` clips, median-imputes, and adds the 7 flags, all in-place.
    _clean_vitals(df)

    # Longitudinal vitals + cardiac rhythm from vitalsign.csv (over the same
    # [intime, intime+4h] window the v3 model uses). This is the one heavy add
    # vs the pure light loader (~a few minutes) - still far cheaper than the
    # full v3 loader, which also parses the 3.3 GB discharge.csv for PMH (which
    # the resolver does not need). `_fill_longitudinal_vitals` falls back to the
    # snapshot for stays with no readings, mirroring inference.
    df["intime"] = pd.to_datetime(df["intime"])
    long_vitals = _aggregate_vitalsigns(df[["stay_id", "intime"]], VITALSIGN_CSV)
    df = df.merge(long_vitals, on="stay_id", how="left")
    _fill_longitudinal_vitals(df)

    # PMH features (prior-diagnosis flags + same-complaint Jaccard) for the
    # gated history term. This is the heavy step (3.3 GB discharge.csv parse),
    # so the per-stay output is cached to parquet and reused across builds.
    df = _attach_pmh(df, edstays)
    return df


def _attach_pmh(df: pd.DataFrame, edstays_full: pd.DataFrame) -> pd.DataFrame:
    """Attach PMH_FEATURE_COLS (incl. no_history) to df, with a parquet cache.

    Reuses the vetted, leakage-safe `pmh_features.aggregate_pmh` verbatim (the
    `prior_admittime < intime` guard lives inside it). The per-stay result is
    cached on first run; subsequent builds skip the discharge.csv parse.
    """
    need = set(df["stay_id"].astype(int))
    pmh_df = None
    if PMH_CACHE.exists():
        cached = pd.read_parquet(PMH_CACHE)
        if need.issubset(set(cached["stay_id"].astype(int))):
            pmh_df = cached[cached["stay_id"].astype(int).isin(need)].copy()
            print(f"  [PMH cache] hit - {len(pmh_df):,} stays from {PMH_CACHE.name}")
    if pmh_df is None:
        print("  [PMH cache] miss - computing PMH (heavy discharge.csv parse)...")
        pmh_df = aggregate_pmh(
            stays_df=df[["stay_id", "subject_id", "intime", "complaint_text_norm"]],
            edstays_full=edstays_full,
            diagnoses_icd_csv_path=DIAGNOSES_ICD_CSV,
            admissions_csv_path=ADMISSIONS_CSV,
            discharge_csv_path=DISCHARGE_NOTES_CSV,
            diagnosis_csv_path=DIAGNOSIS_CSV,
            diagnosis_group_map=DIAGNOSIS_GROUP_MAP,
            catch_all_label=CATCH_ALL_LABEL,
        )
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        pmh_df.to_parquet(PMH_CACHE, index=False)
        print(f"  [PMH cache] saved {len(pmh_df):,} stays -> {PMH_CACHE.name}")
    df = df.merge(pmh_df, on="stay_id", how="left")
    fill_missing_pmh_columns(df)
    return df


def _verify_against_full(light_df: pd.DataFrame) -> None:
    """Assert the light loader's row order matches the full v3 loader's.

    Opt-in (`--verify`) one-time check: if the stay_id sequence matches, the
    train/test split is provably identical (the split depends only on row order
    + count). Heavy - runs the full v3 loader once.
    """
    from proiect_licenta.training.train_nurse_v3 import load_and_clean_data
    print("\n  [--verify] Running the FULL v3 loader to compare row order...")
    full_df = load_and_clean_data()
    a = light_df["stay_id"].tolist()
    b = full_df["stay_id"].tolist()
    if a == b:
        print(f"  [--verify] PASS - identical row order ({len(a):,} rows). "
              f"Light loader split == v3 split.")
    else:
        raise AssertionError(
            f"[--verify] FAIL - row order differs (light={len(a):,}, "
            f"full={len(b):,}). The light loader has drifted from "
            f"train_nurse_v3.load_and_clean_data; fix before trusting the split."
        )


def _train_rows(verify: bool = False) -> tuple[pd.DataFrame, int]:
    """Reproduce the v3 train/test split, return the TRAIN rows + test size.

    The split partition depends only on (n, test_size, random_state, stratify),
    so splitting np.arange(n) with the identical y_diag stratify reproduces the
    exact partition train_nurse_v3.main used - no feature matrix needed here.
    """
    df = _load_light()
    if verify:
        _verify_against_full(df)

    # Reproduce the train_nurse_v3 label encoding + stratify exactly.
    diagnosis_labels = sorted(df["diagnosis_group"].unique())
    diag_map = {l: i for i, l in enumerate(diagnosis_labels)}
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)

    idx_all = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y_diag,
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    print(f"  v3 split reproduced: {len(train_idx):,} train | {len(test_idx):,} test")
    return df_train, len(test_idx)


# Index building at one granularity
def _code_column(df: pd.DataFrame, granularity: str) -> list:
    return df["rollup_code"].tolist() if granularity == "rollup" else df["icd_code"].tolist()


def _build_for_granularity(df_rows: pd.DataFrame, tfidf, granularity: str,
                           standardizer: dict) -> dict:
    vectors = icdr.vectorize_queries(df_rows["complaint_text_norm"].tolist(), tfidf)
    physio = icdr.physio_matrix(df_rows, standardizer)
    return icdr.build_index(
        vectors_l2=vectors,
        codes=_code_column(df_rows, granularity),
        titles=df_rows["icd_title"].tolist(),
        categories=df_rows["diagnosis_group"].tolist(),
        physio=physio,
    )


# Weight tuning - 3-term simplex (text cosine / vitals / prevalence)
def _tune_weights(df_train: pd.DataFrame, tfidf, standardizer: dict,
                  pmh_standardizer: dict):
    """Grid-search the blend weights on a held-out train slice.

    Tunes (a) the 3-term (text, vit, prev) simplex on ALL val rows - the weights
    used for patients without history - and (b) the gated 4-term (text, vit,
    pmh, prev) simplex on the HAS-PMH val rows only, the weights applied to
    patients with prior history. Metric: oracle-category top-5 rollup hit rate.
    Returns best 3-term, best 2-term (vit=0 ablation), best 4-term, and rates.
    """
    build_df, val_df = train_test_split(
        df_train, test_size=TUNE_VAL_FRACTION, random_state=SPLIT_SEED,
        stratify=df_train["diagnosis_group"],
    )
    n_pmh_val = int((val_df["no_history"] == 0).sum())
    print(f"\n  Weight tuning: build on {len(build_df):,}, validate on {len(val_df):,} "
          f"({n_pmh_val:,} with PMH); oracle-category top-{TUNE_K} rollup hit rate")

    index = _build_for_granularity(build_df, tfidf, "rollup", standardizer)
    # Attach PMH centroids from has-history build rows only.
    icdr.attach_centroids(
        index, icdr.physio_matrix(build_df, pmh_standardizer),
        _code_column(build_df, "rollup"), build_df["diagnosis_group"].tolist(),
        key="pmh_centroids", row_mask=(build_df["no_history"] == 0).to_numpy(),
    )

    # Per category, precompute the min-max'd term matrices, true-code indices,
    # and the has-PMH mask.
    precomp = []
    total = 0
    for cat, sub in val_df.groupby("diagnosis_group"):
        entry = index.get(cat)
        total += len(sub)
        if entry is None:
            continue
        Q = icdr.vectorize_queries(sub["complaint_text_norm"].tolist(), tfidf)
        text_cos = np.asarray(Q @ entry["centroids"].T, dtype=np.float32)
        vit = icdr._neg_euclidean(icdr.physio_matrix(sub, standardizer),
                                  entry["vital_centroids"])
        pmh = icdr._neg_euclidean(icdr.physio_matrix(sub, pmh_standardizer),
                                  entry["pmh_centroids"])
        Tn = icdr._minmax_rows(text_cos)
        Vn = icdr._minmax_rows(vit)
        Mn = icdr._minmax_rows(pmh)
        Pn = icdr._minmax(entry["prevalence"])
        code_pos = {c: i for i, c in enumerate(entry["codes"])}
        true_idx = np.array([code_pos.get(c, -1) for c in sub["rollup_code"]], dtype=int)
        has_pmh = (sub["no_history"] == 0).to_numpy()
        precomp.append((Tn, Vn, Mn, Pn, true_idx, has_pmh))

    def _hits(score, true_idx):
        topk = np.argsort(-score, axis=1)[:, :TUNE_K]
        return int(((topk == true_idx[:, None]) & (true_idx[:, None] >= 0)).any(axis=1).sum())

    def _rate3(w):
        hits = 0
        for Tn, Vn, _Mn, Pn, true_idx, _hp in precomp:
            hits += _hits(w["text"] * Tn + w["vit"] * Vn + w["prev"] * Pn[None, :], true_idx)
        return hits / total

    def _rate4(w):
        hits = 0
        for Tn, Vn, Mn, Pn, true_idx, hp in precomp:
            if hp.sum() == 0:
                continue
            score = (w["text"] * Tn[hp] + w["vit"] * Vn[hp]
                     + w["pmh"] * Mn[hp] + w["prev"] * Pn[None, :])
            hits += _hits(score, true_idx[hp])
        return hits / n_pmh_val

    rates3 = [(w, _rate3(w)) for w in WEIGHT_GRID]
    best_w, best_rate = max(rates3, key=lambda x: x[1])
    novit = [(w, r) for w, r in rates3 if w["vit"] == 0.0]
    best_novit_w, best_novit_rate = max(novit, key=lambda x: x[1])

    # 4-term: tuned on has-PMH rows. The vit=pmh=0 corner reproduces the 2-term
    # baseline restricted to has-PMH rows, so the lift is apples-to-apples.
    rates4 = [(w, _rate4(w)) for w in WEIGHT_GRID4]
    best_w_pmh, best_rate_pmh = max(rates4, key=lambda x: x[1])
    nopmh_haspmh = [(w, r) for w, r in rates4 if w["pmh"] == 0.0]
    best_nopmh_w, best_nopmh_rate = max(nopmh_haspmh, key=lambda x: x[1])

    print(f"    best 3-term (all val): text={best_w['text']:.1f} vit={best_w['vit']:.1f} "
          f"prev={best_w['prev']:.1f}  -> {best_rate:.4f}  "
          f"(vitals lift {best_rate - best_novit_rate:+.4f})")
    print(f"    best 4-term (has-PMH val): text={best_w_pmh['text']:.1f} "
          f"vit={best_w_pmh['vit']:.1f} pmh={best_w_pmh['pmh']:.1f} "
          f"prev={best_w_pmh['prev']:.1f}  -> {best_rate_pmh:.4f}")
    print(f"    PMH lift on has-PMH val (vs best pmh=0): {best_rate_pmh - best_nopmh_rate:+.4f}")
    return {
        "weights": best_w, "weights_novit": best_novit_w,
        "weights_pmh": best_w_pmh,
        "rate_3term": best_rate, "rate_2term": best_novit_rate,
        "rate_4term_haspmh": best_rate_pmh, "rate_nopmh_haspmh": best_nopmh_rate,
        "n_pmh_val": n_pmh_val,
    }


# Sanity summary
def _print_summary(index: dict, granularity: str) -> None:
    print(f"\n  [{granularity}] per-category candidate counts + top prevalence code:")
    for cat in sorted(index):
        entry = index[cat]
        m = len(entry["codes"])
        top = int(np.argmax(entry["prevalence"]))
        print(f"    {cat:42s}  {m:>5,} codes  | "
              f"top: {entry['codes'][top]:<8s} {entry['prevalence'][top]:5.1%}  "
              f"{entry['titles'][top][:40]}")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Build the Stage-2 exact-ICD resolver from the Doctor v3 "
                    "training split.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Also run the full v3 loader once and assert the fast light "
             "loader reproduces the identical row order / split (slow).",
    )
    args, _ = parser.parse_known_args()

    print("  Stage-2 exact-ICD resolver builder (Doctor v3 training split)")

    df_train, n_test = _train_rows(verify=args.verify)

    tfidf_path = TRIAGE_V1_DIR / "tfidf_vectorizer.joblib"
    tfidf = joblib.load(tfidf_path)
    print(f"  Loaded TF-IDF vectorizer (vocab={len(tfidf.vocabulary_):,})")

    # Fit standardizers on the training split: physiology (vitals + flags +
    # rhythm + deltas) on all rows; PMH (prior-dx flags + same-complaint) on
    # has-history rows only (so the no-history sentinel doesn't pollute z-scores).
    standardizer = icdr.build_standardizer(df_train, cols=PHYSIO_COLS_V2)
    has_pmh_train = (df_train["no_history"] == 0)
    pmh_standardizer = icdr.build_standardizer(
        df_train[has_pmh_train], cols=PMH_RESOLVER_COLS)
    print(f"  Physiology features: {len(PHYSIO_COLS_V2)} "
          f"(snapshot 13 + rhythm/delta {len(_LONG_RHYTHM_DELTA)}) | "
          f"PMH features: {len(PMH_RESOLVER_COLS)} | "
          f"has-PMH train: {int(has_pmh_train.sum()):,}/{len(df_train):,} "
          f"({100*has_pmh_train.mean():.1f}%)")

    # 1. Tune the 3-term (no-PMH) and gated 4-term (has-PMH) blends.
    tune = _tune_weights(df_train, tfidf, standardizer, pmh_standardizer)
    best_w, best_novit_w, best_w_pmh = (
        tune["weights"], tune["weights_novit"], tune["weights_pmh"])
    # Legacy 2-term alpha for the backward-compatible rank_within_category path.
    denom = best_novit_w["text"] + best_novit_w["prev"]
    best_alpha = best_novit_w["text"] / denom if denom > 0 else 0.5

    # 2. Rebuild final indices on the FULL train split: text + vitals centroids,
    #    then PMH centroids (from has-history rows only).
    print("\n  Building final indices on the full training split...")
    granularities = {}
    pmh_matrix = icdr.physio_matrix(df_train, pmh_standardizer)
    for g in icdr.GRANULARITIES:
        granularities[g] = _build_for_granularity(df_train, tfidf, g, standardizer)
        icdr.attach_centroids(
            granularities[g], pmh_matrix, _code_column(df_train, g),
            df_train["diagnosis_group"].tolist(), key="pmh_centroids",
            row_mask=has_pmh_train.to_numpy(),
        )
        _print_summary(granularities[g], g)

    # 3. Assemble + save resolver.
    resolver = icdr.make_resolver(
        granularities=granularities,
        alpha=best_alpha,
        weights=best_w,
        standardizer=standardizer,
        weights_pmh=best_w_pmh,
        pmh_standardizer=pmh_standardizer,
        vocab_size=len(tfidf.vocabulary_),
        tfidf_path=str(tfidf_path),
        n_train=len(df_train),
        extra={
            "weights_novit": best_novit_w,
            "tuning_metric": f"oracle-category top-{TUNE_K} rollup hit rate",
            "tuning_val_rate_3term": round(float(tune["rate_3term"]), 4),
            "tuning_val_rate_2term": round(float(tune["rate_2term"]), 4),
            "tuning_val_rate_4term_haspmh": round(float(tune["rate_4term_haspmh"]), 4),
            "physio_cols": PHYSIO_COLS_V2,
            "pmh_cols": PMH_RESOLVER_COLS,
            "split": {"test_size": TEST_SIZE, "random_state": SPLIT_SEED},
            "n_test": int(n_test),
        },
    )
    icdr.save_resolver(resolver, DOCTOR_V3_ICD_RESOLVER_DIR)

    # Human-readable metadata sidecar.
    meta = {
        "built_at": resolver["built_at"],
        "weights_3term": best_w,
        "weights_2term": best_novit_w,
        "weights_4term_pmh": best_w_pmh,
        "alpha_2term": round(best_alpha, 3),
        "tuning_val_rate_3term": round(float(tune["rate_3term"]), 4),
        "tuning_val_rate_2term": round(float(tune["rate_2term"]), 4),
        "vitals_lift_on_val": round(float(tune["rate_3term"] - tune["rate_2term"]), 4),
        "tuning_val_rate_4term_haspmh": round(float(tune["rate_4term_haspmh"]), 4),
        "pmh_lift_on_haspmh_val": round(
            float(tune["rate_4term_haspmh"] - tune["rate_nopmh_haspmh"]), 4),
        "n_pmh_val": tune["n_pmh_val"],
        "physio_cols": PHYSIO_COLS_V2,
        "pmh_cols": PMH_RESOLVER_COLS,
        "n_train": len(df_train),
        "n_test": int(n_test),
        "vocab_size": resolver["vocab_size"],
        "granularity_candidate_counts": {
            g: {cat: len(granularities[g][cat]["codes"]) for cat in granularities[g]}
            for g in icdr.GRANULARITIES
        },
    }
    with open(DOCTOR_V3_ICD_RESOLVER_DIR / icdr.RESOLVER_META_FILENAME, "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved resolver -> {DOCTOR_V3_ICD_RESOLVER_DIR}")
    print(f"  - {icdr.RESOLVER_FILENAME}")
    print(f"  - {icdr.RESOLVER_META_FILENAME}")
    print("  RESOLVER BUILD COMPLETE")


if __name__ == "__main__":
    main()
