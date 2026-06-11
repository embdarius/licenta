"""Offline builder for the Stage-2 exact-ICD resolver (Doctor v3).

Builds per-category candidate indices (prototype centroids + prevalence priors)
from the **Doctor v3 training split only**, at two granularities:
  - "rollup" : 3-char ICD category (headline target)
  - "full"   : full ICD code (secondary/stretch target)

Each candidate code also carries a vitals centroid (mean standardized
physiology vector). The 3-term blend weights (text cosine / vitals / prevalence)
are grid-searched on a held-out slice of the training split using the
oracle-category top-5 rollup hit rate, then the final indices are rebuilt on the
*full* training split and saved (with the legacy 2-term alpha kept for the
backward-compatible text+prevalence path).

Run: ``uv run train_icd_resolver``  (heavy: reuses the v3 data loader).

This is offline/benchmark tooling — it is NOT part of the live patient crew.
"""
import argparse
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from proiect_licenta.paths import (
    TRIAGE_V1_DIR, TRIAGE_CSV, EDSTAYS_CSV, PATIENTS_CSV, SERVICES_CSV,
    DIAGNOSIS_CSV, DOCTOR_V3_ICD_RESOLVER_DIR,
)
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.training.train_doctor import (
    DIAGNOSIS_GROUP_MAP, SERVICE_GROUP_MAP, CATCH_ALL_LABEL,
)
from proiect_licenta.training.train_nurse import _clean_vitals
from proiect_licenta import icd_resolution as icdr

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


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
# The resolver only needs each train-split stay's complaint text + ICD code +
# diagnosis group. It does NOT need the snapshot/longitudinal vitals, the
# medication flags, or the PMH features — and building those is where
# train_nurse_v3.load_and_clean_data spends ~all of its time (the uncached
# 3.3 GB discharge.csv parse + the vitalsign.csv per-stay loop).
#
# Crucially, none of that slow data changes which rows survive: medrecon,
# vitalsign and PMH are all LEFT-joined AFTER the row set is fixed, so they
# never drop or reorder rows. The row set + order (hence the train/test split)
# is determined entirely by the triage/edstays/diagnosis/services merges and
# the dropna/catch-all/acuity filters below. `_load_light` reproduces exactly
# those row-determining steps (mirroring train_nurse_v3.load_and_clean_data),
# so the split is identical with no leakage — at a fraction of the cost.
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

    # ── Row-dropping filters (identical to the v3 loader) ──
    df = df.dropna(subset=["chiefcomplaint", "category", "curr_service"])
    df = df[df["chiefcomplaint"].str.strip() != ""]
    df["diagnosis_group"] = df["category"].map(DIAGNOSIS_GROUP_MAP).fillna("Other")
    df["service_group"] = df["curr_service"].map(SERVICE_GROUP_MAP).fillna("OTHER")
    df = df[df["diagnosis_group"] != CATCH_ALL_LABEL].reset_index(drop=True)
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    df = df[df["acuity"].between(1, 5)].reset_index(drop=True)

    # ── Derived columns the resolver needs ──
    df["icd_code"] = df["icd_code"].fillna("").str.strip().str.upper()
    df["icd_version"] = df["icd_version"].fillna("").str.strip()
    df["icd_title"] = df["icd_title"].fillna("")
    df["rollup_code"] = [
        icdr.rollup_icd(c, v) for c, v in zip(df["icd_code"], df["icd_version"])
    ]
    df["complaint_text_norm"] = df["chiefcomplaint"].apply(normalize_complaint_text)

    # Snapshot vitals + abnormal flags for the vitals-conditioned term. These
    # come from triage.csv (already merged), so this stays cheap — no
    # vitalsign.csv / PMH load. `_clean_vitals` clips, median-imputes, and adds
    # the 7 flags (fever/tachycardia/...), all in-place.
    _clean_vitals(df)
    return df


def _verify_against_full(light_df: pd.DataFrame) -> None:
    """Assert the light loader's row order matches the full v3 loader's.

    Opt-in (`--verify`) one-time check: if the stay_id sequence matches, the
    train/test split is provably identical (the split depends only on row order
    + count). Heavy — runs the full v3 loader once.
    """
    from proiect_licenta.training.train_nurse_v3 import load_and_clean_data
    print("\n  [--verify] Running the FULL v3 loader to compare row order...")
    full_df = load_and_clean_data()
    a = light_df["stay_id"].tolist()
    b = full_df["stay_id"].tolist()
    if a == b:
        print(f"  [--verify] PASS — identical row order ({len(a):,} rows). "
              f"Light loader split == v3 split.")
    else:
        raise AssertionError(
            f"[--verify] FAIL — row order differs (light={len(a):,}, "
            f"full={len(b):,}). The light loader has drifted from "
            f"train_nurse_v3.load_and_clean_data; fix before trusting the split."
        )


def _train_rows(verify: bool = False) -> tuple[pd.DataFrame, int]:
    """Reproduce the v3 train/test split, return the TRAIN rows + test size.

    The split partition depends only on (n, test_size, random_state, stratify),
    so splitting np.arange(n) with the identical y_diag stratify reproduces the
    exact partition train_nurse_v3.main used — no feature matrix needed here.
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


# ---------------------------------------------------------------------------
# Index building at one granularity
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Weight tuning — 3-term simplex (text cosine / vitals / prevalence)
# ---------------------------------------------------------------------------
def _tune_weights(df_train: pd.DataFrame, tfidf, standardizer: dict):
    """Grid-search the (text, vit, prev) simplex on a held-out train slice.

    Metric: oracle-category top-5 rollup hit rate. Returns the best 3-term
    weights, the best 2-term weights (vit == 0, for the no-vitals ablation),
    and the full rate table. Precomputes the per-category min-max'd term
    matrices once so the 66-combo grid is just weighted sums + argsort.
    """
    build_df, val_df = train_test_split(
        df_train, test_size=TUNE_VAL_FRACTION, random_state=SPLIT_SEED,
        stratify=df_train["diagnosis_group"],
    )
    print(f"\n  Weight tuning: build on {len(build_df):,}, validate on {len(val_df):,} "
          f"(oracle-category top-{TUNE_K} rollup hit rate, {len(WEIGHT_GRID)} combos)")

    index = _build_for_granularity(build_df, tfidf, "rollup", standardizer)

    # Per category, precompute the three min-max'd term matrices + each val
    # row's true-code index (−1 if the code isn't a candidate).
    precomp = []
    total = 0
    for cat, sub in val_df.groupby("diagnosis_group"):
        entry = index.get(cat)
        total += len(sub)
        if entry is None:
            continue
        Q = icdr.vectorize_queries(sub["complaint_text_norm"].tolist(), tfidf)
        text_cos = np.asarray(Q @ entry["centroids"].T, dtype=np.float32)  # (Nc,m)
        physio = icdr.physio_matrix(sub, standardizer)
        vit = icdr._neg_euclidean(physio, entry["vital_centroids"])        # (Nc,m)
        Tn = icdr._minmax_rows(text_cos)
        Vn = icdr._minmax_rows(vit)
        Pn = icdr._minmax(entry["prevalence"])
        code_pos = {c: i for i, c in enumerate(entry["codes"])}
        true_idx = np.array([code_pos.get(c, -1) for c in sub["rollup_code"]], dtype=int)
        precomp.append((Tn, Vn, Pn, true_idx))

    def _rate(w):
        hits = 0
        for Tn, Vn, Pn, true_idx in precomp:
            score = w["text"] * Tn + w["vit"] * Vn + w["prev"] * Pn[None, :]
            topk = np.argsort(-score, axis=1)[:, :TUNE_K]
            hits += int(((topk == true_idx[:, None]) & (true_idx[:, None] >= 0)).any(axis=1).sum())
        return hits / total

    rates = [(w, _rate(w)) for w in WEIGHT_GRID]
    best_w, best_rate = max(rates, key=lambda x: x[1])
    novit = [(w, r) for w, r in rates if w["vit"] == 0.0]
    best_novit_w, best_novit_rate = max(novit, key=lambda x: x[1])

    print(f"    best 3-term: text={best_w['text']:.1f} vit={best_w['vit']:.1f} "
          f"prev={best_w['prev']:.1f}  -> {best_rate:.4f}")
    print(f"    best 2-term (vit=0): text={best_novit_w['text']:.1f} "
          f"prev={best_novit_w['prev']:.1f}  -> {best_novit_rate:.4f}")
    print(f"    vitals lift on val: {best_rate - best_novit_rate:+.4f}")
    return best_w, best_rate, best_novit_w, best_novit_rate, rates


# ---------------------------------------------------------------------------
# Sanity summary
# ---------------------------------------------------------------------------
def _print_summary(index: dict, granularity: str) -> None:
    print(f"\n  [{granularity}] per-category candidate counts + top prevalence code:")
    for cat in sorted(index):
        entry = index[cat]
        m = len(entry["codes"])
        top = int(np.argmax(entry["prevalence"]))
        print(f"    {cat:42s}  {m:>5,} codes  | "
              f"top: {entry['codes'][top]:<8s} {entry['prevalence'][top]:5.1%}  "
              f"{entry['titles'][top][:40]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    print("\n" + "#" * 70)
    print("  Stage-2 exact-ICD resolver builder (Doctor v3 training split)")
    print("#" * 70)

    df_train, n_test = _train_rows(verify=args.verify)

    tfidf_path = TRIAGE_V1_DIR / "tfidf_vectorizer.joblib"
    tfidf = joblib.load(tfidf_path)
    print(f"  Loaded TF-IDF vectorizer (vocab={len(tfidf.vocabulary_):,})")

    # Fit the physiology standardizer on the full training split.
    standardizer = icdr.build_standardizer(df_train)

    # 1. Tune the 3-term (text/vit/prev) blend on a held-out train slice.
    best_w, best_rate, best_novit_w, best_novit_rate, weight_rates = \
        _tune_weights(df_train, tfidf, standardizer)
    # Legacy 2-term alpha (text vs prevalence) for the backward-compatible
    # rank_within_category path — derived from the best vit=0 combo.
    denom = best_novit_w["text"] + best_novit_w["prev"]
    best_alpha = best_novit_w["text"] / denom if denom > 0 else 0.5

    # 2. Rebuild final indices (with vitals centroids) on the FULL train split.
    print("\n  Building final indices on the full training split...")
    granularities = {}
    for g in icdr.GRANULARITIES:
        granularities[g] = _build_for_granularity(df_train, tfidf, g, standardizer)
        _print_summary(granularities[g], g)

    # 3. Assemble + save resolver.
    resolver = icdr.make_resolver(
        granularities=granularities,
        alpha=best_alpha,
        weights=best_w,
        standardizer=standardizer,
        vocab_size=len(tfidf.vocabulary_),
        tfidf_path=str(tfidf_path),
        n_train=len(df_train),
        extra={
            "weights_novit": best_novit_w,
            "tuning_metric": f"oracle-category top-{TUNE_K} rollup hit rate",
            "tuning_val_rate_3term": round(float(best_rate), 4),
            "tuning_val_rate_2term": round(float(best_novit_rate), 4),
            "physio_cols": icdr.PHYSIO_COLS,
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
        "alpha_2term": round(best_alpha, 3),
        "tuning_val_rate_3term": round(float(best_rate), 4),
        "tuning_val_rate_2term": round(float(best_novit_rate), 4),
        "vitals_lift_on_val": round(float(best_rate - best_novit_rate), 4),
        "physio_cols": icdr.PHYSIO_COLS,
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
    print("\n" + "#" * 70)
    print("  RESOLVER BUILD COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
