"""
Data Pipeline for Doctor Disposition v3 (Option B from plan section 3)
=======================================================================

A peer model alongside Doctor v3-nurse diagnosis + department. Unlike those,
this one is a BINARY admit/discharge classifier and is trained on the FULL
425K ED stays (not just the admitted-only ~102K slice), so it sees the
positives and negatives in their real-world ratio.

The aim is to refine the triage-time disposition (triage v3 iter 2 ref:
77.98% acc, ROC AUC 0.8644, under-triage 15.56%, over-triage 16.90% —
see docs/agents/triage-agent.md) after the nurse step. Expected lift
band per plan section 3: +3-6pp accuracy, +0.03-0.05 ROC AUC.

Differences vs ``train_nurse_v3``:

  * **No admitted-only filter, no catch-all filter, no diagnosis/service
    join.** We keep every stay with a non-null disposition. The label is
    ``admitted = 1 if disposition == 'ADMITTED' else 0``.
  * **Soft cascade from triage** instead of hard cascade. Plan section 2/3
    recommends feeding the full 5-class acuity softmax + the triage
    disposition probability. The doctor model can then weight a "borderline
    ESI 2-3 with abnormal vitals" case honestly rather than hard-locking
    on the triage argmax. Six extra float columns vs the two int columns
    used by the diagnosis/department models.
  * **Binary:logistic objective** with ``scale_pos_weight = N_neg / N_pos``
    (symmetric baseline per plan recommendation; threshold tuning is a
    separate, downstream concern).
  * **Isotonic calibration** on a 10% held-out fit slice (A4 pattern). The
    plan calls out calibration as more important here than for diagnosis,
    because disposition output is a clinical-decision probability — a
    miscalibrated 0.8 hurts.
  * **Leakage guard** kept from nurse_v3: the longitudinal vital window is
    ``[intime, intime + 4h]`` so disposition-time vitals never leak in.

Inputs (FEATURE SET, same as nurse_v3 minus diag/dept cascade):
  - Triage structured: pain, demographics, severity priors (51 cols incl. PMH)
  - Triage TF-IDF (2000 cols)
  - **Triage SOFT cascade** (5 acuity softmax cols + 1 dispo prob col = 6)
  - Snapshot vitals + clinical flags (20 cols)
  - Medication features (24 cols: n_meds + meds_unknown + 22 categories)
  - Longitudinal vitals + rhythm + abnormal-counts (~40 cols)
  - PMH features (19 cols)

Outputs:
  - ``artifacts/doctor/v3/disposition_model.joblib`` — calibrated binary
    classifier. ``predict_proba(X)[:, 1]`` is P(admit).
  - ``artifacts/doctor/v3/metadata.json`` gains a ``disposition`` sub-block
    with accuracy, ROC AUC, over/under-triage rates, calibration ECE, and
    the soft-cascade column names for inference symmetry.
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
    _HAS_FROZEN = True
except ImportError:  # pragma: no cover
    FrozenEstimator = None  # type: ignore[assignment]
    _HAS_FROZEN = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# XGBoost device / tree method — same env-var contract as the other pipelines.
# ---------------------------------------------------------------------------
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")
XGB_TREE_METHOD = os.environ.get("XGB_TREE_METHOD", "hist")

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore

    def tqdm(it, **kw):
        return _tqdm(it, **kw)
except ImportError:  # pragma: no cover

    def tqdm(it, **kw):
        return it


def _make_xgb_progress_callback(n_total: int, desc: str):
    """tqdm-bar XGBoost TrainingCallback (same shape as nurse_v3)."""
    try:
        from tqdm.auto import tqdm as _tqdm_cls
        from xgboost.callback import TrainingCallback
    except ImportError:
        return None

    class _TqdmCallback(TrainingCallback):  # type: ignore[misc]
        def __init__(self):
            self.pbar = _tqdm_cls(total=n_total, desc=desc, unit="iter", leave=False)

        def after_iteration(self, model, epoch, evals_log):
            self.pbar.update(1)
            last_val = None
            for ds in evals_log:
                for metric in evals_log[ds]:
                    vals = evals_log[ds][metric]
                    if vals:
                        last_val = vals[-1]
            if last_val is not None and hasattr(self.pbar, "set_postfix"):
                self.pbar.set_postfix(logloss=f"{last_val:.4f}", refresh=False)
            return False

        def after_training(self, model):
            self.pbar.close()
            return model

    return _TqdmCallback()


# ---------------------------------------------------------------------------
# Paths + shared helpers (reuse everything from nurse_v3 / preprocessing)
# ---------------------------------------------------------------------------
from proiect_licenta.paths import (
    TRIAGE_V3_DIR as TRIAGE_MODELS_DIR,
    DOCTOR_V3_DIR,
    TRIAGE_CSV, VITALSIGN_CSV, EDSTAYS_CSV, PATIENTS_CSV,
    DIAGNOSIS_CSV, MEDRECON_CSV,
    DISCHARGE_NOTES_CSV, DIAGNOSES_ICD_CSV, ADMISSIONS_CSV,
)
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.pmh_vocab import PMH_CATEGORIES
from proiect_licenta.pmh_features import (
    PMH_FEATURE_COLS,
    PMH_NO_PRIOR_DAYS,
    aggregate_pmh as _aggregate_pmh_shared,
    fill_missing_pmh_columns,
)
from proiect_licenta.training.train_nurse import (
    _aggregate_medications, _clean_vitals, MED_CATEGORY_KEYWORDS,
)
from proiect_licenta.training.train_nurse_v3 import (
    _aggregate_vitalsigns, _fill_longitudinal_vitals,
    LONG_VITAL_FEATURE_COLS, _LONG_WINDOW_HOURS, _RHYTHM_BUCKETS,
)
# Triage v3 cascade source. We import the v3 feature builder + vital
# constants to construct the v3-shaped input vector exactly as the v3
# acuity / disposition models expect it. Train/inference symmetry: at
# runtime `triage_tool.py` also loads v3 artifacts, so the soft cascade
# the doctor disposition model learns is the same softmax it sees live.
from proiect_licenta.training import train_triage_v3 as _triage_v3
from proiect_licenta.training.train_triage_v3 import (
    VITAL_COLS as _V3_VITAL_COLS,
    build_features as _build_v3_features,
)
# Reuse the catch-all label name + diagnosis grouping ONLY for the PMH step,
# which still needs to bucket prior ICDs into categories. We do NOT filter on
# diagnosis_group for the disposition model itself.
from proiect_licenta.training.train_doctor import (
    CATCH_ALL_LABEL, DIAGNOSIS_GROUP_MAP,
)


# ---------------------------------------------------------------------------
# 1. Load & clean — FULL dataset, no admitted-only filter
# ---------------------------------------------------------------------------
def load_and_clean_data() -> pd.DataFrame:
    """Load every ED stay with a usable disposition label.

    No filter on disposition == ADMITTED (that's the label here, not a
    pre-filter). No filter on diagnosis_group (the diagnosis catch-all does
    not affect the disposition decision — discharged patients especially
    have no primary diagnosis recorded so we'd lose them all).
    """
    print("=" * 60)
    print("STEP 1: Loading data (disposition v3 — FULL dataset, "
          "no admitted-only filter)")
    print("=" * 60)

    # Triage (vitals + chief complaint + acuity)
    triage = pd.read_csv(TRIAGE_CSV)
    print(f"  triage.csv: {len(triage):,} rows")

    # edstays — every stay, every disposition; need full set for PMH lookups too
    edstays = pd.read_csv(
        EDSTAYS_CSV,
        usecols=["subject_id", "stay_id", "hadm_id", "intime", "gender",
                 "arrival_transport", "disposition"],
    )
    print(f"  edstays.csv: {len(edstays):,} rows")

    # Patients (for age computation)
    patients = pd.read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "anchor_age", "anchor_year"],
    )
    print(f"  patients.csv: {len(patients):,} rows")

    # Medications — left-joined; many stays have none
    print("  Loading medrecon.csv and aggregating per stay...")
    med = pd.read_csv(MEDRECON_CSV, usecols=["stay_id", "name", "etcdescription"])
    med_features = _aggregate_medications(med)
    print(f"  medrecon.csv: {len(med):,} rows -> {len(med_features):,} stay-level records")

    # Disposition label — keep stays with a real disposition only. Two-class
    # collapse: ADMITTED → 1; everything else (HOME, ELOPED, LEFT WITHOUT
    # BEING SEEN, TRANSFER, EXPIRED, OTHER) → 0. The "discharge" label here
    # captures every non-admission, which matches what triage's existing
    # disposition model targets.
    edstays = edstays.dropna(subset=["disposition"]).copy()
    print(f"  edstays after dropping null disposition: {len(edstays):,}")
    print(f"  Disposition distribution before binarization:")
    for d, n in edstays["disposition"].value_counts().items():
        print(f"    {d:30s}  {n:>7,}  ({100*n/len(edstays):.1f}%)")
    edstays["admitted"] = (edstays["disposition"] == "ADMITTED").astype(int)

    # ── Merge ──
    df = triage.merge(edstays, on=["subject_id", "stay_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="left")
    df = df.merge(med_features, on="stay_id", how="left")
    print(f"  After triage + edstays + patients + meds merge: {len(df):,}")

    # ── Age computation ──
    df["intime"] = pd.to_datetime(df["intime"])
    df["visit_year"] = df["intime"].dt.year
    df["age"] = df["anchor_age"] + (df["visit_year"] - df["anchor_year"])
    df["age"] = df["age"].clip(0, 120).fillna(50).astype(int)

    # ── Drop rows with missing chief complaint (the cascade needs text) ──
    initial = len(df)
    df = df.dropna(subset=["chiefcomplaint"]).copy()
    df = df[df["chiefcomplaint"].str.strip() != ""]
    print(f"  After dropping missing chief complaint: {len(df):,} "
          f"(dropped {initial - len(df):,})")

    # ── Pain / demographics / acuity ──
    df["pain_triage"] = pd.to_numeric(df["pain"], errors="coerce")
    df["pain_missing"] = df["pain_triage"].isna().astype(int)
    df["pain"] = df["pain_triage"].fillna(-1).astype(int)
    df.loc[df["pain"] > 10, "pain"] = -1
    df.loc[df["pain"] < 0, "pain"] = -1

    df["gender_male"] = (df["gender"] == "M").astype(int)
    df["arrival_ambulance"] = (df["arrival_transport"] == "AMBULANCE").astype(int)
    df["arrival_helicopter"] = (df["arrival_transport"] == "HELICOPTER").astype(int)
    df["arrival_walk_in"] = (df["arrival_transport"] == "WALK IN").astype(int)

    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    # Drop rows with missing acuity — the cascade needs a triage prediction.
    # The triage model itself can't have been run on these in MIMIC since they
    # never received an acuity, and we don't want to pollute training with
    # null cascade values.
    before = len(df)
    df = df[df["acuity"].between(1, 5)].copy()
    df["acuity"] = df["acuity"].astype(int)
    print(f"  After acuity-in-[1,5] filter: {len(df):,} (dropped {before - len(df):,})")

    # ── Snapshot vitals (triage.csv) ──
    _clean_vitals(df)

    # ── Longitudinal vitals (vitalsign.csv) — first 4h window ──
    print("\n  Aggregating longitudinal vitals from vitalsign.csv "
          "(window: [intime, intime + 4h] — leakage guard)...")
    long_vitals = _aggregate_vitalsigns(
        df[["stay_id", "intime"]],
        VITALSIGN_CSV,
    )
    df = df.merge(long_vitals, on="stay_id", how="left")
    _fill_longitudinal_vitals(df)
    coverage = (df["has_longitudinal_vitals"] == 1).mean()
    print(f"  Longitudinal vitals coverage: {100*coverage:.1f}% of full dataset")

    # ── PMH (Change 1 recipe) ──
    # PMH operates on prior encounters strictly before intime → no leakage.
    print("\n  Aggregating PMH features from prior encounters...")
    df["complaint_text_norm"] = df["chiefcomplaint"].apply(normalize_complaint_text)
    # We need edstays_full (not the merged frame) for the prior-encounter lookups
    # so PMH can see EVERY past stay even those filtered out above.
    edstays_full = pd.read_csv(
        EDSTAYS_CSV,
        usecols=["subject_id", "stay_id", "hadm_id", "intime", "disposition"],
    )
    pmh_df = _aggregate_pmh_shared(
        stays_df=df[["stay_id", "subject_id", "intime", "complaint_text_norm"]],
        edstays_full=edstays_full,
        diagnoses_icd_csv_path=DIAGNOSES_ICD_CSV,
        admissions_csv_path=ADMISSIONS_CSV,
        discharge_csv_path=DISCHARGE_NOTES_CSV,
        diagnosis_csv_path=DIAGNOSIS_CSV,
        diagnosis_group_map=DIAGNOSIS_GROUP_MAP,
        catch_all_label=CATCH_ALL_LABEL,
    )
    df = df.merge(pmh_df, on="stay_id", how="left")
    fill_missing_pmh_columns(df)

    # ── Fill missing medication flags ──
    med_flag_cols = ["n_medications", "meds_unknown"] + list(MED_CATEGORY_KEYWORDS.keys())
    for col in med_flag_cols:
        if col not in df.columns:
            df[col] = 0
    df["n_medications"] = df["n_medications"].fillna(0).astype(int)
    df["meds_unknown"] = df["meds_unknown"].fillna(1).astype(int)
    for flag in MED_CATEGORY_KEYWORDS:
        df[flag] = df[flag].fillna(0).astype(int)

    # ── Class balance report ──
    pos = int(df["admitted"].sum())
    neg = int(len(df) - pos)
    print(f"\n  Final disposition label distribution:")
    print(f"    Admitted    : {pos:>7,}  ({100*pos/len(df):.1f}%)")
    print(f"    Discharged  : {neg:>7,}  ({100*neg/len(df):.1f}%)")
    print(f"    Ratio (neg/pos) = {neg/max(pos,1):.2f}")

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering — same triage feature set as nurse_v3, but with
#    SOFT cascade (5 acuity softmax cols + 1 disposition prob col)
# ---------------------------------------------------------------------------
SOFT_CASCADE_COLS = [
    "triage_acuity_proba_1", "triage_acuity_proba_2",
    "triage_acuity_proba_3", "triage_acuity_proba_4",
    "triage_acuity_proba_5",
    "triage_disposition_proba_admit",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix using **triage v3** as the cascade source.

    Pipeline:
      1. Load triage v3 artifacts (tfidf, severity_map, vital_medians,
         acuity_model, disposition_model).
      2. Construct a "cascade input" copy of ``df`` with vitals re-masked
         to NaN for walk-ins (since v3 was trained that way at
         ``train_triage_v3.load_and_clean_data:301``). The ``_missing``
         flags inside ``train_triage_v3.build_features`` then compute
         correctly even though ``df``'s main vital columns are already
         imputed by ``_clean_vitals`` upstream for the disposition
         model's own non-cascade use.
      3. Call ``train_triage_v3.build_features(fit=False)`` to build the
         v3 input vector (~2069 cols: 23 v1 structured + ~28 v2 vital +
         19 PMH + 2000 TF-IDF). Single source of truth — if v3 changes
         layout, this picks it up automatically.
      4. Compute the soft cascade:
           - ``acuity_proba``  = v3 acuity model softmax (5 cols)
           - ``predicted_acuity_int`` = argmax + 1 (v3 disposition model
             was trained with this int alongside the 2069-col base; see
             ``train_triage_v3.py:843-846``).
           - ``dispo_proba_admit`` = v3 disposition model on
             (v3 features + predicted_acuity).
      5. Final disposition feature matrix = v3 features + 6 soft cascade
         + 24 medication cols + ~40 longitudinal vital + rhythm cols. PMH
         and snapshot vitals are already included via the v3 features
         matrix (no duplication).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering (triage v3 cascade)")
    print("=" * 60)

    # ── Load triage v3 artifacts ──
    tfidf = joblib.load(TRIAGE_MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(TRIAGE_MODELS_DIR / "severity_map.joblib")
    vital_medians_v3 = joblib.load(TRIAGE_MODELS_DIR / "vital_medians.joblib")
    acuity_model = joblib.load(TRIAGE_MODELS_DIR / "acuity_model.joblib")
    disposition_model = joblib.load(TRIAGE_MODELS_DIR / "disposition_model.joblib")
    print(f"  Loaded triage v3 artifacts from {TRIAGE_MODELS_DIR.name}/")

    # Defensive: strip the custom `neg_quadratic_kappa` eval_metric callable
    # off the v3 acuity model so unpickled artifacts trained before the
    # `set_params(eval_metric=None)` fix in train_triage_v3 don't blow up
    # at predict_proba time. The shim that triage_tool.py uses for the
    # same reason is also picked up here implicitly via sys.modules patch
    # if/when this script is run under runpy with __main__ set.
    try:
        acuity_model.set_params(eval_metric=None)
    except Exception:
        pass

    # ── Build cascade-input copy with walk-in vital masking ──
    # train_triage_v3.build_features sets `<col>_missing = df[col].isna()`
    # before fillna. Since `_clean_vitals` already imputed our `df`'s
    # vital columns, NaN has been lost — we have to restore it on a copy
    # for the cascade input so the `_missing` flags compute the way the
    # v3 model expects.
    df_cascade = df.copy()
    walkin_mask = ~df_cascade["arrival_transport"].isin(["AMBULANCE", "HELICOPTER"])
    for col in _V3_VITAL_COLS:
        df_cascade.loc[walkin_mask, col] = np.nan
    n_walkin = int(walkin_mask.sum())
    print(f"  Cascade input: re-masked vitals to NaN for "
          f"{n_walkin:,} walk-in rows ({100*n_walkin/len(df_cascade):.1f}%)")

    # ── Run v3 build_features (transform mode) to get the cascade input ──
    triage_features, _, _, _ = _build_v3_features(
        df_cascade, tfidf=tfidf, severity_map=severity_map,
        vital_medians=vital_medians_v3, fit=False,
    )
    triage_features = triage_features.reset_index(drop=True)
    print(f"  v3 cascade input shape: {triage_features.shape}")

    # ── Soft cascade — v3 acuity softmax + v3 disposition probability ──
    print("  Generating SOFT-cascade predictions on v3 features "
          "(5-class acuity softmax + dispo probability)...")
    acuity_proba = acuity_model.predict_proba(triage_features)
    if acuity_proba.shape[1] != 5:
        raise ValueError(
            f"Expected triage v3 acuity model to output 5 classes; "
            f"got {acuity_proba.shape[1]}."
        )
    # Triage v3 disposition model expects `predicted_acuity` int alongside
    # the v3 base vector (see train_triage_v3.py:843-846 — the v3 cascade
    # contract is hard-int from acuity argmax + 1). Reproduce that input
    # here. The doctor disposition model receives the SOFT softmax, but
    # we still call the triage v3 dispo model the way it was trained.
    predicted_acuity_int = acuity_proba.argmax(axis=1) + 1
    triage_features_disp = triage_features.copy()
    triage_features_disp["predicted_acuity"] = predicted_acuity_int
    dispo_proba_admit = disposition_model.predict_proba(triage_features_disp)[:, 1]

    # Append the soft-cascade columns onto the v3 features matrix. These
    # 6 floats are what the doctor disposition model will treat as the
    # "what triage v3 thinks" cascade signal.
    for k in range(5):
        triage_features[f"triage_acuity_proba_{k + 1}"] = acuity_proba[:, k]
    triage_features["triage_disposition_proba_admit"] = dispo_proba_admit
    assert all(c in triage_features.columns for c in SOFT_CASCADE_COLS)

    # ── Medications ──
    med_cols = ["n_medications", "meds_unknown"] + list(MED_CATEGORY_KEYWORDS.keys())
    meds = df[med_cols].reset_index(drop=True)
    print(f"  Medication features: {len(med_cols)}")

    # ── Longitudinal vitals + rhythm ──
    long_vitals = df[LONG_VITAL_FEATURE_COLS].reset_index(drop=True)
    print(f"  Longitudinal + rhythm features: {len(LONG_VITAL_FEATURE_COLS)}")

    # ── Assemble final feature matrix ──
    # v3 features (~2069: structured + v3 vitals + PMH + TF-IDF) already
    # contains structured + snapshot vitals + PMH + TF-IDF. We do NOT
    # re-append a separate "snapshot vital block" or a "PMH block" — they
    # are already in `triage_features`. Adding them again would create
    # duplicate column names and inflate the importance of those signals.
    features = pd.concat([triage_features, meds, long_vitals], axis=1)
    print(f"  Total features: {features.shape[1]}  "
          f"(v3 base + 6 cascade + {len(med_cols)} meds + "
          f"{len(LONG_VITAL_FEATURE_COLS)} longitudinal)")
    # Sanity: no duplicate columns after the concat.
    dup = features.columns[features.columns.duplicated()].tolist()
    if dup:
        raise ValueError(
            f"Feature matrix has duplicate columns after concat: {dup[:10]}"
        )
    return features


# ---------------------------------------------------------------------------
# 3. Train binary disposition model with isotonic calibration
# ---------------------------------------------------------------------------
def train_disposition(
    X_train, y_train, X_test, y_test, model_name="DISPOSITION v3",
):
    """Train a binary XGBoost classifier with symmetric scale_pos_weight,
    then wrap with isotonic calibration on a 10% held-out slice.

    Returns
    -------
    (calibrated_model, raw_model, metrics_dict)
        - ``calibrated_model``: ``CalibratedClassifierCV`` wrapped over the
          XGB classifier. ``predict_proba(X)[:, 1]`` is the deployment
          admission probability.
        - ``raw_model``: the underlying ``XGBClassifier`` (kept for audit /
          feature-importance inspection).
        - ``metrics_dict``: a dict of every reported metric on ``X_test``.
    """
    print(f"\n{'='*60}")
    print(f"  Training {model_name}  (binary:logistic, full 425K dataset)")
    print(f"{'='*60}")

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(np.sqrt(neg / max(pos, 1)))
    # Plan recommendation: SYMMETRIC baseline = sqrt(N_neg/N_pos). Using the
    # sqrt rather than the raw ratio matches the doctor-v3 sweep's near-optimal
    # cw_exponent of 0.52 — full-ratio over-corrects for class balance and
    # leaves too much accuracy on the table on the negative class.
    print(f"  scale_pos_weight = sqrt({neg}/{pos}) = {scale_pos_weight:.4f}")

    # 10% calibration holdout cut from the train split (A4 pattern).
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42, stratify=y_train,
    )
    print(f"  Fit on {len(X_fit):,} | Calibration on {len(X_cal):,} | "
          f"Held-out test on {len(X_test):,}")

    progress_cb = _make_xgb_progress_callback(
        n_total=5000, desc=f"    XGBoost ({model_name})",
    )
    callbacks = [progress_cb] if progress_cb is not None else None

    raw = XGBClassifier(
        n_estimators=5000, max_depth=10, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.5, colsample_bylevel=0.7,
        min_child_weight=3, gamma=0.05, reg_alpha=0.5, reg_lambda=2.0,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=150,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, verbosity=0,
        tree_method=XGB_TREE_METHOD, device=XGB_DEVICE,
        callbacks=callbacks,
    )
    print(f"  Training (up to 5000 trees, lr=0.02, early stopping=150, "
          f"device={XGB_DEVICE}, tree_method={XGB_TREE_METHOD})...")
    raw.fit(
        X_fit, y_fit, sample_weight=None,
        eval_set=[(X_test, y_test)], verbose=False,
    )
    print(f"  Best iteration: {raw.best_iteration}")

    # Detach the tqdm callback so the model pickles cleanly.
    try:
        raw.set_params(callbacks=None)
    except Exception:
        pass

    # ── Isotonic calibration ──
    print(f"\n  Fitting isotonic calibration on held-out {len(X_cal):,} rows...")
    if _HAS_FROZEN:
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(raw), method="isotonic", cv=None,
        )
    else:
        calibrated = CalibratedClassifierCV(raw, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    # ── Metrics on the held-out test set ──
    proba_uncal = raw.predict_proba(X_test)[:, 1]
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    y_pred_uncal = (proba_uncal >= 0.5).astype(int)
    y_pred_cal = (proba_cal >= 0.5).astype(int)

    acc_uncal = accuracy_score(y_test, y_pred_uncal)
    acc_cal = accuracy_score(y_test, y_pred_cal)
    auc_uncal = roc_auc_score(y_test, proba_uncal)
    auc_cal = roc_auc_score(y_test, proba_cal)
    brier_uncal = brier_score_loss(y_test, proba_uncal)
    brier_cal = brier_score_loss(y_test, proba_cal)

    # Over/under-triage rates at 0.5 threshold
    cm = confusion_matrix(y_test, y_pred_cal)
    # rows = truth (0=discharge, 1=admit); cols = pred
    tn, fp, fn, tp = cm.ravel()
    over_triage = fp / max(tn + fp, 1)    # discharge predicted as admit
    under_triage = fn / max(fn + tp, 1)   # admit predicted as discharge
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    # Expected Calibration Error (10-bin)
    def _ece(y, p, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(p, bins[1:-1])
        ece = 0.0
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            conf = p[mask].mean()
            acc = y[mask].mean()
            ece += (mask.sum() / len(p)) * abs(conf - acc)
        return float(ece)

    ece_uncal = _ece(np.asarray(y_test), proba_uncal)
    ece_cal = _ece(np.asarray(y_test), proba_cal)

    print(f"\n  ── Test metrics ──")
    print(f"  Accuracy   uncal {acc_uncal:.4f} | cal {acc_cal:.4f}  "
          f"(delta {acc_cal - acc_uncal:+.4f})")
    print(f"  ROC AUC    uncal {auc_uncal:.4f} | cal {auc_cal:.4f}  "
          f"(delta {auc_cal - auc_uncal:+.4f})")
    print(f"  Brier      uncal {brier_uncal:.4f} | cal {brier_cal:.4f}  "
          f"(lower is better)")
    print(f"  ECE (10b)  uncal {ece_uncal:.4f} | cal {ece_cal:.4f}")
    print(f"  Over-triage  (false admit ): {over_triage:.4f}  "
          f"(triage v3 iter 2 reference: 0.1690)")
    print(f"  Under-triage (missed admit): {under_triage:.4f}  "
          f"(triage v3 iter 2 reference: 0.1556)")
    print(f"  Sensitivity (admit recall): {sens:.4f}")
    print(f"  Specificity (discharge recall): {spec:.4f}")
    print()
    print(classification_report(
        y_test, y_pred_cal,
        target_names=["discharge", "admit"], digits=3,
    ))

    metrics = {
        "accuracy_uncalibrated": round(float(acc_uncal), 4),
        "accuracy_calibrated":   round(float(acc_cal), 4),
        "roc_auc_uncalibrated":  round(float(auc_uncal), 4),
        "roc_auc_calibrated":    round(float(auc_cal), 4),
        "brier_uncalibrated":    round(float(brier_uncal), 4),
        "brier_calibrated":      round(float(brier_cal), 4),
        "ece_uncalibrated":      round(float(ece_uncal), 4),
        "ece_calibrated":        round(float(ece_cal), 4),
        "over_triage_rate":      round(float(over_triage), 4),
        "under_triage_rate":     round(float(under_triage), 4),
        "sensitivity":           round(float(sens), 4),
        "specificity":           round(float(spec), 4),
        "n_fit":                 int(len(X_fit)),
        "n_cal":                 int(len(X_cal)),
        "n_test":                int(len(X_test)),
        "scale_pos_weight":      round(float(scale_pos_weight), 4),
        "best_iteration":        int(raw.best_iteration),
    }
    return calibrated, raw, metrics


# ---------------------------------------------------------------------------
# 4. Save artifacts (extend existing metadata.json with a `disposition` block)
# ---------------------------------------------------------------------------
def save_disposition_model(
    calibrated_model, raw_model, metrics: dict, feature_cols: list,
):
    print(f"\n{'='*60}")
    print("  Saving Doctor disposition v3 artifacts")
    print(f"{'='*60}")

    DOCTOR_V3_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated_model, DOCTOR_V3_DIR / "disposition_model.joblib")
    joblib.dump(raw_model, DOCTOR_V3_DIR / "disposition_model_raw.joblib")
    print(f"  Saved disposition_model.joblib       (calibrated, deployment)")
    print(f"  Saved disposition_model_raw.joblib   (uncalibrated, audit)")

    # ── Merge into existing metadata.json so the file stays the single
    # source-of-truth for the v3 model bundle. If the file doesn't exist
    # (clean training run with no prior diagnosis/dept training), create it.
    metadata_path = DOCTOR_V3_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"  Merging into existing metadata.json")
    else:
        metadata = {}
        print(f"  Creating new metadata.json (no prior diag/dept training found)")

    metadata["disposition"] = {
        "trained_at": datetime.now().isoformat(),
        "version": "3",
        "model_type": "CalibratedClassifierCV(XGBClassifier, isotonic)",
        "objective": "binary:logistic",
        "calibration": "isotonic",
        "leakage_guard": "longitudinal vitals window [intime, intime + 4h]",
        "label": "disposition == 'ADMITTED'",
        "training_set": "FULL 425K (admitted + non-admitted)",
        "soft_cascade_cols": SOFT_CASCADE_COLS,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "pmh_categories": PMH_CATEGORIES,
        "pmh_feature_cols": PMH_FEATURE_COLS,
        "pmh_no_prior_days": PMH_NO_PRIOR_DAYS,
        "longitudinal_window_hours": _LONG_WINDOW_HOURS,
        "rhythm_buckets": _RHYTHM_BUCKETS,
        "long_vital_feature_cols": LONG_VITAL_FEATURE_COLS,
        "metrics": metrics,
        "note": (
            "Doctor disposition v3 (Option B from plan section 3). A peer "
            "binary admit/discharge classifier alongside the diagnosis + "
            "department models. Trained on the FULL 425K dataset (admitted "
            "+ non-admitted) so it sees both classes in real-world ratio. "
            "Consumes the triage SOFT cascade (5 acuity softmax + 1 "
            "disposition probability) rather than the hard cascade used by "
            "the diagnosis/department models — softens reliance on the "
            "triage argmax and lets the model honestly weight borderline "
            "cases. Isotonic calibration on a 10% held-out fit slice "
            "(same A4 pattern as the department model)."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  Updated metadata.json with `disposition` block")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "#" * 60)
    print("  Doctor disposition v3 — Model Training Pipeline")
    print("  (Full 425K, soft cascade from triage, isotonic-calibrated)")
    print("#" * 60)

    df = load_and_clean_data()

    print(f"\n{'='*60}")
    print(f"  Using full dataset (no sub-sampling) — {len(df):,} rows")
    print(f"{'='*60}")

    features = build_features(df)
    y = df["admitted"].astype(int).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("  Train/test split (80/20, stratified on admit label)")
    print(f"{'='*60}")
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train admit rate: {y_train.mean():.4f}  | "
          f"Test admit rate: {y_test.mean():.4f}")

    calibrated, raw, metrics = train_disposition(
        X_train, y_train, X_test, y_test,
    )

    feature_cols = list(features.columns)
    save_disposition_model(calibrated, raw, metrics, feature_cols)

    print(f"\n{'#'*60}")
    print("  Doctor disposition v3 training complete")
    print(f"{'#'*60}")
    print(f"  Headline numbers (calibrated, 0.5 threshold):")
    print(f"    Accuracy   : {metrics['accuracy_calibrated']:.4f}  "
          f"(triage v3 iter 2 ref: 0.7798)")
    print(f"    ROC AUC    : {metrics['roc_auc_calibrated']:.4f}  "
          f"(triage v3 iter 2 ref: 0.8644)")
    print(f"    Under-triage: {metrics['under_triage_rate']:.4f}  "
          f"(triage v3 iter 2 ref: 0.1556)")
    print(f"    Over-triage : {metrics['over_triage_rate']:.4f}  "
          f"(triage v3 iter 2 ref: 0.1690)")


if __name__ == "__main__":
    main()
