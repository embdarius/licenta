"""
Data Pipeline for Triage Agent v3 — MIMIC-IV Emergency Department

Builds on v2 (v1 features + triage vital signs masked-for-walk-ins) by adding
the 19-feature PMH (Past Medical History) block first introduced by Doctor v3
nurse Change 1:

  - 13 binary pmh_<diagnosis_group> flags derived from the "Past Medical
    History" section of prior MIMIC discharge summaries (OR'd with prior-
    admission ICD codes via categorized_diagnosis.csv).
  - 6 prior-encounter numerics: n_prior_admissions, n_prior_ed_visits,
    days_since_last_admission, days_since_last_ed, same_complaint_as_prior,
    no_history.

Aggregation is delegated to `proiect_licenta.pmh_features.aggregate_pmh`, the
same helper used by `train_nurse_v3.py`. Leakage is zero by construction:
prior-admission and prior-ED filters use `prior_*time < current_intime`.

Trains two supervised models on the full triage dataset (~334K stays):
  1. Acuity model: ESI 1-5 (multi:softprob)
  2. Disposition model: ADMITTED vs DISCHARGED (binary:logistic, cascading on
     predicted_acuity)

Models saved to artifacts/triage/v3/ — v1 and v2 artifacts remain untouched.
"""

import os
import json
import warnings
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score
from xgboost import XGBClassifier

# Shared text normalization (peers with v1, v2)
from proiect_licenta.preprocessing import normalize_complaint_text, ABBREVIATIONS  # noqa: F401

# PMH aggregator extracted from train_nurse_v3 — same feature set, same
# leakage guarantees. Triage v3 reuses it on the FULL ED dataset (not just
# admitted), so the model sees PMH lift on both acuity and disposition.
from proiect_licenta.pmh_features import (
    PMH_FEATURE_COLS,
    PMH_NO_PRIOR_DAYS,
    aggregate_pmh,
    fill_missing_pmh_columns,
)
# The PMH aggregator needs the ICD → diagnosis_group mapping from the doctor
# pipeline (it's the canonical mapping; pmh_features doesn't redefine it).
from proiect_licenta.training.train_doctor import (
    CATCH_ALL_LABEL,
    DIAGNOSIS_GROUP_MAP,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# XGBoost device / tree method — env-var-driven so the same pipeline runs on
# Colab GPU without code changes. Defaults reproduce the CPU runtime exactly.
#   XGB_DEVICE       "cpu" (default) | "cuda" | "cuda:0" ...
#   XGB_TREE_METHOD  "hist" (default; works on both CPU and GPU in xgboost >= 2.0)
# Set both via `os.environ` (or `export`) before `uv run train_triage_v3`.
# ---------------------------------------------------------------------------
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")
XGB_TREE_METHOD = os.environ.get("XGB_TREE_METHOD", "hist")

# ---------------------------------------------------------------------------
# Optional progress bars (tqdm.auto picks Jupyter widgets in Colab, plain bar
# in a terminal). If tqdm isn't installed, fall back to a no-op wrapper so
# local `uv run train_triage_v3` still works without an extra dependency.
# ---------------------------------------------------------------------------
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore

    def tqdm(it, **kw):
        return _tqdm(it, **kw)
except ImportError:  # pragma: no cover

    def tqdm(it, **kw):  # type: ignore[misc]
        return it


def _make_xgb_progress_callback(n_total: int, desc: str):
    """Return an XGBoost TrainingCallback that updates a tqdm bar per iteration.

    Falls back to ``None`` if tqdm or the XGBoost callback API isn't available
    (older XGBoost versions), in which case .fit() runs silently as before.

    The tqdm postfix shows the latest validation-metric value so the user can
    see eval metrics in real time during training (in the Colab cell).
    """
    try:
        from tqdm.auto import tqdm as _tqdm_cls
        from xgboost.callback import TrainingCallback
    except ImportError:
        return None

    class _TqdmCallback(TrainingCallback):  # type: ignore[misc]
        def __init__(self):
            self.pbar = _tqdm_cls(
                total=n_total, desc=desc, unit="iter", leave=False,
            )

        def after_iteration(self, model, epoch, evals_log):
            self.pbar.update(1)
            # evals_log structure: {'validation_0': {'metric_name': [v1, v2, ...]}}
            postfix = {}
            for ds in evals_log:
                for metric, vals in evals_log[ds].items():
                    if vals:
                        postfix[metric] = f"{vals[-1]:.4f}"
            if postfix and hasattr(self.pbar, "set_postfix"):
                self.pbar.set_postfix(postfix, refresh=False)
            return False  # do not stop training

        def after_training(self, model):
            self.pbar.close()
            return model

    return _TqdmCallback()


# ---------------------------------------------------------------------------
# Section 1.2 — ordinal-aware acuity weighting
# ---------------------------------------------------------------------------
# Multiply the existing sqrt-balanced sample weights by these per-class
# factors to push the acuity head to pay more attention to clinically critical
# / rare extremes. The plain `multi:softprob` objective treats every misclass-
# ification the same; sample-weighting biases the gradient updates toward the
# classes we care about most.
#
#   ESI 1  (resuscitation, ~1.1% of stays):  1.5x — critical; under-triage = death risk
#   ESI 2  (emergent, ~33% of stays):        1.3x — most clinically dangerous to miss
#   ESI 3  (urgent, dominant class):         1.0x — baseline
#   ESI 4  (less urgent):                    1.0x — baseline
#   ESI 5  (non-urgent, ~0.3% of stays):     2.0x — current recall 14% on v3 base
#
# Combined with the QWK eval metric below (which steers early stopping toward
# ordinal-friendly iterations), this gives the acuity head a soft ordinal
# objective without requiring a custom loss function.
ESI_EXTREME_BOOST = {1: 1.5, 2: 1.3, 3: 1.0, 4: 1.0, 5: 2.0}


def neg_quadratic_kappa(y_true, y_pred):
    """Negative quadratic-weighted kappa for XGBoost minimization-based early stopping.

    XGBoost minimizes its eval metric by default. Returning -κ means "lowest
    -κ" = "highest κ", so early stopping picks the iteration with the best
    ordinal agreement. Used as the acuity model's eval_metric in place of
    plain mlogloss — early stopping now optimizes for the ordinal structure
    of ESI 1-5 rather than treating every class-confusion identically.

    y_pred has shape (n_samples, n_classes) — softmax probabilities. argmax
    gives the predicted 0-indexed class; y_true is also 0-indexed at fit-time
    (we shift ESI 1-5 → 0-4 before calling .fit()), so κ comparisons are
    consistent. κ is invariant to label-set shifts anyway.
    """
    y_pred_class = np.argmax(y_pred, axis=1)
    return -cohen_kappa_score(y_true, y_pred_class, weights="quadratic")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_CAP = None  # None = use full dataset

# Section 1.1 — train both heads longer with a lower learning rate. The v3
# baseline hit best_iteration = 2999/3000 (acuity) and 2983/3000 (disposition)
# — neither head had converged. Halving lr and giving the booster more trees
# lets it actually find a minimum.
ACUITY_N_ESTIMATORS = 5000
ACUITY_LEARNING_RATE = 0.01
ACUITY_EARLY_STOPPING_ROUNDS = 150

DISP_N_ESTIMATORS = 5000
DISP_LEARNING_RATE = 0.01
DISP_EARLY_STOPPING_ROUNDS = 150

# ---------------------------------------------------------------------------
# Paths (canonical layout in proiect_licenta.paths)
# ---------------------------------------------------------------------------
from proiect_licenta.paths import (
    TRIAGE_V3_DIR as MODELS_DIR,
    TRIAGE_CSV, EDSTAYS_CSV, PATIENTS_CSV,
    DIAGNOSIS_CSV, DIAGNOSES_ICD_CSV, ADMISSIONS_CSV, DISCHARGE_NOTES_CSV,
)


# ---------------------------------------------------------------------------
# Vital sign constants (shared with v2)
# ---------------------------------------------------------------------------
VITAL_CLIP_RANGES = {
    "temperature": (90.0, 110.0),
    "heartrate":   (20.0, 250.0),
    "resprate":    (4.0, 60.0),
    "o2sat":       (50.0, 100.0),
    "sbp":         (50.0, 300.0),
    "dbp":         (20.0, 200.0),
}

VITAL_COLS = list(VITAL_CLIP_RANGES.keys())

ABNORMALITY_THRESHOLDS = {
    "fever":        ("temperature", ">",  100.4),
    "hypothermic":  ("temperature", "<",  96.8),
    "tachycardic":  ("heartrate",   ">",  100),
    "bradycardic":  ("heartrate",   "<",  60),
    "tachypneic":   ("resprate",    ">",  20),
    "hypoxic":      ("o2sat",       "<",  94),
    "hypertensive": ("sbp",         ">",  140),
    "hypotensive":  ("sbp",         "<",  90),
}


# ---------------------------------------------------------------------------
# 1. Load & Clean Data
# ---------------------------------------------------------------------------
def load_and_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load triage + edstays + patients, clean, merge, process vitals, and
    aggregate PMH features.

    Returns (df, edstays_full). `df` is the per-stay working frame with PMH
    columns merged in. `edstays_full` is returned only so callers (e.g.
    benchmarks) can re-run the same train/test split deterministically.
    """
    print("=" * 60)
    print("STEP 1: Loading data (v3 — v2 features + PMH)...")
    print("=" * 60)

    triage = pd.read_csv(TRIAGE_CSV)
    print(f"  triage.csv: {len(triage):,} rows")

    # Load FULL edstays — aggregate_pmh needs every ED visit per subject
    # for n_prior_ed_visits and same_complaint_as_prior. The "current"
    # working set is derived after merging with triage; the prior-encounter
    # lookups walk the unfiltered table.
    edstays = pd.read_csv(
        EDSTAYS_CSV,
        usecols=["subject_id", "stay_id", "hadm_id", "intime", "gender",
                 "arrival_transport", "disposition"],
    )
    print(f"  edstays.csv: {len(edstays):,} rows")

    patients = pd.read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "anchor_age", "anchor_year"],
    )
    print(f"  patients.csv: {len(patients):,} rows")

    df = triage.merge(edstays, on=["subject_id", "stay_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="left")
    print(f"  Merged: {len(df):,} rows")

    # --- Compute age ---
    df["intime"] = pd.to_datetime(df["intime"])
    df["visit_year"] = df["intime"].dt.year
    df["age"] = df["anchor_age"] + (df["visit_year"] - df["anchor_year"])
    df["age"] = df["age"].clip(0, 120).fillna(50).astype(int)

    # --- Clean chief complaints & acuity ---
    initial_count = len(df)
    df = df.dropna(subset=["chiefcomplaint", "acuity"])
    df = df[df["chiefcomplaint"].str.strip() != ""]
    print(f"  After dropping missing complaints/acuity: {len(df):,} "
          f"(dropped {initial_count - len(df):,})")

    df["acuity"] = df["acuity"].astype(int)
    df = df[df["acuity"].between(1, 5)]

    # --- Clean pain ---
    df["pain"] = pd.to_numeric(df["pain"], errors="coerce")
    df["pain_missing"] = df["pain"].isna().astype(int)
    df["pain"] = df["pain"].fillna(-1).astype(int)
    df.loc[df["pain"] > 10, "pain"] = -1
    df.loc[df["pain"] < 0, "pain"] = -1

    # --- Gender / arrival / disposition ---
    df["gender_male"] = (df["gender"] == "M").astype(int)
    df["arrival_ambulance"] = (df["arrival_transport"] == "AMBULANCE").astype(int)
    df["arrival_helicopter"] = (df["arrival_transport"] == "HELICOPTER").astype(int)
    df["arrival_walk_in"] = (df["arrival_transport"] == "WALK IN").astype(int)
    df["admitted"] = (df["disposition"] == "ADMITTED").astype(int)

    # --- Clean vital signs (same physiological clipping as v2) ---
    print(f"\n  Vital sign processing:")
    for col in VITAL_COLS:
        raw_missing = df[col].isna().sum()
        lo, hi = VITAL_CLIP_RANGES[col]
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        total_missing = df[col].isna().sum()
        print(f"    {col}: {raw_missing:,} raw NaN + {out_of_range:,} out-of-range "
              f"-> {total_missing:,} total missing ({100*total_missing/len(df):.1f}%)")

    # --- Mask vitals for non-ambulance/helicopter patients ---
    walkin_mask = ~df["arrival_transport"].isin(["AMBULANCE", "HELICOPTER"])
    n_masked = walkin_mask.sum()
    print(f"\n  Masking vitals for non-ambulance/helicopter patients:")
    print(f"    {n_masked:,} patients ({100*n_masked/len(df):.1f}%) -> vitals set to NaN")
    for col in VITAL_COLS:
        df.loc[walkin_mask, col] = np.nan

    # --- Cap rows ---
    if TRAIN_CAP and len(df) > TRAIN_CAP:
        print(f"\n  Capping to {TRAIN_CAP:,} rows (stratified by acuity)...")
        df, _ = train_test_split(
            df, train_size=TRAIN_CAP, random_state=42, stratify=df["acuity"],
        )
        print(f"  After cap: {len(df):,} rows")

    # ── PMH features (Doctor v3 nurse Change 1 recipe applied to the FULL
    # triage dataset, not just admitted) ──
    print("\n  Aggregating PMH features from prior encounters...")
    df = df.reset_index(drop=True)
    df["complaint_text_norm"] = df["chiefcomplaint"].apply(normalize_complaint_text)
    pmh_df = aggregate_pmh(
        stays_df=df[["stay_id", "subject_id", "intime", "complaint_text_norm"]],
        edstays_full=edstays,
        diagnoses_icd_csv_path=DIAGNOSES_ICD_CSV,
        admissions_csv_path=ADMISSIONS_CSV,
        discharge_csv_path=DISCHARGE_NOTES_CSV,
        diagnosis_csv_path=DIAGNOSIS_CSV,
        diagnosis_group_map=DIAGNOSIS_GROUP_MAP,
        catch_all_label=CATCH_ALL_LABEL,
    )
    df = df.merge(pmh_df, on="stay_id", how="left")
    fill_missing_pmh_columns(df)

    # --- Distributions ---
    print(f"\n  Acuity distribution:")
    for level in sorted(df["acuity"].unique()):
        count = (df["acuity"] == level).sum()
        print(f"    ESI {level}: {count:,} ({100 * count / len(df):.1f}%)")

    print(f"\n  Disposition distribution:")
    admitted = df["admitted"].sum()
    print(f"    ADMITTED:     {admitted:,} ({100 * admitted / len(df):.1f}%)")
    print(f"    NOT ADMITTED: {len(df) - admitted:,} "
          f"({100 * (len(df) - admitted) / len(df):.1f}%)")

    pmh_any = (df[[c for c in PMH_FEATURE_COLS if c.startswith("pmh_")]].sum(axis=1) > 0).mean()
    has_prior = (df["no_history"] == 0).mean()
    print(f"\n  PMH coverage:")
    print(f"    Stays with ≥1 PMH flag:    {100*pmh_any:.1f}%")
    print(f"    Stays with ≥1 prior visit: {100*has_prior:.1f}%")

    return df, edstays


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    tfidf: TfidfVectorizer = None,
    severity_map: dict = None,
    vital_medians: dict = None,
    fit: bool = True,
) -> tuple:
    """Build full feature matrix: v1 structured + v2 vitals + v3 PMH + TF-IDF."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Feature engineering ({'fitting' if fit else 'transforming'})")
    print("=" * 60)

    df = df.copy()
    df["complaint_text"] = df["chiefcomplaint"].apply(normalize_complaint_text)

    df["n_complaints"] = df["chiefcomplaint"].apply(
        lambda x: len([c.strip() for c in str(x).split(",") if c.strip()])
    )
    df["complaint_length"] = df["complaint_text"].apply(len)

    # --- TF-IDF ---
    if fit:
        tfidf = TfidfVectorizer(
            max_features=2000,
            min_df=20,
            max_df=0.95,
            ngram_range=(1, 3),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        tfidf_matrix = tfidf.fit_transform(df["complaint_text"])
        print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
    else:
        tfidf_matrix = tfidf.transform(df["complaint_text"])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        index=df.index,
    )

    # --- Severity Priors ---
    if fit:
        word_acuity = defaultdict(list)
        texts = df["complaint_text"].values
        acuities = df["acuity"].values
        for i in range(len(texts)):
            for word in str(texts[i]).split():
                word = word.strip()
                if len(word) > 1:
                    word_acuity[word].append(acuities[i])
        severity_map = {
            w: np.mean(vals)
            for w, vals in word_acuity.items()
            if len(vals) >= 20
        }
        print(f"  Severity prior vocabulary: {len(severity_map)} words")

    def compute_severity_priors(text: str) -> tuple:
        words = text.split()
        severities = [severity_map[w] for w in words if w in severity_map]
        if severities:
            return (min(severities), np.mean(severities),
                    max(severities),
                    np.std(severities) if len(severities) > 1 else 0.0)
        return 3.0, 3.0, 3.0, 0.0

    severity_features = df["complaint_text"].apply(compute_severity_priors)
    df["min_severity_prior"] = severity_features.apply(lambda x: x[0])
    df["mean_severity_prior"] = severity_features.apply(lambda x: x[1])
    df["max_severity_prior"] = severity_features.apply(lambda x: x[2])
    df["std_severity_prior"] = severity_features.apply(lambda x: x[3])

    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 50, 65, 80, 120],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float).fillna(2)

    df["pain_low"] = ((df["pain"] >= 0) & (df["pain"] <= 3)).astype(int)
    df["pain_mid"] = ((df["pain"] >= 4) & (df["pain"] <= 6)).astype(int)
    df["pain_high"] = ((df["pain"] >= 7) & (df["pain"] <= 10)).astype(int)

    df["age_ambulance"] = df["age"] * df["arrival_ambulance"]
    df["pain_x_min_severity"] = df["pain"].clip(0, 10) * (5 - df["min_severity_prior"])
    df["age_severity"] = df["age"] * (5 - df["min_severity_prior"])
    df["high_pain_ambulance"] = df["pain_high"] * df["arrival_ambulance"]
    df["elderly"] = (df["age"] >= 65).astype(int)
    df["elderly_ambulance"] = df["elderly"] * df["arrival_ambulance"]

    # --- v2 vital sign features ---
    for col in VITAL_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    if fit:
        vital_medians = {col: float(df[col].median()) for col in VITAL_COLS}
        print(f"  Vital medians (training): {vital_medians}")
    for col in VITAL_COLS:
        df[col] = df[col].fillna(vital_medians[col])

    for flag_name, (col, op, threshold) in ABNORMALITY_THRESHOLDS.items():
        if op == ">":
            df[flag_name] = (df[col] > threshold).astype(int)
        else:
            df[flag_name] = (df[col] < threshold).astype(int)

    abnormality_flag_names = list(ABNORMALITY_THRESHOLDS.keys())
    df["abnormal_vital_count"] = df[abnormality_flag_names].sum(axis=1)

    df["tachycardic_ambulance"] = df["tachycardic"] * df["arrival_ambulance"]
    df["hypoxic_ambulance"] = df["hypoxic"] * df["arrival_ambulance"]
    df["hypotensive_ambulance"] = df["hypotensive"] * df["arrival_ambulance"]
    df["fever_ambulance"] = df["fever"] * df["arrival_ambulance"]

    df["tachycardic_elderly"] = df["tachycardic"] * df["elderly"]
    df["hypoxic_elderly"] = df["hypoxic"] * df["elderly"]
    df["hypotensive_elderly"] = df["hypotensive"] * df["elderly"]

    # ===================================================================
    # Assemble feature vector
    # ===================================================================
    v1_structured_cols = [
        "pain", "pain_missing", "pain_low", "pain_mid", "pain_high",
        "n_complaints", "complaint_length",
        "min_severity_prior", "mean_severity_prior",
        "max_severity_prior", "std_severity_prior",
        "age", "age_bin", "gender_male",
        "arrival_ambulance", "arrival_helicopter", "arrival_walk_in",
        "age_ambulance", "pain_x_min_severity", "age_severity",
        "high_pain_ambulance", "elderly", "elderly_ambulance",
    ]

    v2_vital_cols = (
        VITAL_COLS
        + [f"{col}_missing" for col in VITAL_COLS]
        + abnormality_flag_names
        + ["abnormal_vital_count"]
        + ["tachycardic_ambulance", "hypoxic_ambulance",
           "hypotensive_ambulance", "fever_ambulance"]
        + ["tachycardic_elderly", "hypoxic_elderly", "hypotensive_elderly"]
    )

    # v3 PMH cols (already on df via fill_missing_pmh_columns)
    v3_pmh_cols = list(PMH_FEATURE_COLS)

    all_structured_cols = v1_structured_cols + v2_vital_cols + v3_pmh_cols

    structured = df[all_structured_cols].reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)

    features = pd.concat([structured, tfidf_df], axis=1)

    print(f"  v1 structured features: {len(v1_structured_cols)}")
    print(f"  v2 vital features: {len(v2_vital_cols)}")
    print(f"  v3 PMH features: {len(v3_pmh_cols)}")
    print(f"  TF-IDF features: {tfidf_df.shape[1]}")
    print(f"  Total features: {features.shape[1]}")

    return features, tfidf, severity_map, vital_medians


# ---------------------------------------------------------------------------
# 3. Train Models
# ---------------------------------------------------------------------------
def train_acuity_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier:
    """Train the acuity head with section 1.1 (longer training) + section 1.2
    (ordinal-aware sample weights + QWK early stopping) applied on top of the
    v3 PMH-augmented feature set.

    Hyperparameters vs the initial v3 iteration:
      n_estimators       3000 → 5000        (section 1.1)
      learning_rate      0.02 → 0.01        (section 1.1)
      early_stopping     100  → 150         (section 1.1)
      sample_weight      sqrt-balanced × ESI_EXTREME_BOOST  (section 1.2)
      eval_metric        mlogloss → neg_quadratic_kappa     (section 1.2)
    """
    print("\n" + "=" * 60)
    print("STEP 4a: Training ACUITY model (XGBoost, ESI 1-5)")
    print(f"         Section 1.1 (longer training) + Section 1.2 (ordinal weights + QWK)")
    print("=" * 60)

    class_counts = y_train.value_counts()
    total = len(y_train)
    n_classes = len(class_counts)
    base_weights = y_train.map(
        lambda x: np.sqrt(total / (n_classes * class_counts[x]))
    )
    extreme_factor = y_train.map(ESI_EXTREME_BOOST)
    sample_weights = base_weights * extreme_factor

    print(f"  Class distribution + weighting:")
    for esi in sorted(class_counts.index):
        n = int(class_counts[esi])
        base = float(np.sqrt(total / (n_classes * n)))
        boost = ESI_EXTREME_BOOST[esi]
        print(f"    ESI {esi}: n={n:>7,}  base_w={base:.3f}  boost={boost:.2f}x  "
              f"final_w={base * boost:.3f}")

    y_train_shifted = y_train - 1
    y_test_shifted = y_test - 1

    progress_cb = _make_xgb_progress_callback(
        n_total=ACUITY_N_ESTIMATORS, desc="    XGBoost (acuity)",
    )
    callbacks = [progress_cb] if progress_cb is not None else None

    model = XGBClassifier(
        n_estimators=ACUITY_N_ESTIMATORS,
        max_depth=10,
        learning_rate=ACUITY_LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.5,
        colsample_bylevel=0.7,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=5,
        eval_metric=neg_quadratic_kappa,
        early_stopping_rounds=ACUITY_EARLY_STOPPING_ROUNDS,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        device=XGB_DEVICE,
        tree_method=XGB_TREE_METHOD,
        callbacks=callbacks,
    )

    print(f"\n  Training XGBoost (up to {ACUITY_N_ESTIMATORS} trees, "
          f"lr={ACUITY_LEARNING_RATE}, early stopping={ACUITY_EARLY_STOPPING_ROUNDS}; "
          f"device={XGB_DEVICE}, tree_method={XGB_TREE_METHOD}, eval=neg_qwk)...")
    # verbose=100 prints a one-line update every 100 boosting iterations
    # in addition to the tqdm bar — gives the user a textual progress trail
    # in the notebook output even if the tqdm widget didn't render.
    model.fit(
        X_train, y_train_shifted,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test_shifted)],
        verbose=100,
    )

    print(f"  Best iteration: {model.best_iteration}  "
          f"(eval metric was -κ, so best = highest κ)")

    # Strip the tqdm callback so joblib.dump can pickle the model — the bar
    # holds an open file handle that pickle chokes on. Only matters during
    # .fit(); .predict() / .predict_proba() never touch it.
    try:
        model.set_params(callbacks=None)
    except Exception:
        pass

    y_pred = model.predict(X_test) + 1
    accuracy = accuracy_score(y_test, y_pred)
    within_1 = np.mean(np.abs(y_pred - y_test.values) <= 1)
    kappa_q = cohen_kappa_score(y_test, y_pred, weights="quadratic")

    print(f"\n  Accuracy (exact):              {accuracy:.4f}")
    print(f"  Accuracy (within 1 ESI level): {within_1:.4f}")
    print(f"  Cohen's κ (quadratic):         {kappa_q:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    return model


def train_disposition_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier:
    """Train the disposition head with section 1.1 (longer training).

    Section 1.2 (ordinal weights, QWK eval) is acuity-only — disposition is
    binary and admit/discharge has no ordinal structure.

    Hyperparameters vs the initial v3 iteration:
      n_estimators       3000 → 5000        (section 1.1)
      learning_rate      0.02 → 0.01        (section 1.1)
      early_stopping     100  → 150         (section 1.1)
    """
    print("\n" + "=" * 60)
    print("STEP 4b: Training DISPOSITION model (XGBoost) — Section 1.1 longer training")
    print("=" * 60)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count

    progress_cb = _make_xgb_progress_callback(
        n_total=DISP_N_ESTIMATORS, desc="    XGBoost (disposition)",
    )
    callbacks = [progress_cb] if progress_cb is not None else None

    model = XGBClassifier(
        n_estimators=DISP_N_ESTIMATORS,
        max_depth=10,
        learning_rate=DISP_LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.5,
        colsample_bylevel=0.7,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=DISP_EARLY_STOPPING_ROUNDS,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        device=XGB_DEVICE,
        tree_method=XGB_TREE_METHOD,
        callbacks=callbacks,
    )

    print(f"  Training XGBoost (up to {DISP_N_ESTIMATORS} trees, "
          f"lr={DISP_LEARNING_RATE}, early stopping={DISP_EARLY_STOPPING_ROUNDS}; "
          f"device={XGB_DEVICE}, tree_method={XGB_TREE_METHOD})...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    print(f"  Best iteration: {model.best_iteration}")

    try:
        model.set_params(callbacks=None)
    except Exception:
        pass

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["NOT ADMITTED", "ADMITTED"],
        digits=3,
    ))

    return model


# ---------------------------------------------------------------------------
# 4. Save Artifacts
# ---------------------------------------------------------------------------
def save_models(
    acuity_model,
    disposition_model,
    tfidf: TfidfVectorizer,
    severity_map: dict,
    vital_medians: dict,
    acuity_accuracy: float,
    within_1_accuracy: float,
    disposition_accuracy: float,
    n_features: int,
):
    print("\n" + "=" * 60)
    print("STEP 5: Saving model artifacts")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Strip the custom eval_metric callable before pickling. Training-time only:
    # neg_quadratic_kappa is a function defined in this module, so when training
    # is invoked via `runpy.run_path(..., run_name="__main__")` (the Colab flow),
    # the callable gets pickled with __module__="__main__". At load time pickle
    # cannot resolve __main__.neg_quadratic_kappa unless the caller manually
    # injects it. Stripping eval_metric here makes the saved model loadable
    # cleanly via plain joblib.load() — eval_metric is fit-time only and is
    # never consulted at predict() / predict_proba() time anyway.
    try:
        acuity_model.set_params(eval_metric=None)
    except Exception:
        pass

    joblib.dump(acuity_model, MODELS_DIR / "acuity_model.joblib")
    joblib.dump(disposition_model, MODELS_DIR / "disposition_model.joblib")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(dict(severity_map), MODELS_DIR / "severity_map.joblib")
    joblib.dump(vital_medians, MODELS_DIR / "vital_medians.joblib")

    metadata = {
        "version": "5-v3",
        "iteration": "1.1+1.2 (longer training + ordinal-aware acuity)",
        "trained_at": datetime.now().isoformat(),
        "train_cap": TRAIN_CAP,
        "n_total_features": n_features,
        "n_tfidf_features": len(tfidf.vocabulary_),
        "n_severity_words": len(severity_map),
        "vital_medians": vital_medians,
        "vital_clip_ranges": VITAL_CLIP_RANGES,
        "abnormality_thresholds": {
            k: {"col": v[0], "op": v[1], "threshold": v[2]}
            for k, v in ABNORMALITY_THRESHOLDS.items()
        },
        "pmh_feature_cols": PMH_FEATURE_COLS,
        "pmh_no_prior_days": PMH_NO_PRIOR_DAYS,
        "acuity_classes": [1, 2, 3, 4, 5],
        "disposition_classes": ["NOT ADMITTED", "ADMITTED"],
        "acuity_accuracy_exact": round(acuity_accuracy, 4),
        "acuity_accuracy_within_1": round(within_1_accuracy, 4),
        "disposition_accuracy": round(disposition_accuracy, 4),
        "model_type": "XGBClassifier",
        "training_config": {
            "acuity_n_estimators": ACUITY_N_ESTIMATORS,
            "acuity_learning_rate": ACUITY_LEARNING_RATE,
            "acuity_early_stopping_rounds": ACUITY_EARLY_STOPPING_ROUNDS,
            "acuity_eval_metric": "neg_quadratic_kappa (section 1.2)",
            "acuity_sample_weight_strategy": "sqrt(N/(K*count)) * ESI_EXTREME_BOOST",
            "esi_extreme_boost": ESI_EXTREME_BOOST,
            "disp_n_estimators": DISP_N_ESTIMATORS,
            "disp_learning_rate": DISP_LEARNING_RATE,
            "disp_early_stopping_rounds": DISP_EARLY_STOPPING_ROUNDS,
            "disp_eval_metric": "logloss",
        },
        "note": "Triage v3 iteration 2 (2026-05-27): v2 features + 19 PMH features "
                "+ section 1.1 longer training (lr 0.02->0.01, n_est 3000->5000) "
                "+ section 1.2 ordinal-aware acuity (extreme-class boost + QWK early stopping). "
                "Acuity model outputs classes 0-4 (add 1 to get ESI 1-5).",
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {MODELS_DIR}")
    for fname in ["acuity_model.joblib", "disposition_model.joblib",
                   "tfidf_vectorizer.joblib", "severity_map.joblib",
                   "vital_medians.joblib", "model_metadata.json"]:
        print(f"  - {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "#" * 60)
    print("  MIMIC-IV Triage Model Training Pipeline v3")
    cap_str = f"{TRAIN_CAP:,}" if TRAIN_CAP else "full dataset"
    print(f"  (v2 features + PMH, {cap_str} rows)")
    print("#" * 60)

    df, _edstays = load_and_clean_data()

    print("\n" + "=" * 60)
    print("STEP 3: Train/test split (80/20, stratified by acuity)")
    print("=" * 60)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["acuity"],
    )
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    X_train, tfidf, severity_map, vital_medians = build_features(
        train_df, fit=True,
    )
    y_acuity_train = train_df["acuity"].reset_index(drop=True)
    y_admit_train = train_df["admitted"].reset_index(drop=True)

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        X_test, _, _, _ = build_features(
            test_df, tfidf=tfidf, severity_map=severity_map,
            vital_medians=vital_medians, fit=False,
        )
    y_acuity_test = test_df["acuity"].reset_index(drop=True)
    y_admit_test = test_df["admitted"].reset_index(drop=True)

    acuity_model = train_acuity_model(
        X_train, y_acuity_train, X_test, y_acuity_test,
    )
    acuity_preds = acuity_model.predict(X_test) + 1
    acuity_accuracy = accuracy_score(y_acuity_test, acuity_preds)
    within_1 = float(np.mean(np.abs(acuity_preds - y_acuity_test.values) <= 1))

    X_train_disp = X_train.copy()
    X_train_disp["predicted_acuity"] = acuity_model.predict(X_train) + 1
    X_test_disp = X_test.copy()
    X_test_disp["predicted_acuity"] = acuity_model.predict(X_test) + 1

    disposition_model = train_disposition_model(
        X_train_disp, y_admit_train, X_test_disp, y_admit_test,
    )
    disp_accuracy = accuracy_score(
        y_admit_test, disposition_model.predict(X_test_disp),
    )

    save_models(
        acuity_model, disposition_model,
        tfidf, severity_map, vital_medians,
        acuity_accuracy, within_1, disp_accuracy,
        n_features=X_train.shape[1],
    )

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE!")
    print("#" * 60)
    print(f"  Training rows (before split): {len(df):,}")
    print(f"  Total features: {X_train.shape[1]}")
    print(f"  Acuity accuracy (exact):    {acuity_accuracy:.4f}")
    print(f"  Acuity accuracy (within 1): {within_1:.4f}")
    print(f"  Disposition accuracy:       {disp_accuracy:.4f}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
