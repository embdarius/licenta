"""
Data Pipeline for Triage Agent v2 — MIMIC-IV Emergency Department

Builds on v1 (TF-IDF + pain + demographics + arrival transport + severity priors
+ interaction features) by adding triage vital signs:
  - 6 raw vitals: temperature, heartrate, resprate, o2sat, sbp, dbp
  - 6 missing flags (one per vital)
  - 8 clinical abnormality flags (fever, hypothermic, tachycardic, bradycardic,
    tachypneic, hypoxic, hypertensive, hypotensive)
  - Interaction features: vitals x ambulance, vitals x elderly, abnormal_vital_count

Realistic training: vitals are masked to NaN for non-ambulance/helicopter patients
before feature engineering, matching inference behavior where only EMS patients
have vitals at triage.

Trains two supervised models:
  1. Acuity: all features → ESI 1-5
  2. Disposition: same + predicted acuity → ADMITTED vs DISCHARGED

Models saved to artifacts/triage/v2/ to keep v1 artifacts intact.
"""

import os
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Shared text normalization (peers with v1)
from proiect_licenta.preprocessing import normalize_complaint_text, ABBREVIATIONS  # noqa: F401

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_CAP = None  # None = use full dataset

# ---------------------------------------------------------------------------
# Paths (canonical layout in proiect_licenta.paths)
# ---------------------------------------------------------------------------
from proiect_licenta.paths import (
    TRIAGE_V2_DIR as MODELS_DIR,
    TRIAGE_CSV, EDSTAYS_CSV, PATIENTS_CSV,
)


# ---------------------------------------------------------------------------
# Vital sign constants
# ---------------------------------------------------------------------------
# Physiologically plausible ranges — values outside are treated as data entry errors
VITAL_CLIP_RANGES = {
    "temperature": (90.0, 110.0),   # Fahrenheit (data is in °F, median ~98)
    "heartrate":   (20.0, 250.0),
    "resprate":    (4.0, 60.0),
    "o2sat":       (50.0, 100.0),
    "sbp":         (50.0, 300.0),
    "dbp":         (20.0, 200.0),
}

VITAL_COLS = list(VITAL_CLIP_RANGES.keys())

# Clinical thresholds for abnormality flags
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
def load_and_clean_data() -> pd.DataFrame:
    """Load triage + edstays + patients, clean, merge, and process vitals."""
    print("=" * 60)
    print("STEP 1: Loading data...")
    print("=" * 60)

    # Load triage (now including vital sign columns)
    triage = pd.read_csv(TRIAGE_CSV)
    print(f"  triage.csv: {len(triage):,} rows")

    # Load edstays
    edstays = pd.read_csv(
        EDSTAYS_CSV,
        usecols=["subject_id", "stay_id", "intime", "gender",
                 "arrival_transport", "disposition"],
    )
    print(f"  edstays.csv: {len(edstays):,} rows")

    # Load patients
    patients = pd.read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "anchor_age", "anchor_year"],
    )
    print(f"  patients.csv: {len(patients):,} rows")

    # Merge
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

    # --- Gender ---
    df["gender_male"] = (df["gender"] == "M").astype(int)

    # --- Arrival transport ---
    df["arrival_ambulance"] = (df["arrival_transport"] == "AMBULANCE").astype(int)
    df["arrival_helicopter"] = (df["arrival_transport"] == "HELICOPTER").astype(int)
    df["arrival_walk_in"] = (df["arrival_transport"] == "WALK IN").astype(int)

    # --- Disposition ---
    df["admitted"] = (df["disposition"] == "ADMITTED").astype(int)

    # --- Clean vital signs ---
    print(f"\n  Vital sign processing:")
    for col in VITAL_COLS:
        raw_missing = df[col].isna().sum()
        # Clip out-of-range to NaN (data entry errors)
        lo, hi = VITAL_CLIP_RANGES[col]
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        total_missing = df[col].isna().sum()
        print(f"    {col}: {raw_missing:,} raw NaN + {out_of_range:,} out-of-range "
              f"-> {total_missing:,} total missing ({100*total_missing/len(df):.1f}%)")

    # --- Mask vitals for non-ambulance/helicopter patients ---
    # At inference, only ambulance/helicopter patients have EMS vitals at triage.
    # Walk-in/other patients haven't had vitals measured yet. We mask them to NaN
    # during training so the model learns to handle missing vitals for walk-ins.
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

    # --- Print distributions ---
    print(f"\n  Acuity distribution:")
    for level in sorted(df["acuity"].unique()):
        count = (df["acuity"] == level).sum()
        print(f"    ESI {level}: {count:,} ({100 * count / len(df):.1f}%)")

    print(f"\n  Disposition distribution:")
    admitted = df["admitted"].sum()
    print(f"    ADMITTED:     {admitted:,} ({100 * admitted / len(df):.1f}%)")
    print(f"    NOT ADMITTED: {len(df) - admitted:,} "
          f"({100 * (len(df) - admitted) / len(df):.1f}%)")

    print(f"\n  Gender: M={df['gender_male'].sum():,}  "
          f"F={len(df) - df['gender_male'].sum():,}")
    print(f"  Age: mean={df['age'].mean():.1f}  "
          f"median={df['age'].median():.0f}  "
          f"range=[{df['age'].min()}, {df['age'].max()}]")
    print(f"\n  Arrival transport:")
    for t in ["AMBULANCE", "WALK IN", "HELICOPTER", "OTHER", "UNKNOWN"]:
        count = (df["arrival_transport"] == t).sum()
        if count > 0:
            print(f"    {t}: {count:,} ({100 * count / len(df):.1f}%)")

    return df


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
    """Build full feature matrix including vital signs."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Feature engineering ({'fitting' if fit else 'transforming'})")
    print("=" * 60)

    df = df.copy()
    df["complaint_text"] = df["chiefcomplaint"].apply(normalize_complaint_text)

    # --- Count of complaints ---
    df["n_complaints"] = df["chiefcomplaint"].apply(
        lambda x: len([c.strip() for c in str(x).split(",") if c.strip()])
    )

    # --- Complaint text length ---
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

    # --- Age bins ---
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 50, 65, 80, 120],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float).fillna(2)

    # --- Pain bins ---
    df["pain_low"] = ((df["pain"] >= 0) & (df["pain"] <= 3)).astype(int)
    df["pain_mid"] = ((df["pain"] >= 4) & (df["pain"] <= 6)).astype(int)
    df["pain_high"] = ((df["pain"] >= 7) & (df["pain"] <= 10)).astype(int)

    # --- v1 interaction features ---
    df["age_ambulance"] = df["age"] * df["arrival_ambulance"]
    df["pain_x_min_severity"] = df["pain"].clip(0, 10) * (5 - df["min_severity_prior"])
    df["age_severity"] = df["age"] * (5 - df["min_severity_prior"])
    df["high_pain_ambulance"] = df["pain_high"] * df["arrival_ambulance"]
    df["elderly"] = (df["age"] >= 65).astype(int)
    df["elderly_ambulance"] = df["elderly"] * df["arrival_ambulance"]

    # ===================================================================
    # NEW in v2: Vital sign features
    # ===================================================================

    # --- Missing flags (before imputation) ---
    for col in VITAL_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # --- Compute or apply medians for imputation ---
    if fit:
        vital_medians = {}
        for col in VITAL_COLS:
            vital_medians[col] = float(df[col].median())
        print(f"  Vital medians (training): {vital_medians}")
    for col in VITAL_COLS:
        df[col] = df[col].fillna(vital_medians[col])

    # --- Abnormality flags ---
    for flag_name, (col, op, threshold) in ABNORMALITY_THRESHOLDS.items():
        if op == ">":
            df[flag_name] = (df[col] > threshold).astype(int)
        else:
            df[flag_name] = (df[col] < threshold).astype(int)

    # --- Count of abnormal vitals ---
    abnormality_flag_names = list(ABNORMALITY_THRESHOLDS.keys())
    df["abnormal_vital_count"] = df[abnormality_flag_names].sum(axis=1)

    # --- Vital-transport interactions ---
    df["tachycardic_ambulance"] = df["tachycardic"] * df["arrival_ambulance"]
    df["hypoxic_ambulance"] = df["hypoxic"] * df["arrival_ambulance"]
    df["hypotensive_ambulance"] = df["hypotensive"] * df["arrival_ambulance"]
    df["fever_ambulance"] = df["fever"] * df["arrival_ambulance"]

    # --- Vital-age interactions ---
    df["tachycardic_elderly"] = df["tachycardic"] * df["elderly"]
    df["hypoxic_elderly"] = df["hypoxic"] * df["elderly"]
    df["hypotensive_elderly"] = df["hypotensive"] * df["elderly"]

    # ===================================================================
    # Assemble feature vector
    # ===================================================================
    # v1 structured columns (same order as v1)
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

    # v2 vital columns
    v2_vital_cols = (
        # Raw vitals (imputed)
        VITAL_COLS
        # Missing flags
        + [f"{col}_missing" for col in VITAL_COLS]
        # Abnormality flags
        + abnormality_flag_names
        # Abnormal count
        + ["abnormal_vital_count"]
        # Vital-transport interactions
        + ["tachycardic_ambulance", "hypoxic_ambulance",
           "hypotensive_ambulance", "fever_ambulance"]
        # Vital-age interactions
        + ["tachycardic_elderly", "hypoxic_elderly", "hypotensive_elderly"]
    )

    all_structured_cols = v1_structured_cols + v2_vital_cols

    structured = df[all_structured_cols].reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)

    features = pd.concat([structured, tfidf_df], axis=1)

    print(f"  v1 structured features: {len(v1_structured_cols)}")
    print(f"  v2 vital features: {len(v2_vital_cols)}")
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
    """Train acuity model with XGBoost + early stopping."""
    print("\n" + "=" * 60)
    print("STEP 4a: Training ACUITY model (XGBoost, ESI 1-5)")
    print("=" * 60)

    # Soft class weights
    class_counts = y_train.value_counts()
    total = len(y_train)
    n_classes = len(class_counts)
    sample_weights = y_train.map(
        lambda x: np.sqrt(total / (n_classes * class_counts[x]))
    )

    y_train_shifted = y_train - 1
    y_test_shifted = y_test - 1

    model = XGBClassifier(
        n_estimators=3000,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.5,
        colsample_bylevel=0.7,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("  Training XGBoost (up to 3000 trees, lr=0.02, early stopping=100)...")
    model.fit(
        X_train, y_train_shifted,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test_shifted)],
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    y_pred = model.predict(X_test) + 1
    accuracy = accuracy_score(y_test, y_pred)
    within_1 = np.mean(np.abs(y_pred - y_test.values) <= 1)

    print(f"\n  Accuracy (exact): {accuracy:.4f}")
    print(f"  Accuracy (within 1 ESI level): {within_1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    return model


def train_disposition_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier:
    """Train disposition model with XGBoost."""
    print("\n" + "=" * 60)
    print("STEP 4b: Training DISPOSITION model (XGBoost)")
    print("=" * 60)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count

    model = XGBClassifier(
        n_estimators=3000,
        max_depth=10,
        learning_rate=0.02,
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
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("  Training XGBoost (up to 3000 trees, lr=0.02, early stopping=100)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

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
    """Save trained models and metadata to artifacts/triage/v2/."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving model artifacts")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(acuity_model, MODELS_DIR / "acuity_model.joblib")
    joblib.dump(disposition_model, MODELS_DIR / "disposition_model.joblib")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(dict(severity_map), MODELS_DIR / "severity_map.joblib")
    joblib.dump(vital_medians, MODELS_DIR / "vital_medians.joblib")

    metadata = {
        "version": "4-v2",
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
        "acuity_classes": [1, 2, 3, 4, 5],
        "disposition_classes": ["NOT ADMITTED", "ADMITTED"],
        "acuity_accuracy_exact": round(acuity_accuracy, 4),
        "acuity_accuracy_within_1": round(within_1_accuracy, 4),
        "disposition_accuracy": round(disposition_accuracy, 4),
        "model_type": "XGBClassifier",
        "note": "Triage v2: v1 features + vital signs. "
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
    print("  MIMIC-IV Triage Model Training Pipeline v2")
    cap_str = f"{TRAIN_CAP:,}" if TRAIN_CAP else "full dataset"
    print(f"  (v1 features + Vital Signs, {cap_str} rows)")
    print("#" * 60)

    # 1. Load (includes row cap)
    df = load_and_clean_data()

    # 2. Split (80/20, stratified)
    print("\n" + "=" * 60)
    print("STEP 3: Train/test split (80/20, stratified by acuity)")
    print("=" * 60)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["acuity"],
    )
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # 3. Features
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

    # 4a. Acuity model
    acuity_model = train_acuity_model(
        X_train, y_acuity_train, X_test, y_acuity_test,
    )
    acuity_preds = acuity_model.predict(X_test) + 1
    acuity_accuracy = accuracy_score(y_acuity_test, acuity_preds)
    within_1 = float(np.mean(np.abs(acuity_preds - y_acuity_test.values) <= 1))

    # 4b. Disposition model (cascading: uses predicted acuity)
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

    # 5. Save
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
