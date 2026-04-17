"""
Data Pipeline for Doctor Agent — MIMIC-IV Emergency Department

Trains two supervised models for admitted patients:
  1. Diagnosis Category: chief complaints + demographics + triage results → 14 categories
  2. Department Prediction: same + predicted diagnosis → 11 hospital services

Uses identical text preprocessing as the triage pipeline.
Reuses saved triage model artifacts (TF-IDF vectorizer, severity map, acuity/disposition models)
to generate cascading features.
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

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "datasets_mimic-iv" / "mimic-iv-ed"
HOSP_DIR = BASE_DIR / "datasets" / "datasets_mimic-iv" / "mimic-iv" / "hosp"
TRIAGE_MODELS_DIR = BASE_DIR / "models"
DOCTOR_MODELS_DIR = BASE_DIR / "models" / "doctor"

TRIAGE_CSV = DATASET_DIR / "triage.csv"
EDSTAYS_CSV = DATASET_DIR / "edstays.csv"
PATIENTS_CSV = HOSP_DIR / "patients.csv"
DIAGNOSIS_CSV = DATASET_DIR / "files_created" / "categorized_diagnosis.csv"
SERVICES_CSV = HOSP_DIR / "services.csv"

# ---------------------------------------------------------------------------
# Category Grouping Maps
# ---------------------------------------------------------------------------

# 22 raw diagnosis categories → 14 grouped categories
DIAGNOSIS_GROUP_MAP = {
    # Keep as-is (large enough)
    "Symptoms, Signs, Ill-Defined Conditions": "Symptoms, Signs, Ill-Defined",
    "Circulatory System": "Circulatory",
    "Digestive System": "Digestive",
    "Injury and Poisoning": "Injury and Poisoning",
    "Respiratory System": "Respiratory",
    "Genitourinary System": "Genitourinary",
    "Musculoskeletal System": "Musculoskeletal",
    "Endocrine, Nutritional, Metabolic": "Endocrine, Nutritional, Metabolic",
    "Mental Disorders": "Mental Disorders",
    "Skin and Subcutaneous Tissue": "Skin and Subcutaneous Tissue",
    "Blood and Blood-Forming Organs": "Blood and Blood-Forming Organs",
    "Infectious and Parasitic Diseases": "Infectious and Parasitic",
    # Merge: Nervous System variants + Eye + Ear (ICD-9/10 artifact)
    "Nervous System": "Nervous System and Sense Organs",
    "Nervous System and Sense Organs": "Nervous System and Sense Organs",
    "Diseases of the Eye": "Nervous System and Sense Organs",
    "Diseases of the Ear": "Nervous System and Sense Organs",
    # Merge small categories into Other
    "Pregnancy, Childbirth": "Other",
    "Neoplasms": "Other",
    "Supplemental / Health Status": "Other",
    "Congenital Anomalies": "Other",
    "Invalid Code": "Other",
    "Perinatal Period Conditions": "Other",
}

# 19 raw services → 11 grouped services
SERVICE_GROUP_MAP = {
    # Keep as-is
    "MED": "MED",
    "CMED": "CMED",
    "NMED": "NMED",
    "SURG": "SURG",
    "OMED": "OMED",
    "ORTHO": "ORTHO",
    "NSURG": "NSURG",
    "TRAUM": "TRAUM",
    # Merge surgical specialties
    "VSURG": "OTHER_SURG",
    "TSURG": "OTHER_SURG",
    "CSURG": "OTHER_SURG",
    "PSURG": "OTHER_SURG",
    # Merge OB/GYN
    "GYN": "OB_GYN",
    "OBS": "OB_GYN",
    # Merge small specialties
    "GU": "OTHER",
    "PSYCH": "OTHER",
    "ENT": "OTHER",
    "EYE": "OTHER",
    "DENT": "OTHER",
}

# Human-readable department names (for output)
DEPARTMENT_NAMES = {
    "MED": "General Medicine",
    "CMED": "Cardiac Medicine",
    "NMED": "Neuro Medicine",
    "SURG": "General Surgery",
    "OMED": "Oncology Medicine",
    "ORTHO": "Orthopedics",
    "NSURG": "Neurosurgery",
    "TRAUM": "Trauma",
    "OTHER_SURG": "Other Surgery (Vascular/Thoracic/Cardiac/Plastic)",
    "OB_GYN": "Obstetrics / Gynecology",
    "OTHER": "Other (Urology/Psychiatry/ENT/Eye/Dental)",
}


# ---------------------------------------------------------------------------
# Reuse triage preprocessing
# ---------------------------------------------------------------------------
from proiect_licenta.triage_pipeline_v1 import normalize_complaint_text, ABBREVIATIONS


# ---------------------------------------------------------------------------
# 1. Load & Clean Data — Admitted patients with diagnosis + service
# ---------------------------------------------------------------------------
def load_and_clean_data() -> pd.DataFrame:
    """Load triage + edstays + patients + diagnosis + services for admitted patients."""
    print("=" * 60)
    print("STEP 1: Loading data (admitted patients only)")
    print("=" * 60)

    # Load triage
    triage = pd.read_csv(TRIAGE_CSV)
    print(f"  triage.csv: {len(triage):,} rows")

    # Load edstays
    edstays = pd.read_csv(
        EDSTAYS_CSV,
        usecols=["subject_id", "stay_id", "hadm_id", "intime", "gender",
                 "arrival_transport", "disposition"],
    )
    print(f"  edstays.csv: {len(edstays):,} rows")

    # Load patients
    patients = pd.read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "anchor_age", "anchor_year"],
    )
    print(f"  patients.csv: {len(patients):,} rows")

    # Load primary diagnoses (seq_num=1 only)
    diag = pd.read_csv(DIAGNOSIS_CSV)
    diag = diag[diag["seq_num"] == 1][["stay_id", "category", "icd_code", "icd_title"]]
    print(f"  categorized_diagnosis.csv (primary): {len(diag):,} rows")

    # Load services (first service per admission)
    services = pd.read_csv(SERVICES_CSV)
    services["transfertime"] = pd.to_datetime(services["transfertime"])
    services_first = (
        services.sort_values("transfertime")
        .groupby("hadm_id")
        .first()
        .reset_index()[["hadm_id", "curr_service"]]
    )
    print(f"  services.csv (first per admission): {len(services_first):,} rows")

    # ── Filter to admitted patients ──
    admitted = edstays[edstays["disposition"] == "ADMITTED"].copy()
    print(f"\n  Admitted stays: {len(admitted):,}")

    # ── Merge all tables ──
    df = triage.merge(admitted, on=["subject_id", "stay_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="left")
    df = df.merge(diag, on="stay_id", how="inner")
    df = df.merge(services_first, on="hadm_id", how="inner")
    print(f"  After full merge (triage+edstays+patients+diag+services): {len(df):,}")

    # ── Compute age ──
    df["intime"] = pd.to_datetime(df["intime"])
    df["visit_year"] = df["intime"].dt.year
    df["age"] = df["anchor_age"] + (df["visit_year"] - df["anchor_year"])
    df["age"] = df["age"].clip(0, 120).fillna(50).astype(int)

    # ── Clean chief complaints ──
    initial = len(df)
    df = df.dropna(subset=["chiefcomplaint", "category", "curr_service"])
    df = df[df["chiefcomplaint"].str.strip() != ""]
    print(f"  After dropping missing data: {len(df):,} (dropped {initial - len(df):,})")

    # ── Apply category groupings ──
    df["diagnosis_group"] = df["category"].map(DIAGNOSIS_GROUP_MAP).fillna("Other")
    df["service_group"] = df["curr_service"].map(SERVICE_GROUP_MAP).fillna("OTHER")

    # ── Clean pain ──
    df["pain"] = pd.to_numeric(df["pain"], errors="coerce")
    df["pain_missing"] = df["pain"].isna().astype(int)
    df["pain"] = df["pain"].fillna(-1).astype(int)
    df.loc[df["pain"] > 10, "pain"] = -1
    df.loc[df["pain"] < 0, "pain"] = -1

    # ── Encode demographics ──
    df["gender_male"] = (df["gender"] == "M").astype(int)
    df["arrival_ambulance"] = (df["arrival_transport"] == "AMBULANCE").astype(int)
    df["arrival_helicopter"] = (df["arrival_transport"] == "HELICOPTER").astype(int)
    df["arrival_walk_in"] = (df["arrival_transport"] == "WALK IN").astype(int)

    # ── Acuity (for ground truth reference; triage predictions will be generated) ──
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    df = df[df["acuity"].between(1, 5)]
    df["acuity"] = df["acuity"].astype(int)
    df["admitted"] = 1  # all rows are admitted

    # ── Print distributions ──
    print(f"\n  Diagnosis Category Distribution (grouped):")
    for cat in df["diagnosis_group"].value_counts().index:
        count = (df["diagnosis_group"] == cat).sum()
        print(f"    {cat:45s} {count:>7,}  ({100 * count / len(df):5.1f}%)")

    print(f"\n  Department Distribution (grouped):")
    for svc in df["service_group"].value_counts().index:
        count = (df["service_group"] == svc).sum()
        print(f"    {svc:20s} {count:>7,}  ({100 * count / len(df):5.1f}%)")

    print(f"\n  Age: mean={df['age'].mean():.1f}  median={df['age'].median():.0f}")
    print(f"  Gender: M={df['gender_male'].sum():,}  F={len(df) - df['gender_male'].sum():,}")

    return df


# ---------------------------------------------------------------------------
# 2. Build features (reuses triage TF-IDF + severity map + triage predictions)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix using saved triage artifacts for cascading predictions."""
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering (using saved triage artifacts)")
    print("=" * 60)

    df = df.copy()

    # ── Load triage artifacts ──
    tfidf = joblib.load(TRIAGE_MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(TRIAGE_MODELS_DIR / "severity_map.joblib")
    acuity_model = joblib.load(TRIAGE_MODELS_DIR / "acuity_model.joblib")
    disposition_model = joblib.load(TRIAGE_MODELS_DIR / "disposition_model.joblib")
    print("  Loaded triage artifacts: tfidf, severity_map, acuity_model, disposition_model")

    # ── Normalize complaint text ──
    df["complaint_text"] = df["chiefcomplaint"].apply(normalize_complaint_text)
    df["n_complaints"] = df["chiefcomplaint"].apply(
        lambda x: len([c.strip() for c in str(x).split(",") if c.strip()])
    )
    df["complaint_length"] = df["complaint_text"].apply(len)

    # ── TF-IDF (transform only, using saved vectorizer) ──
    tfidf_matrix = tfidf.transform(df["complaint_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        index=df.index,
    )
    print(f"  TF-IDF features: {tfidf_df.shape[1]}")

    # ── Severity priors ──
    def compute_severity_priors(text: str) -> tuple:
        words = text.split()
        severities = [severity_map[w] for w in words if w in severity_map]
        if severities:
            return (min(severities), np.mean(severities),
                    max(severities), np.std(severities) if len(severities) > 1 else 0.0)
        return 3.0, 3.0, 3.0, 0.0

    severity_features = df["complaint_text"].apply(compute_severity_priors)
    df["min_severity_prior"] = severity_features.apply(lambda x: x[0])
    df["mean_severity_prior"] = severity_features.apply(lambda x: x[1])
    df["max_severity_prior"] = severity_features.apply(lambda x: x[2])
    df["std_severity_prior"] = severity_features.apply(lambda x: x[3])

    # ── Age bins ──
    df["age_bin"] = pd.cut(
        df["age"], bins=[0, 18, 35, 50, 65, 80, 120],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float).fillna(2)

    # ── Pain bins ──
    df["pain_low"] = ((df["pain"] >= 0) & (df["pain"] <= 3)).astype(int)
    df["pain_mid"] = ((df["pain"] >= 4) & (df["pain"] <= 6)).astype(int)
    df["pain_high"] = ((df["pain"] >= 7) & (df["pain"] <= 10)).astype(int)

    # ── Interaction features ──
    df["age_ambulance"] = df["age"] * df["arrival_ambulance"]
    df["pain_x_min_severity"] = df["pain"].clip(0, 10) * (5 - df["min_severity_prior"])
    df["age_severity"] = df["age"] * (5 - df["min_severity_prior"])
    df["high_pain_ambulance"] = df["pain_high"] * df["arrival_ambulance"]
    df["elderly"] = (df["age"] >= 65).astype(int)
    df["elderly_ambulance"] = df["elderly"] * df["arrival_ambulance"]

    # ── Assemble triage feature vector (same order as triage pipeline) ──
    structured_cols = [
        "pain", "pain_missing", "pain_low", "pain_mid", "pain_high",
        "n_complaints", "complaint_length",
        "min_severity_prior", "mean_severity_prior",
        "max_severity_prior", "std_severity_prior",
        "age", "age_bin", "gender_male",
        "arrival_ambulance", "arrival_helicopter", "arrival_walk_in",
        "age_ambulance", "pain_x_min_severity", "age_severity",
        "high_pain_ambulance", "elderly", "elderly_ambulance",
    ]

    structured = df[structured_cols].reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)
    triage_features = pd.concat([structured, tfidf_df], axis=1)

    # ── Generate triage predictions (cascading) ──
    print("  Generating triage predictions (acuity + disposition)...")
    predicted_acuity = acuity_model.predict(triage_features) + 1  # 0-4 → 1-5
    triage_features_disp = triage_features.copy()
    triage_features_disp["predicted_acuity"] = predicted_acuity
    predicted_disposition = disposition_model.predict(triage_features_disp)

    # ── Add triage predictions as features for doctor model ──
    triage_features["predicted_acuity"] = predicted_acuity
    triage_features["predicted_disposition"] = predicted_disposition

    print(f"  Total features: {triage_features.shape[1]}")
    print(f"  (structured: {len(structured_cols)} + tfidf: {tfidf_df.shape[1]} "
          f"+ triage_preds: 2)")

    return triage_features


# ---------------------------------------------------------------------------
# 3. Train Diagnosis Model
# ---------------------------------------------------------------------------
def train_diagnosis_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_names: list,
) -> XGBClassifier:
    """Train diagnosis category model with XGBoost."""
    print("\n" + "=" * 60)
    print("STEP 4a: Training DIAGNOSIS CATEGORY model (XGBoost)")
    print("=" * 60)

    n_classes = len(label_names)

    # Soft class weights (sqrt inverse frequency)
    class_counts = y_train.value_counts()
    total = len(y_train)
    sample_weights = y_train.map(
        lambda x: np.sqrt(total / (n_classes * class_counts[x]))
    )

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
        num_class=n_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print(f"  Training XGBoost ({n_classes} classes, up to 3000 trees, lr=0.02)...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=3))

    return model


# ---------------------------------------------------------------------------
# 4. Train Department Model
# ---------------------------------------------------------------------------
def train_department_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_names: list,
) -> XGBClassifier:
    """Train department prediction model with XGBoost."""
    print("\n" + "=" * 60)
    print("STEP 4b: Training DEPARTMENT model (XGBoost)")
    print("=" * 60)

    n_classes = len(label_names)

    # Soft class weights
    class_counts = y_train.value_counts()
    total = len(y_train)
    sample_weights = y_train.map(
        lambda x: np.sqrt(total / (n_classes * class_counts[x]))
    )

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
        num_class=n_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print(f"  Training XGBoost ({n_classes} classes, up to 3000 trees, lr=0.02)...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=3))

    return model


# ---------------------------------------------------------------------------
# 5. Save Artifacts
# ---------------------------------------------------------------------------
def save_models(
    diagnosis_model,
    department_model,
    diagnosis_labels: list,
    department_labels: list,
    diagnosis_accuracy: float,
    department_accuracy: float,
):
    """Save trained doctor models and metadata."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving doctor model artifacts")
    print("=" * 60)

    DOCTOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(diagnosis_model, DOCTOR_MODELS_DIR / "diagnosis_model.joblib")
    joblib.dump(department_model, DOCTOR_MODELS_DIR / "department_model.joblib")

    metadata = {
        "version": 1,
        "trained_at": datetime.now().isoformat(),
        "diagnosis_labels": diagnosis_labels,
        "department_labels": department_labels,
        "department_names": DEPARTMENT_NAMES,
        "diagnosis_accuracy": round(diagnosis_accuracy, 4),
        "department_accuracy": round(department_accuracy, 4),
        "model_type": "XGBClassifier",
        "n_diagnosis_classes": len(diagnosis_labels),
        "n_department_classes": len(department_labels),
        "note": "Uses triage model predictions as cascading features. "
                "Department model also uses predicted diagnosis as input.",
    }

    with open(DOCTOR_MODELS_DIR / "doctor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {DOCTOR_MODELS_DIR}")
    for fname in ["diagnosis_model.joblib", "department_model.joblib",
                   "doctor_metadata.json"]:
        print(f"  - {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "#" * 60)
    print("  Doctor Agent — Model Training Pipeline v1")
    print("  (Diagnosis Category + Department Prediction)")
    print("#" * 60)

    # 1. Load
    df = load_and_clean_data()

    # 2. Sample 100K rows (stratified by diagnosis)
    print("\n" + "=" * 60)
    print("STEP 2b: Sampling 100K rows (stratified by diagnosis)")
    print("=" * 60)
    if len(df) > 100_000:
        df, _ = train_test_split(
            df, train_size=100_000, random_state=42,
            stratify=df["diagnosis_group"],
        )
        print(f"  Sampled: {len(df):,} rows")
    else:
        print(f"  Using all {len(df):,} rows (under 100K)")

    # 3. Build features
    features = build_features(df)

    # 4. Encode labels
    diagnosis_labels = sorted(df["diagnosis_group"].unique())
    department_labels = sorted(df["service_group"].unique())
    diag_label_to_idx = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_label_to_idx = {l: i for i, l in enumerate(department_labels)}

    y_diagnosis = df["diagnosis_group"].map(diag_label_to_idx).reset_index(drop=True)
    y_department = df["service_group"].map(dept_label_to_idx).reset_index(drop=True)

    print(f"\n  Diagnosis classes ({len(diagnosis_labels)}): {diagnosis_labels}")
    print(f"  Department classes ({len(department_labels)}): {department_labels}")

    # 5. Train/test split (80/20, stratified by diagnosis)
    print("\n" + "=" * 60)
    print("STEP 3: Train/test split (80/20, stratified by diagnosis)")
    print("=" * 60)

    X_train, X_test, y_diag_train, y_diag_test, y_dept_train, y_dept_test = \
        train_test_split(
            features, y_diagnosis, y_department,
            test_size=0.2, random_state=42,
            stratify=y_diagnosis,
        )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 6a. Train diagnosis model
    diagnosis_model = train_diagnosis_model(
        X_train, y_diag_train, X_test, y_diag_test, diagnosis_labels,
    )
    diag_accuracy = accuracy_score(y_diag_test, diagnosis_model.predict(X_test))

    # 6b. Train department model (add predicted diagnosis as feature)
    X_train_dept = X_train.copy()
    X_train_dept["predicted_diagnosis"] = diagnosis_model.predict(X_train)
    X_test_dept = X_test.copy()
    X_test_dept["predicted_diagnosis"] = diagnosis_model.predict(X_test)

    department_model = train_department_model(
        X_train_dept, y_dept_train, X_test_dept, y_dept_test, department_labels,
    )
    dept_accuracy = accuracy_score(y_dept_test, department_model.predict(X_test_dept))

    # 7. Save
    save_models(
        diagnosis_model, department_model,
        diagnosis_labels, department_labels,
        diag_accuracy, dept_accuracy,
    )

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE!")
    print("#" * 60)
    print(f"  Diagnosis accuracy:   {diag_accuracy:.4f}")
    print(f"  Department accuracy:  {dept_accuracy:.4f}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
