"""
Data Pipeline for Doctor Agent v2 — Enhanced with Nurse Data

Trains two supervised models for admitted patients (same as doctor_data_pipeline)
but with additional features from vital signs (triage.csv) and medication
history (medrecon.csv).

Comparison:
  - Doctor v1: complaints + demographics + triage predictions (2025 features)
  - Doctor v2: v1 + vitals + vital flags + medication features (~2055 features)
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
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
MEDRECON_CSV = DATASET_DIR / "medrecon.csv"

# ---------------------------------------------------------------------------
# Reuse from doctor pipeline
# ---------------------------------------------------------------------------
from proiect_licenta.doctor_data_pipeline import (
    DIAGNOSIS_GROUP_MAP, SERVICE_GROUP_MAP, DEPARTMENT_NAMES,
)
from proiect_licenta.triage_pipeline_v1 import normalize_complaint_text

# ---------------------------------------------------------------------------
# Medication category grouping
# ---------------------------------------------------------------------------
# Both the drug-name map and the class-keyword map live in the shared
# `tools.med_vocab` module so training (this file) and inference
# (`tools.doctor_tool_v2`) use exactly the same vocabulary. See the module
# docstring for the audit that motivated centralization.
from proiect_licenta.tools.med_vocab import (
    DRUG_NAME_MAP, MED_CLASS_KEYWORDS, MED_CATEGORIES,
    flags_from_row,
)

# Back-compat alias: the old name was MED_CATEGORY_KEYWORDS and held only the
# class-description side. Kept for the metadata dump below.
MED_CATEGORY_KEYWORDS = MED_CLASS_KEYWORDS


def classify_medication(name: str, etcdescription: str) -> set:
    """Training-time classifier: union of drug-name-map and class-keyword hits.

    Replaces the older etcdescription-only version so the training side
    flags the same rows that the inference side (which only sees the drug
    name) would flag.
    """
    return flags_from_row(name, etcdescription)


# ---------------------------------------------------------------------------
# 1. Load & Clean Data — same as doctor pipeline + vitals + meds
# ---------------------------------------------------------------------------
def load_and_clean_data() -> pd.DataFrame:
    """Load admitted patient data with diagnosis, service, vitals, and medications."""
    print("=" * 60)
    print("STEP 1: Loading data (admitted patients + vitals + meds)")
    print("=" * 60)

    # Load triage (includes vitals)
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

    # Load primary diagnoses
    diag = pd.read_csv(DIAGNOSIS_CSV)
    diag = diag[diag["seq_num"] == 1][["stay_id", "category", "icd_code", "icd_title"]]
    print(f"  categorized_diagnosis.csv (primary): {len(diag):,} rows")

    # Load services (first per admission)
    services = pd.read_csv(SERVICES_CSV)
    services["transfertime"] = pd.to_datetime(services["transfertime"])
    services_first = (
        services.sort_values("transfertime")
        .groupby("hadm_id").first().reset_index()[["hadm_id", "curr_service"]]
    )
    print(f"  services.csv (first per admission): {len(services_first):,} rows")

    # Load medications and aggregate per stay
    print("  Loading medrecon.csv and aggregating per stay...")
    med = pd.read_csv(MEDRECON_CSV, usecols=["stay_id", "name", "etcdescription"])
    med_features = _aggregate_medications(med)
    print(f"  medrecon.csv: {len(med):,} rows -> {len(med_features):,} stay-level records")

    # ── Filter to admitted patients ──
    admitted = edstays[edstays["disposition"] == "ADMITTED"].copy()
    print(f"\n  Admitted stays: {len(admitted):,}")

    # ── Merge all tables ──
    df = triage.merge(admitted, on=["subject_id", "stay_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="left")
    df = df.merge(diag, on="stay_id", how="inner")
    df = df.merge(services_first, on="hadm_id", how="inner")
    df = df.merge(med_features, on="stay_id", how="left")  # left join: not all have meds
    print(f"  After full merge: {len(df):,}")

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
    df["pain_triage"] = pd.to_numeric(df["pain"], errors="coerce")
    df["pain_missing"] = df["pain_triage"].isna().astype(int)
    df["pain"] = df["pain_triage"].fillna(-1).astype(int)
    df.loc[df["pain"] > 10, "pain"] = -1
    df.loc[df["pain"] < 0, "pain"] = -1

    # ── Encode demographics ──
    df["gender_male"] = (df["gender"] == "M").astype(int)
    df["arrival_ambulance"] = (df["arrival_transport"] == "AMBULANCE").astype(int)
    df["arrival_helicopter"] = (df["arrival_transport"] == "HELICOPTER").astype(int)
    df["arrival_walk_in"] = (df["arrival_transport"] == "WALK IN").astype(int)

    # ── Acuity ──
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    df = df[df["acuity"].between(1, 5)]
    df["acuity"] = df["acuity"].astype(int)
    df["admitted"] = 1

    # ── Clean vital signs ──
    _clean_vitals(df)

    # ── Fill missing medication flags (patients with no medrecon data) ──
    med_flag_cols = ["n_medications", "meds_unknown"] + list(MED_CATEGORY_KEYWORDS.keys())
    for col in med_flag_cols:
        if col not in df.columns:
            df[col] = 0
    df["n_medications"] = df["n_medications"].fillna(0).astype(int)
    df["meds_unknown"] = df["meds_unknown"].fillna(1).astype(int)  # no medrecon data = unknown
    for flag in MED_CATEGORY_KEYWORDS:
        df[flag] = df[flag].fillna(0).astype(int)

    # ── Print distributions ──
    print(f"\n  Diagnosis Category Distribution (grouped):")
    for cat in df["diagnosis_group"].value_counts().index:
        count = (df["diagnosis_group"] == cat).sum()
        print(f"    {cat:45s} {count:>7,}  ({100 * count / len(df):5.1f}%)")

    print(f"\n  Vital signs availability:")
    for v in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]:
        present = (df[f"{v}_missing"] == 0).sum()
        print(f"    {v:15s}  present: {present:,}/{len(df):,} ({100*present/len(df):.1f}%)")

    has_meds = (df["meds_unknown"] == 0).sum()
    print(f"\n  Medication data: {has_meds:,}/{len(df):,} ({100*has_meds/len(df):.1f}%)")
    print(f"  Mean medications per patient (where known): "
          f"{df.loc[df['meds_unknown']==0, 'n_medications'].mean():.1f}")

    return df


def _aggregate_medications(med: pd.DataFrame) -> pd.DataFrame:
    """Aggregate medication data per stay_id into feature columns.

    For each row we take the union of flags from the drug `name` (inference-
    side vocabulary) and the `etcdescription` class (training-side vocabulary),
    then OR the flags across all rows for a given stay.
    """
    # Count medications per stay
    med_count = med.groupby("stay_id").size().rename("n_medications")

    # Per-row flag set. Using shared med_vocab.flags_from_row keeps training
    # aligned with inference vocabulary.
    med = med.copy()
    med["_flags"] = [
        flags_from_row(n, d)
        for n, d in zip(med["name"].values, med["etcdescription"].values)
    ]

    # Per-stay union of flags, expanded to one column per category.
    def _union(series):
        out = set()
        for s in series:
            out |= s
        return out

    per_stay = med.groupby("stay_id")["_flags"].apply(_union)
    flag_records = [
        {"stay_id": sid, **{cat: int(cat in flags) for cat in MED_CATEGORIES}}
        for sid, flags in per_stay.items()
    ]

    flags_df = pd.DataFrame(flag_records)
    result = flags_df.merge(med_count.reset_index(), on="stay_id", how="left")
    result["meds_unknown"] = 0  # these stays have medication data
    return result


def _clean_vitals(df: pd.DataFrame):
    """Clean vital sign columns in-place, add missing flags and clinical flags."""
    vital_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

    for col in vital_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Clip to physiologically plausible ranges, then fill missing with median
    df["temperature"] = df["temperature"].clip(90, 110)
    df["heartrate"] = df["heartrate"].clip(20, 250)
    df["resprate"] = df["resprate"].clip(4, 60)
    df["o2sat"] = df["o2sat"].clip(50, 100)
    df["sbp"] = df["sbp"].clip(50, 300)
    df["dbp"] = df["dbp"].clip(20, 200)

    # Fill missing with population medians
    medians = {"temperature": 98.1, "heartrate": 84, "resprate": 18,
               "o2sat": 98, "sbp": 134, "dbp": 78}
    for col, med_val in medians.items():
        df[col] = df[col].fillna(med_val)

    # Clinical flags
    df["fever"] = (df["temperature"] > 100.4).astype(int)
    df["tachycardia"] = (df["heartrate"] > 100).astype(int)
    df["bradycardia"] = (df["heartrate"] < 60).astype(int)
    df["tachypnea"] = (df["resprate"] > 20).astype(int)
    df["hypoxia"] = (df["o2sat"] < 94).astype(int)
    df["hypertension"] = (df["sbp"] > 140).astype(int)
    df["hypotension"] = (df["sbp"] < 90).astype(int)

    # MAP (mean arterial pressure)
    df["map"] = (df["sbp"] + 2 * df["dbp"]) / 3


# ---------------------------------------------------------------------------
# 2. Build features
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix: triage features + triage predictions + vitals + meds."""
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering (triage artifacts + vitals + meds)")
    print("=" * 60)

    df = df.copy()

    # ── Load triage artifacts ──
    tfidf = joblib.load(TRIAGE_MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(TRIAGE_MODELS_DIR / "severity_map.joblib")
    acuity_model = joblib.load(TRIAGE_MODELS_DIR / "acuity_model.joblib")
    disposition_model = joblib.load(TRIAGE_MODELS_DIR / "disposition_model.joblib")
    print("  Loaded triage artifacts")

    # ── Text features ──
    df["complaint_text"] = df["chiefcomplaint"].apply(normalize_complaint_text)
    df["n_complaints"] = df["chiefcomplaint"].apply(
        lambda x: len([c.strip() for c in str(x).split(",") if c.strip()])
    )
    df["complaint_length"] = df["complaint_text"].apply(len)

    # ── TF-IDF ──
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

    sev = df["complaint_text"].apply(compute_severity_priors)
    df["min_severity_prior"] = sev.apply(lambda x: x[0])
    df["mean_severity_prior"] = sev.apply(lambda x: x[1])
    df["max_severity_prior"] = sev.apply(lambda x: x[2])
    df["std_severity_prior"] = sev.apply(lambda x: x[3])

    # ── Age/pain bins ──
    df["age_bin"] = pd.cut(
        df["age"], bins=[0, 18, 35, 50, 65, 80, 120], labels=[0, 1, 2, 3, 4, 5],
    ).astype(float).fillna(2)
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
    print("  Generating triage predictions...")
    predicted_acuity = acuity_model.predict(triage_features) + 1
    triage_features_disp = triage_features.copy()
    triage_features_disp["predicted_acuity"] = predicted_acuity
    predicted_disposition = disposition_model.predict(triage_features_disp)

    triage_features["predicted_acuity"] = predicted_acuity
    triage_features["predicted_disposition"] = predicted_disposition

    # ── Vital sign features ──
    vital_cols = [
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
        "temperature_missing", "heartrate_missing", "resprate_missing",
        "o2sat_missing", "sbp_missing", "dbp_missing",
        "fever", "tachycardia", "bradycardia", "tachypnea",
        "hypoxia", "hypertension", "hypotension", "map",
    ]
    vitals = df[vital_cols].reset_index(drop=True)
    print(f"  Vital sign features: {len(vital_cols)}")

    # ── Medication features ──
    med_cols = ["n_medications", "meds_unknown"] + list(MED_CATEGORY_KEYWORDS.keys())
    meds = df[med_cols].reset_index(drop=True)
    print(f"  Medication features: {len(med_cols)}")

    # ── Assemble all ──
    features = pd.concat([triage_features, vitals, meds], axis=1)
    print(f"  Total features: {features.shape[1]}")

    return features


# ---------------------------------------------------------------------------
# 3-4. Train models (same as doctor_data_pipeline)
# ---------------------------------------------------------------------------
def train_model(
    X_train, y_train, X_test, y_test, label_names, model_name,
) -> XGBClassifier:
    """Train an XGBoost multiclass model."""
    print(f"\n{'='*60}")
    print(f"  Training {model_name} ({len(label_names)} classes, XGBoost)")
    print(f"{'='*60}")

    n_classes = len(label_names)
    class_counts = y_train.value_counts()
    total = len(y_train)
    sample_weights = y_train.map(
        lambda x: np.sqrt(total / (n_classes * class_counts[x]))
    )

    model = XGBClassifier(
        n_estimators=3000, max_depth=10, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.5, colsample_bylevel=0.7,
        min_child_weight=3, gamma=0.05, reg_alpha=0.5, reg_lambda=2.0,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", early_stopping_rounds=100,
        random_state=42, n_jobs=-1, verbosity=0,
    )

    print(f"  Training (up to 3000 trees, lr=0.02, early stopping=100)...")
    model.fit(
        X_train, y_train, sample_weight=sample_weights,
        eval_set=[(X_test, y_test)], verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=3))

    return model


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
def save_models(
    diagnosis_model, department_model,
    diagnosis_labels, department_labels,
    diagnosis_accuracy, department_accuracy,
    vital_medians,
):
    """Save v2 doctor models and metadata."""
    print(f"\n{'='*60}")
    print("  Saving Doctor v2 model artifacts")
    print(f"{'='*60}")

    DOCTOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(diagnosis_model, DOCTOR_MODELS_DIR / "diagnosis_model_v2.joblib")
    joblib.dump(department_model, DOCTOR_MODELS_DIR / "department_model_v2.joblib")

    metadata = {
        "version": 2,
        "trained_at": datetime.now().isoformat(),
        "diagnosis_labels": diagnosis_labels,
        "department_labels": department_labels,
        "department_names": DEPARTMENT_NAMES,
        "diagnosis_accuracy": round(diagnosis_accuracy, 4),
        "department_accuracy": round(department_accuracy, 4),
        "model_type": "XGBClassifier",
        "n_diagnosis_classes": len(diagnosis_labels),
        "n_department_classes": len(department_labels),
        "vital_medians": vital_medians,
        "med_category_keywords": {k: v for k, v in MED_CATEGORY_KEYWORDS.items()},
        "note": "Doctor v2: includes vital signs + medication features from Nurse Agent. "
                "Uses triage model predictions as cascading features.",
    }

    with open(DOCTOR_MODELS_DIR / "doctor_v2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {DOCTOR_MODELS_DIR}")
    for fname in ["diagnosis_model_v2.joblib", "department_model_v2.joblib",
                   "doctor_v2_metadata.json"]:
        print(f"  - {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "#" * 60)
    print("  Doctor Agent v2 — Model Training Pipeline")
    print("  (With Vital Signs + Medication Features)")
    print("#" * 60)

    # 1. Load
    df = load_and_clean_data()

    # 2. Sample 100K
    print(f"\n{'='*60}")
    print("  Sampling 100K rows (stratified by diagnosis)")
    print(f"{'='*60}")
    if len(df) > 100_000:
        df, _ = train_test_split(
            df, train_size=100_000, random_state=42,
            stratify=df["diagnosis_group"],
        )
        print(f"  Sampled: {len(df):,} rows")

    # 3. Build features
    features = build_features(df)

    # 4. Encode labels
    diagnosis_labels = sorted(df["diagnosis_group"].unique())
    department_labels = sorted(df["service_group"].unique())
    diag_map = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_map = {l: i for i, l in enumerate(department_labels)}
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)
    y_dept = df["service_group"].map(dept_map).reset_index(drop=True)

    # 5. Split
    print(f"\n{'='*60}")
    print("  Train/test split (80/20, stratified)")
    print(f"{'='*60}")
    X_train, X_test, y_diag_train, y_diag_test, y_dept_train, y_dept_test = \
        train_test_split(features, y_diag, y_dept, test_size=0.2,
                         random_state=42, stratify=y_diag)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 6a. Train diagnosis model
    diag_model = train_model(
        X_train, y_diag_train, X_test, y_diag_test,
        diagnosis_labels, "DIAGNOSIS v2",
    )
    diag_acc = accuracy_score(y_diag_test, diag_model.predict(X_test))

    # 6b. Train department model (cascading)
    X_train_dept = X_train.copy()
    X_train_dept["predicted_diagnosis"] = diag_model.predict(X_train)
    X_test_dept = X_test.copy()
    X_test_dept["predicted_diagnosis"] = diag_model.predict(X_test)

    dept_model = train_model(
        X_train_dept, y_dept_train, X_test_dept, y_dept_test,
        department_labels, "DEPARTMENT v2",
    )
    dept_acc = accuracy_score(y_dept_test, dept_model.predict(X_test_dept))

    # 7. Save
    vital_medians = {"temperature": 98.1, "heartrate": 84, "resprate": 18,
                     "o2sat": 98, "sbp": 134, "dbp": 78}
    save_models(
        diag_model, dept_model, diagnosis_labels, department_labels,
        diag_acc, dept_acc, vital_medians,
    )

    print(f"\n{'#'*60}")
    print("  TRAINING COMPLETE!")
    print(f"{'#'*60}")
    print(f"  Diagnosis v2 accuracy: {diag_acc:.4f}")
    print(f"  Department v2 accuracy: {dept_acc:.4f}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
