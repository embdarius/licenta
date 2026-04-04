"""
Data Pipeline for Triage Agent — MIMIC-IV Emergency Department (v2)

Improvements over v1:
  1. TF-IDF word vectorization (instead of multi-hot top-100 complaints)
  2. XGBoost (instead of RandomForest)
  3. n_complaints feature (number of chief complaints)
  4. Complaint severity prior (mean acuity per complaint word)
  5. Better class imbalance handling via XGBoost scale_pos_weight

Trains two supervised models:
  1. Acuity Prediction: chiefcomplaint (TF-IDF) + pain + engineered features → ESI 1-5
  2. Disposition Prediction: same features + predicted acuity → ADMITTED vs DISCHARGED
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
MODELS_DIR = BASE_DIR / "models"

TRIAGE_CSV = DATASET_DIR / "triage.csv"
EDSTAYS_CSV = DATASET_DIR / "edstays.csv"


# ---------------------------------------------------------------------------
# 1. Load & Clean Data
# ---------------------------------------------------------------------------
def load_and_clean_data() -> pd.DataFrame:
    """Load triage.csv + edstays.csv, clean and merge them."""
    print("=" * 60)
    print("STEP 1: Loading data...")
    print("=" * 60)

    triage = pd.read_csv(TRIAGE_CSV)
    print(f"  triage.csv: {len(triage):,} rows")

    edstays = pd.read_csv(EDSTAYS_CSV, usecols=["stay_id", "disposition"])
    print(f"  edstays.csv: {len(edstays):,} rows")

    df = triage.merge(edstays, on="stay_id", how="inner")
    print(f"  Merged: {len(df):,} rows")

    # Drop rows with missing chief complaints or acuity
    initial_count = len(df)
    df = df.dropna(subset=["chiefcomplaint", "acuity"])
    df = df[df["chiefcomplaint"].str.strip() != ""]
    print(f"  After dropping missing complaints/acuity: {len(df):,} rows "
          f"(dropped {initial_count - len(df):,})")

    # Clean acuity: should be 1-5
    df["acuity"] = df["acuity"].astype(int)
    df = df[df["acuity"].between(1, 5)]

    # Clean pain: fill missing with -1 (unknown indicator), cap at 0-10
    df["pain"] = pd.to_numeric(df["pain"], errors="coerce").fillna(-1).astype(int)
    df.loc[df["pain"] > 10, "pain"] = -1
    df.loc[df["pain"] < 0, "pain"] = -1

    # Binarize disposition
    df["admitted"] = (df["disposition"] == "ADMITTED").astype(int)

    print(f"\n  Acuity distribution:")
    for level in sorted(df["acuity"].unique()):
        count = (df["acuity"] == level).sum()
        print(f"    ESI {level}: {count:,} ({100 * count / len(df):.1f}%)")

    print(f"\n  Disposition distribution:")
    admitted = df["admitted"].sum()
    print(f"    ADMITTED:     {admitted:,} ({100 * admitted / len(df):.1f}%)")
    print(f"    NOT ADMITTED: {len(df) - admitted:,} "
          f"({100 * (len(df) - admitted) / len(df):.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 2. Normalize complaint text for TF-IDF
# ---------------------------------------------------------------------------
def normalize_complaint_text(text: str) -> str:
    """
    Normalize chief complaint text for TF-IDF.
    Converts comma-separated complaint list into a space-separated string of words.
    Handles abbreviations and common patterns.
    """
    if pd.isna(text) or not str(text).strip():
        return ""
    
    text = str(text).lower().strip()
    
    # Replace common separators with spaces
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    
    # Expand common ED abbreviations
    abbreviations = {
        "abd": "abdominal",
        "n/v": "nausea vomiting",
        "n v": "nausea vomiting",
        "s/p": "status post",
        "s p": "status post",
        "sob": "shortness of breath",
        "cp": "chest pain",
        "ha": "headache",
        "ams": "altered mental status",
        "loc": "loss of consciousness",
        "etoh": "alcohol",
        "uti": "urinary tract infection",
        "uri": "upper respiratory infection",
        "mv": "motor vehicle",
        "mva": "motor vehicle accident",
        "mvc": "motor vehicle collision",
        "htn": "hypertension",
        "dm": "diabetes",
        "chf": "congestive heart failure",
        "gi": "gastrointestinal",
        "r/o": "rule out",
        "w/": "with",
        "w/o": "without",
        "fx": "fracture",
        "lac": "laceration",
        "inj": "injury",
        "sx": "symptoms",
        "dx": "diagnosis",
        "tx": "treatment",
        "hx": "history",
        "bld": "blood",
        "diff": "difficulty",
    }
    
    words = text.split()
    expanded = []
    for word in words:
        word = word.strip()
        if word in abbreviations:
            expanded.append(abbreviations[word])
        elif len(word) > 1:  # Skip single characters
            expanded.append(word)
    
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    tfidf: TfidfVectorizer = None,
    severity_map: dict = None,
    fit: bool = True,
) -> tuple:
    """
    Build feature matrix with:
      - TF-IDF of complaint text (word-level)
      - pain score
      - n_complaints (number of chief complaints)
      - max_severity_prior (severity of the most concerning complaint word)
      - mean_severity_prior (average severity across complaint words)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering")
    print("=" * 60)

    # Normalize complaint text
    df = df.copy()
    df["complaint_text"] = df["chiefcomplaint"].apply(normalize_complaint_text)

    # Count number of original complaints (before normalization)
    df["n_complaints"] = df["chiefcomplaint"].apply(
        lambda x: len([c.strip() for c in str(x).split(",") if c.strip()])
    )

    # --- TF-IDF ---
    if fit:
        tfidf = TfidfVectorizer(
            max_features=500,        # Top 500 word features
            min_df=50,               # Word must appear in at least 50 documents
            max_df=0.95,             # Ignore words in >95% of documents
            ngram_range=(1, 2),      # Unigrams + bigrams (captures "chest pain" as single feature)
            sublinear_tf=True,       # Apply log normalization to TF
            strip_accents="unicode",
        )
        tfidf_matrix = tfidf.fit_transform(df["complaint_text"])
        print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
    else:
        tfidf_matrix = tfidf.transform(df["complaint_text"])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
        index=df.index,
    )

    # --- Severity Prior ---
    if fit:
        # Compute mean acuity per word across training data
        severity_map = defaultdict(list)
        for _, row in df[["complaint_text", "acuity"]].iterrows():
            for word in row["complaint_text"].split():
                word = word.strip()
                if len(word) > 1:
                    severity_map[word].append(row["acuity"])
        # Convert to mean
        severity_map = {w: np.mean(vals) for w, vals in severity_map.items() if len(vals) >= 50}
        print(f"  Severity prior vocabulary: {len(severity_map)} words")
        print(f"  Most severe words (lowest mean acuity):")
        sorted_severity = sorted(severity_map.items(), key=lambda x: x[1])[:10]
        for word, score in sorted_severity:
            print(f"    {word}: {score:.2f}")

    def compute_severity_priors(text: str) -> tuple:
        words = text.split()
        severities = [severity_map[w] for w in words if w in severity_map]
        if severities:
            return min(severities), np.mean(severities)
        return 3.0, 3.0  # Default to median acuity

    severity_features = df["complaint_text"].apply(compute_severity_priors)
    df["max_severity_prior"] = severity_features.apply(lambda x: x[0])
    df["mean_severity_prior"] = severity_features.apply(lambda x: x[1])

    # --- Assemble feature matrix ---
    engineered = df[["pain", "n_complaints", "max_severity_prior", "mean_severity_prior"]].reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)

    features = pd.concat([engineered, tfidf_df], axis=1)

    print(f"\n  Final feature matrix: {features.shape}")
    print(f"    - pain: 1 column")
    print(f"    - n_complaints: 1 column")
    print(f"    - severity priors: 2 columns")
    print(f"    - TF-IDF: {tfidf_df.shape[1]} columns")

    return features, tfidf, severity_map


# ---------------------------------------------------------------------------
# 4. Train Models
# ---------------------------------------------------------------------------
def train_acuity_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier:
    """Train and evaluate the acuity prediction model with XGBoost."""
    print("\n" + "=" * 60)
    print("STEP 4a: Training ACUITY model (XGBoost, ESI 1-5)")
    print("=" * 60)

    # Compute sample weights for class imbalance
    class_counts = y_train.value_counts()
    total = len(y_train)
    n_classes = len(class_counts)
    sample_weights = y_train.map(
        lambda x: total / (n_classes * class_counts[x])
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("  Training XGBoost (500 trees, max_depth=8, lr=0.1)...")
    # XGBoost needs classes 0-4
    y_train_shifted = y_train - 1
    y_test_shifted = y_test - 1

    model.fit(
        X_train, y_train_shifted,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test_shifted)],
        verbose=False,
    )

    y_pred_shifted = model.predict(X_test)
    y_pred = y_pred_shifted + 1  # Back to 1-5

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, digits=3)
    print(report)

    return model


def train_disposition_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier:
    """Train and evaluate the admission/discharge prediction model."""
    print("\n" + "=" * 60)
    print("STEP 4b: Training DISPOSITION model (XGBoost, Admit vs Discharge)")
    print("=" * 60)

    # Scale positive weight for imbalanced binary classification
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("  Training XGBoost (500 trees, max_depth=8, lr=0.1)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=["NOT ADMITTED", "ADMITTED"],
        digits=3,
    )
    print(report)

    return model


# ---------------------------------------------------------------------------
# 5. Save Artifacts
# ---------------------------------------------------------------------------
def save_models(
    acuity_model,
    disposition_model,
    tfidf: TfidfVectorizer,
    severity_map: dict,
    acuity_accuracy: float,
    disposition_accuracy: float,
):
    """Save trained models and metadata."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving model artifacts")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(acuity_model, MODELS_DIR / "acuity_model.joblib")
    joblib.dump(disposition_model, MODELS_DIR / "disposition_model.joblib")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(dict(severity_map), MODELS_DIR / "severity_map.joblib")

    # Build vocabulary list for fuzzy matching (from TF-IDF features)
    vocab = list(tfidf.vocabulary_.keys())

    metadata = {
        "version": 2,
        "trained_at": datetime.now().isoformat(),
        "n_tfidf_features": len(tfidf.vocabulary_),
        "n_severity_words": len(severity_map),
        "acuity_classes": [1, 2, 3, 4, 5],
        "disposition_classes": ["NOT ADMITTED", "ADMITTED"],
        "acuity_accuracy": round(acuity_accuracy, 4),
        "disposition_accuracy": round(disposition_accuracy, 4),
        "model_type": "XGBClassifier",
        "model_params": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.1,
        },
        "features": [
            "pain", "n_complaints",
            "max_severity_prior", "mean_severity_prior",
            f"tfidf (500 word/bigram features)",
        ],
        "note": "Acuity model outputs classes 0-4 (add 1 to get ESI 1-5)",
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Remove old v1 artifacts if they exist
    old_files = ["complaint_encoder.joblib"]
    for f in old_files:
        p = MODELS_DIR / f
        if p.exists():
            p.unlink()
            print(f"  Removed old artifact: {f}")

    print(f"  Saved to: {MODELS_DIR}")
    print(f"  - acuity_model.joblib")
    print(f"  - disposition_model.joblib")
    print(f"  - tfidf_vectorizer.joblib")
    print(f"  - severity_map.joblib")
    print(f"  - model_metadata.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "#" * 60)
    print("  MIMIC-IV Triage Model Training Pipeline v2")
    print("  (TF-IDF + XGBoost + Severity Priors)")
    print("#" * 60)

    # 1. Load data
    df = load_and_clean_data()

    # 2. Train/test split FIRST (prevent data leakage in severity prior)
    print("\n" + "=" * 60)
    print("STEP 3: Train/test split (80/20, stratified by acuity)")
    print("=" * 60)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["acuity"],
    )
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # 3. Build features (fit on train, transform test)
    X_train, tfidf, severity_map = build_features(train_df, fit=True)
    y_acuity_train = train_df["acuity"].reset_index(drop=True)
    y_admit_train = train_df["admitted"].reset_index(drop=True)

    # Silence the step header for test set
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        X_test, _, _ = build_features(test_df, tfidf=tfidf, severity_map=severity_map, fit=False)
    y_acuity_test = test_df["acuity"].reset_index(drop=True)
    y_admit_test = test_df["admitted"].reset_index(drop=True)

    # 4a. Train acuity model
    acuity_model = train_acuity_model(X_train, y_acuity_train, X_test, y_acuity_test)
    acuity_preds = acuity_model.predict(X_test) + 1  # shift back to 1-5
    acuity_accuracy = accuracy_score(y_acuity_test, acuity_preds)

    # 4b. Train disposition model (with predicted acuity as feature)
    X_train_disp = X_train.copy()
    X_train_disp["predicted_acuity"] = acuity_model.predict(X_train) + 1
    X_test_disp = X_test.copy()
    X_test_disp["predicted_acuity"] = acuity_model.predict(X_test) + 1

    disposition_model = train_disposition_model(
        X_train_disp, y_admit_train, X_test_disp, y_admit_test,
    )
    disp_accuracy = accuracy_score(y_admit_test, disposition_model.predict(X_test_disp))

    # 5. Save
    save_models(
        acuity_model, disposition_model,
        tfidf, severity_map,
        acuity_accuracy, disp_accuracy,
    )

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE!")
    print("#" * 60)
    print(f"  Acuity model accuracy:     {acuity_accuracy:.4f}")
    print(f"  Disposition model accuracy: {disp_accuracy:.4f}")
    print(f"  Models saved to: {MODELS_DIR}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
