import os
import json
import warnings
import gc
from pathlib import Path
from datetime import datetime
import io
import contextlib

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

from proiect_licenta.data_pipeline import load_and_clean_data, build_features

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "datasets_mimic-iv" / "mimic-iv-ed"
HOSP_DIR = BASE_DIR / "datasets" / "datasets_mimic-iv" / "mimic-iv" / "hosp"
MODELS_DIR = BASE_DIR / "models"

DIAGNOSIS_CSV = DATASET_DIR / "files_created" / "categorized_diagnosis.csv"
SERVICES_CSV = HOSP_DIR / "services.csv"
EDSTAYS_CSV = DATASET_DIR / "edstays.csv"


def load_doctor_data():
    """Load base data + diagnosis + services."""
    print("=" * 60)
    print("STEP 1: Loading data for Doctor Agent...")
    print("=" * 60)
    
    df = load_and_clean_data()
    
    # 1. Merge Diagnosis
    print("  Loading categorized_diagnosis.csv...")
    diag = pd.read_csv(DIAGNOSIS_CSV)
    diag_primary = diag[diag['seq_num'] == 1][['stay_id', 'category']].drop_duplicates()
    df = df.merge(diag_primary, on='stay_id', how='inner')
    print(f"  Merged primary diagnosis, rows: {len(df)}")
    
    # 2. Merge edstays to get hadm_id
    print("  Loading hadm_id from edstays.csv...")
    edstays = pd.read_csv(EDSTAYS_CSV, usecols=['stay_id', 'hadm_id'])
    df = df.merge(edstays, on='stay_id', how='inner')
    
    # 3. Merge Services
    print("  Loading services.csv...")
    services = pd.read_csv(SERVICES_CSV, usecols=['hadm_id', 'curr_service'])
    services_first = services.groupby('hadm_id').first().reset_index()
    df = df.merge(services_first, on='hadm_id', how='left')
    
    # Free memory
    del diag, diag_primary, edstays, services, services_first
    gc.collect()
    
    return df


def generate_extended_features(df, tfidf, severity_map, acuity_model, disp_model):
    """Memory-efficient feature building."""
    with contextlib.redirect_stdout(io.StringIO()):
        X_raw, _, _ = build_features(df, tfidf=tfidf, severity_map=severity_map, fit=False)
    
    # Predict triage models
    pred_acuity = acuity_model.predict(X_raw) + 1
    
    X_disp = X_raw.copy()
    X_disp['predicted_acuity'] = pred_acuity
    pred_admit = disp_model.predict(X_disp)
    del X_disp
    
    X_raw['predicted_acuity'] = pred_acuity
    X_raw['predicted_admit'] = pred_admit
    
    return X_raw


def main():
    print("\n" + "#" * 60)
    print("  MIMIC-IV Doctor Model Training Pipeline (Phase 2)")
    print("#" * 60)
    
    # Load and prep data
    df = load_doctor_data()
    # ---------------------------------------------------------
    # BENCHMARK SETTING: Limit to 100k rows for faster training
    # ---------------------------------------------------------
    print("\n  [BENCHMARK] Downsampling dataset to 100,000 rows...")
    df = df.sample(n=min(100000, len(df)), random_state=42).reset_index(drop=True)
    
    valid_categories = df['category'].value_counts()[lambda x: x >= 50].index
    df = df[df['category'].isin(valid_categories)].reset_index(drop=True)
    
    print("\n  Splitting data FIRST to avoid Out-Of-Memory errors...")
    
    # Prepare Diagnosis Labels up front
    y_diag = df['category']
    le_diag = LabelEncoder()
    df['diag_encoded'] = le_diag.fit_transform(y_diag)
    
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['diag_encoded']
    )
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Load models
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
    acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
    disp_model = joblib.load(MODELS_DIR / "disposition_model.joblib")
    
    print("\n" + "=" * 60)
    print("STEP 2: Generating Features & Triage Predictions")
    print("=" * 60)
    
    # Generate chunks
    print("  Processing Train features...")
    X_train = generate_extended_features(df_train, tfidf, severity_map, acuity_model, disp_model)
    y_train = df_train['diag_encoded'].values
    
    print("  Processing Test features...")
    X_test = generate_extended_features(df_test, tfidf, severity_map, acuity_model, disp_model)
    y_test = df_test['diag_encoded'].values
    
    # Delete original df's to save RAM
    del df
    gc.collect()
    
    # -----------------------------------------------------------------------
    # MODEL 1: DIAGNOSIS CATEGORY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Training DIAGNOSIS CATEGORY model")
    print("=" * 60)
    
    diag_model = XGBClassifier(
        n_estimators=1000,       
        max_depth=8,            
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        objective="multi:softprob",
        num_class=len(le_diag.classes_),
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    print(f"  Training Diagnosis XGBoost on {len(X_train)} samples ({len(le_diag.classes_)} classes)...")
    diag_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    diag_preds = diag_model.predict(X_test)
    diag_acc = accuracy_score(y_test, diag_preds)
    print(f"  Best iteration: {diag_model.best_iteration}")
    print(f"  Diagnosis Accuracy: {diag_acc:.4f}")
    
    pd_report_diag = pd.DataFrame(classification_report(y_test, diag_preds, target_names=le_diag.classes_, output_dict=True)).T
    top_support = pd_report_diag.sort_values(by='support', ascending=False).head(5)
    print("\n  Top 5 Categories Performance:")
    print(top_support[['precision', 'recall', 'f1-score', 'support']])
    
    # -----------------------------------------------------------------------
    # MODEL 2: HOSPITAL SERVICE DEPARTMENT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Training SERVICE DEPARTMENT model")
    print("=" * 60)
    
    # We must predict the diagnosis on the train/test sets to use as feature
    X_train['predicted_diagnosis_enc'] = diag_model.predict(X_train)
    X_test['predicted_diagnosis_enc'] = diag_model.predict(X_test)
    
    # Filter for admitted patients with a valid service
    train_admit_mask = (df_train['admitted'] == 1) & (df_train['curr_service'].notnull())
    test_admit_mask = (df_test['admitted'] == 1) & (df_test['curr_service'].notnull())
    
    X_srv_train = X_train.loc[train_admit_mask].copy()
    y_srv_train_raw = df_train.loc[train_admit_mask, 'curr_service']
    
    X_srv_test = X_test.loc[test_admit_mask].copy()
    y_srv_test_raw = df_test.loc[test_admit_mask, 'curr_service']
    
    # Valid services must show up at least 50 times in training
    valid_services = y_srv_train_raw.value_counts()[lambda x: x >= 50].index
    
    train_valid = y_srv_train_raw.isin(valid_services)
    test_valid = y_srv_test_raw.isin(valid_services)
    
    X_srv_train = X_srv_train[train_valid]
    y_srv_train = y_srv_train_raw[train_valid]
    
    X_srv_test = X_srv_test[test_valid]
    y_srv_test = y_srv_test_raw[test_valid]
    
    le_srv = LabelEncoder()
    y_srv_train_enc = le_srv.fit_transform(y_srv_train)
    y_srv_test_enc = le_srv.transform(y_srv_test) # MUST transform on test set
    
    del X_train, X_test, df_train, df_test
    gc.collect()
    
    print(f"  Filtering to {len(X_srv_train)} train admitted samples across {len(le_srv.classes_)} services.")
    
    srv_model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        objective="multi:softprob",
        num_class=len(le_srv.classes_),
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    print("  Training Service XGBoost...")
    srv_model.fit(
        X_srv_train, y_srv_train_enc,
        eval_set=[(X_srv_test, y_srv_test_enc)],
        verbose=False
    )
    
    srv_preds = srv_model.predict(X_srv_test)
    srv_acc = accuracy_score(y_srv_test_enc, srv_preds)
    print(f"  Best iteration: {srv_model.best_iteration}")
    print(f"  Service Accuracy: {srv_acc:.4f}")
    
    pd_report_srv = pd.DataFrame(classification_report(y_srv_test_enc, srv_preds, target_names=le_srv.classes_, output_dict=True)).T
    top_srv_support = pd_report_srv.sort_values(by='support', ascending=False).head(5)
    print("\n  Top 5 Services Performance:")
    print(top_srv_support[['precision', 'recall', 'f1-score', 'support']])
    
    # -----------------------------------------------------------------------
    # SAVE MODELS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Saving Model Artifacts")
    print("=" * 60)
    
    joblib.dump(diag_model, MODELS_DIR / "diagnosis_model.joblib")
    joblib.dump(le_diag, MODELS_DIR / "diagnosis_le.joblib")
    joblib.dump(srv_model, MODELS_DIR / "service_model.joblib")
    joblib.dump(le_srv, MODELS_DIR / "service_le.joblib")
    
    print(f"  Saved artifacts to {MODELS_DIR}")
    print("  - diagnosis_model.joblib")
    print("  - diagnosis_le.joblib")
    print("  - service_model.joblib")
    print("  - service_le.joblib")

if __name__ == "__main__":
    main()
