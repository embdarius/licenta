import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from sentence_transformers import SentenceTransformer
from proiect_licenta.doctor_data_pipeline import load_doctor_data, generate_extended_features
import gc

from pathlib import Path
BASE_DIR = Path("d:/Projects/Licenta/licenta/proiect_licenta/src/proiect_licenta")
MODELS_DIR = BASE_DIR / "models"

def main():
    print("Loading data...")
    df = load_doctor_data()
    df = df.sample(n=min(100000, len(df)), random_state=42).reset_index(drop=True)
    valid_categories = df['category'].value_counts()[lambda x: x >= 50].index
    df = df[df['category'].isin(valid_categories)].reset_index(drop=True)

    y_diag = df['category']
    le_diag = joblib.load(MODELS_DIR / "diagnosis_le.joblib")
    df['diag_encoded'] = le_diag.transform(y_diag)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diag_encoded'])
    df_test = df_test.reset_index(drop=True)
    
    del df, df_train
    gc.collect()

    print("Loading models...")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
    acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
    disp_model = joblib.load(MODELS_DIR / "disposition_model.joblib")
    diag_model = joblib.load(MODELS_DIR / "diagnosis_model.joblib")
    srv_model = joblib.load(MODELS_DIR / "service_model.joblib")
    srv_le = joblib.load(MODELS_DIR / "service_le.joblib")
    bert_name = joblib.load(MODELS_DIR / "doctor_bert_name.joblib")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model = SentenceTransformer(bert_name, device=device)

    print("Encoding Test subset...")
    X_test = generate_extended_features(df_test, tfidf, severity_map, acuity_model, disp_model, bert_model)
    y_test = df_test['diag_encoded'].values

    print("Evaluating Diagnosis...")
    diag_preds = diag_model.predict(X_test)
    rep_diag = classification_report(y_test, diag_preds, target_names=le_diag.classes_, zero_division=0)

    print("Evaluating Service...")
    X_test['predicted_diagnosis_enc'] = diag_model.predict(X_test)
    test_admit_mask = (df_test['admitted'] == 1) & (df_test['curr_service'].notnull())
    X_srv_test = X_test.loc[test_admit_mask].copy()
    y_srv_test_raw = df_test.loc[test_admit_mask, 'curr_service']

    valid_services = srv_le.classes_
    test_valid = y_srv_test_raw.isin(valid_services)
    X_srv_test = X_srv_test[test_valid]
    y_srv_test_raw = y_srv_test_raw[test_valid]
    y_srv_test = srv_le.transform(y_srv_test_raw)

    srv_preds = srv_model.predict(X_srv_test)
    rep_srv = classification_report(y_srv_test, srv_preds, target_names=srv_le.classes_, zero_division=0)

    with open("full_benchmark.txt", "w", encoding='utf-8') as f:
        f.write("=== DIAGNOSIS MODEL BENCHMARK ===\n")
        f.write(rep_diag + "\n")
        f.write("\n=== SERVICE MODEL BENCHMARK ===\n")
        f.write(rep_srv + "\n")
        
    print("Done! Saved to full_benchmark.txt.")

if __name__ == "__main__":
    main()
