"""
Benchmark: Doctor v1 vs v2 Comparison

Evaluates both model sets on the same held-out test data to quantify
the impact of adding vital signs and medication features (nurse data).

v1: complaints + demographics + triage predictions (2025 features)
v2: v1 + vital signs + clinical flags + medication categories (~2056 features)
"""

import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    cohen_kappa_score, top_k_accuracy_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent / "src" / "proiect_licenta"
MODELS_DIR = BASE_DIR / "models"
DOCTOR_MODELS_DIR = MODELS_DIR / "doctor"

import sys
sys.path.insert(0, str(BASE_DIR.parent))
from proiect_licenta.nurse_data_pipeline import (
    load_and_clean_data as load_v2_data,
    build_features as build_v2_features,
)
from proiect_licenta.doctor_data_pipeline import (
    build_features as build_v1_features,
    DEPARTMENT_NAMES,
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def evaluate_model(model, X_test, y_test, labels, model_name, cascading_model=None, X_test_base=None):
    """Evaluate a model and return metrics dict."""
    if cascading_model is not None and X_test_base is not None:
        # Department model: need cascading diagnosis prediction
        X = X_test_base.copy()
        X["predicted_diagnosis"] = cascading_model.predict(X_test_base)
    else:
        X = X_test

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(y_test, y_prob, k=3)
    top5 = top_k_accuracy_score(y_test, y_prob, k=min(5, len(labels)))
    kappa = cohen_kappa_score(y_test, y_pred)

    return {
        "accuracy": acc,
        "top3": top3,
        "top5": top5,
        "kappa": kappa,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def main():
    print("\n" + "#" * 70)
    print("  DOCTOR v1 vs v2 BENCHMARK COMPARISON")
    print("  Impact of vital signs + medication features")
    print("#" * 70)

    # ------------------------------------------------------------------
    # 1. Load data (v2 superset — includes vitals + meds)
    # ------------------------------------------------------------------
    df = load_v2_data()

    # ------------------------------------------------------------------
    # 2. Sample 100K (same seed as both training pipelines)
    # ------------------------------------------------------------------
    print_section("SAMPLING 100K + BUILDING FEATURES")
    if len(df) > 100_000:
        df, _ = train_test_split(
            df, train_size=100_000, random_state=42,
            stratify=df["diagnosis_group"],
        )
    print(f"  Sampled: {len(df):,}")

    # Build features for both versions
    print("\n  Building v1 features (triage only)...")
    features_v1 = build_v1_features(df)
    print(f"  v1 features: {features_v1.shape[1]}")

    print("\n  Building v2 features (triage + vitals + meds)...")
    features_v2 = build_v2_features(df)
    print(f"  v2 features: {features_v2.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Encode labels + split (same seed)
    # ------------------------------------------------------------------
    meta_v1 = json.loads((DOCTOR_MODELS_DIR / "doctor_metadata.json").read_text())
    meta_v2 = json.loads((DOCTOR_MODELS_DIR / "doctor_v2_metadata.json").read_text())

    diagnosis_labels = meta_v1["diagnosis_labels"]
    department_labels = meta_v1["department_labels"]

    diag_map = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_map = {l: i for i, l in enumerate(department_labels)}

    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)
    y_dept = df["service_group"].map(dept_map).reset_index(drop=True)

    # v1 split
    X1_train, X1_test, y_diag_train, y_diag_test, y_dept_train, y_dept_test = \
        train_test_split(features_v1, y_diag, y_dept,
                         test_size=0.2, random_state=42, stratify=y_diag)

    # v2 split (same indices)
    X2_train, X2_test, _, _, _, _ = \
        train_test_split(features_v2, y_diag, y_dept,
                         test_size=0.2, random_state=42, stratify=y_diag)

    print(f"  Train: {len(X1_train):,} | Test: {len(X1_test):,}")

    # ------------------------------------------------------------------
    # 4. Load models
    # ------------------------------------------------------------------
    print_section("LOADING MODELS")
    v1_diag = joblib.load(DOCTOR_MODELS_DIR / "diagnosis_model.joblib")
    v1_dept = joblib.load(DOCTOR_MODELS_DIR / "department_model.joblib")
    v2_diag = joblib.load(DOCTOR_MODELS_DIR / "diagnosis_model_v2.joblib")
    v2_dept = joblib.load(DOCTOR_MODELS_DIR / "department_model_v2.joblib")
    print(f"  v1 trained: {meta_v1['trained_at']}")
    print(f"  v2 trained: {meta_v2['trained_at']}")

    # ------------------------------------------------------------------
    # 5. Evaluate DIAGNOSIS models
    # ------------------------------------------------------------------
    print_section("DIAGNOSIS CATEGORY — v1 vs v2")

    d1 = evaluate_model(v1_diag, X1_test, y_diag_test, diagnosis_labels, "Diagnosis v1")
    d2 = evaluate_model(v2_diag, X2_test, y_diag_test, diagnosis_labels, "Diagnosis v2")

    print(f"\n  {'Metric':<30s}  {'v1':>10s}  {'v2':>10s}  {'Delta':>10s}")
    print(f"  {'-'*65}")
    for metric, label in [("accuracy", "Top-1 Accuracy"), ("top3", "Top-3 Accuracy"),
                           ("top5", "Top-5 Accuracy"), ("kappa", "Cohen's Kappa")]:
        v1_val = d1[metric]
        v2_val = d2[metric]
        delta = v2_val - v1_val
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<30s}  {v1_val:>9.4f}  {v2_val:>9.4f}  {sign}{delta:>9.4f}")

    # Per-class comparison
    print(f"\n  Per-Class Accuracy Comparison (Diagnosis):")
    print(f"  {'Category':<40s}  {'v1':>8s}  {'v2':>8s}  {'Delta':>8s}  {'n':>6s}")
    print(f"  {'-'*75}")
    for idx, label in enumerate(diagnosis_labels):
        mask = y_diag_test == idx
        if mask.sum() > 0:
            a1 = accuracy_score(y_diag_test[mask], d1["y_pred"][mask])
            a2 = accuracy_score(y_diag_test[mask], d2["y_pred"][mask])
            delta = a2 - a1
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<40s}  {a1:>7.3f}  {a2:>7.3f}  {sign}{delta:>7.3f}  {mask.sum():>6,}")

    # v2 classification report
    print(f"\n  v2 Classification Report (Diagnosis):")
    short_diag = [l[:30] for l in diagnosis_labels]
    print(classification_report(
        y_diag_test, d2["y_pred"],
        target_names=short_diag, digits=4, zero_division=0,
    ))

    # ------------------------------------------------------------------
    # 6. Evaluate DEPARTMENT models
    # ------------------------------------------------------------------
    print_section("DEPARTMENT — v1 vs v2")

    # Department models use cascading diagnosis prediction
    X1_test_dept = X1_test.copy()
    X1_test_dept["predicted_diagnosis"] = v1_diag.predict(X1_test)

    X2_test_dept = X2_test.copy()
    X2_test_dept["predicted_diagnosis"] = v2_diag.predict(X2_test)

    dp1_pred = v1_dept.predict(X1_test_dept)
    dp1_prob = v1_dept.predict_proba(X1_test_dept)
    dp2_pred = v2_dept.predict(X2_test_dept)
    dp2_prob = v2_dept.predict_proba(X2_test_dept)

    dp1_acc = accuracy_score(y_dept_test, dp1_pred)
    dp1_top3 = top_k_accuracy_score(y_dept_test, dp1_prob, k=3)
    dp1_kappa = cohen_kappa_score(y_dept_test, dp1_pred)

    dp2_acc = accuracy_score(y_dept_test, dp2_pred)
    dp2_top3 = top_k_accuracy_score(y_dept_test, dp2_prob, k=3)
    dp2_kappa = cohen_kappa_score(y_dept_test, dp2_pred)

    print(f"\n  {'Metric':<30s}  {'v1':>10s}  {'v2':>10s}  {'Delta':>10s}")
    print(f"  {'-'*65}")
    for label, v1_val, v2_val in [
        ("Top-1 Accuracy", dp1_acc, dp2_acc),
        ("Top-3 Accuracy", dp1_top3, dp2_top3),
        ("Cohen's Kappa", dp1_kappa, dp2_kappa),
    ]:
        delta = v2_val - v1_val
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<30s}  {v1_val:>9.4f}  {v2_val:>9.4f}  {sign}{delta:>9.4f}")

    # Per-class comparison
    print(f"\n  Per-Class Accuracy Comparison (Department):")
    print(f"  {'Department':<20s}  {'Full Name':<35s}  {'v1':>7s}  {'v2':>7s}  {'Delta':>7s}  {'n':>6s}")
    print(f"  {'-'*85}")
    for idx, label in enumerate(department_labels):
        mask = y_dept_test == idx
        if mask.sum() > 0:
            a1 = accuracy_score(y_dept_test[mask], dp1_pred[mask])
            a2 = accuracy_score(y_dept_test[mask], dp2_pred[mask])
            delta = a2 - a1
            sign = "+" if delta >= 0 else ""
            full = DEPARTMENT_NAMES.get(label, label)[:35]
            print(f"  {label:<20s}  {full:<35s}  {a1:>6.3f}  {a2:>6.3f}  {sign}{delta:>6.3f}  {mask.sum():>6,}")

    # v2 classification report
    print(f"\n  v2 Classification Report (Department):")
    print(classification_report(
        y_dept_test, dp2_pred,
        target_names=department_labels, digits=4, zero_division=0,
    ))

    # ------------------------------------------------------------------
    # 7. Feature importance — v2 top features (new features only)
    # ------------------------------------------------------------------
    print_section("TOP 20 FEATURE IMPORTANCES — v2 DIAGNOSIS MODEL")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_vocab_inv = {v: k for k, v in tfidf.vocabulary_.items()}
    feature_names = list(X2_test.columns)
    importances = v2_diag.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for rank, idx in enumerate(sorted_idx[:20], 1):
        name = feature_names[idx]
        if name.startswith("tfidf_"):
            tfidf_idx = int(name.split("_")[1])
            word = tfidf_vocab_inv.get(tfidf_idx, "?")
            display = f"{name} ({word})"
        else:
            display = name
        bar = "#" * max(1, int(40 * importances[idx] / importances[sorted_idx[0]]))
        print(f"  {rank:>2}. {display:<50s} {importances[idx]:.5f}  {bar}")

    # Highlight new (nurse) features in top 50
    print(f"\n  Nurse-specific features in top 50:")
    nurse_features = {
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
        "temperature_missing", "heartrate_missing", "resprate_missing",
        "o2sat_missing", "sbp_missing", "dbp_missing",
        "fever", "tachycardia", "bradycardia", "tachypnea",
        "hypoxia", "hypertension", "hypotension", "map",
        "n_medications", "meds_unknown",
        "has_cardiac_meds", "has_diabetes_meds", "has_psych_meds",
        "has_respiratory_meds", "has_opioid_meds", "has_anticoagulant_meds",
        "has_gi_meds", "has_thyroid_meds", "has_anticonvulsant_meds",
    }
    for rank, idx in enumerate(sorted_idx[:50], 1):
        name = feature_names[idx]
        if name in nurse_features:
            print(f"    #{rank:>2}: {name:<35s}  importance={importances[idx]:.5f}")

    # ------------------------------------------------------------------
    # 8. Baseline comparisons
    # ------------------------------------------------------------------
    print_section("BASELINE COMPARISONS")

    diag_majority = y_diag_train.mode()[0]
    diag_majority_acc = accuracy_score(y_diag_test, [diag_majority] * len(y_diag_test))
    dept_majority = y_dept_train.mode()[0]
    dept_majority_acc = accuracy_score(y_dept_test, [dept_majority] * len(y_dept_test))

    print(f"  Diagnosis majority baseline:   {diag_majority_acc:.4f} (always '{diagnosis_labels[diag_majority]}')")
    print(f"  Diagnosis v1:                  {d1['accuracy']:.4f}  (+{d1['accuracy']-diag_majority_acc:.4f})")
    print(f"  Diagnosis v2:                  {d2['accuracy']:.4f}  (+{d2['accuracy']-diag_majority_acc:.4f})")
    print(f"")
    print(f"  Department majority baseline:  {dept_majority_acc:.4f} (always '{department_labels[dept_majority]}')")
    print(f"  Department v1:                 {dp1_acc:.4f}  (+{dp1_acc-dept_majority_acc:.4f})")
    print(f"  Department v2:                 {dp2_acc:.4f}  (+{dp2_acc-dept_majority_acc:.4f})")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print_section("BENCHMARK SUMMARY")
    print(f"  Test set size:          {len(y_diag_test):,}")
    print(f"  v1 features:            {features_v1.shape[1]}")
    print(f"  v2 features:            {features_v2.shape[1]} (+{features_v2.shape[1]-features_v1.shape[1]} nurse features)")
    print(f"")
    print(f"  DIAGNOSIS (14 classes):")
    print(f"    {'':20s}  {'v1':>8s}  {'v2':>8s}  {'Improve':>8s}")
    print(f"    {'Top-1 accuracy':<20s}  {d1['accuracy']:>7.2%}  {d2['accuracy']:>7.2%}  {d2['accuracy']-d1['accuracy']:>+7.2%}")
    print(f"    {'Top-3 accuracy':<20s}  {d1['top3']:>7.2%}  {d2['top3']:>7.2%}  {d2['top3']-d1['top3']:>+7.2%}")
    print(f"    {'Cohen kappa':<20s}  {d1['kappa']:>7.4f}  {d2['kappa']:>7.4f}  {d2['kappa']-d1['kappa']:>+7.4f}")
    print(f"")
    print(f"  DEPARTMENT (11 classes):")
    print(f"    {'':20s}  {'v1':>8s}  {'v2':>8s}  {'Improve':>8s}")
    print(f"    {'Top-1 accuracy':<20s}  {dp1_acc:>7.2%}  {dp2_acc:>7.2%}  {dp2_acc-dp1_acc:>+7.2%}")
    print(f"    {'Top-3 accuracy':<20s}  {dp1_top3:>7.2%}  {dp2_top3:>7.2%}  {dp2_top3-dp1_top3:>+7.2%}")
    print(f"    {'Cohen kappa':<20s}  {dp1_kappa:>7.4f}  {dp2_kappa:>7.4f}  {dp2_kappa-dp1_kappa:>+7.4f}")

    print("\n" + "#" * 70)
    print("  BENCHMARK COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
