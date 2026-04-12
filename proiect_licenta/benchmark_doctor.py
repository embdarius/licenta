"""
Benchmark script for the trained doctor models (diagnosis + department).

Reproduces the exact data loading, sampling, and train/test split from training
(random_state=42, stratified), loads saved model artifacts, and evaluates
with detailed metrics.
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent / "src" / "proiect_licenta"
DOCTOR_MODELS_DIR = BASE_DIR / "models" / "doctor"

import sys
sys.path.insert(0, str(BASE_DIR.parent))
from proiect_licenta.doctor_data_pipeline import (
    load_and_clean_data, build_features,
    DIAGNOSIS_GROUP_MAP, SERVICE_GROUP_MAP, DEPARTMENT_NAMES,
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_confusion_matrix(cm, labels, short_labels=None):
    """Pretty-print a confusion matrix with truncated labels."""
    display = short_labels or labels
    col_w = max(8, max(len(str(l)) for l in display) + 1)
    row_w = max(len(str(l)) for l in display) + 2

    header = " " * (row_w + 4) + "".join(f"{('P:' + str(l)):>{col_w}}" for l in display)
    print(header)
    print(" " * (row_w + 4) + "-" * (col_w * len(display)))
    for i, label in enumerate(display):
        row_vals = "".join(f"{cm[i][j]:>{col_w},}" for j in range(len(display)))
        print(f"  {'T:' + str(label):>{row_w}} |{row_vals}")


def main():
    print("\n" + "#" * 70)
    print("  DOCTOR MODEL BENCHMARK")
    print("  Evaluating diagnosis + department models on held-out test set (20%)")
    print("#" * 70)

    # ------------------------------------------------------------------
    # 1. Load data (same as training)
    # ------------------------------------------------------------------
    df = load_and_clean_data()

    # ------------------------------------------------------------------
    # 2. Reproduce exact sampling + split
    # ------------------------------------------------------------------
    print_section("SAMPLING 100K + TRAIN/TEST SPLIT (80/20, stratified)")
    if len(df) > 100_000:
        df, _ = train_test_split(
            df, train_size=100_000, random_state=42,
            stratify=df["diagnosis_group"],
        )
    print(f"  Sampled: {len(df):,}")

    features = build_features(df)

    # Encode labels
    metadata = json.loads((DOCTOR_MODELS_DIR / "doctor_metadata.json").read_text())
    diagnosis_labels = metadata["diagnosis_labels"]
    department_labels = metadata["department_labels"]
    diag_label_to_idx = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_label_to_idx = {l: i for i, l in enumerate(department_labels)}

    y_diagnosis = df["diagnosis_group"].map(diag_label_to_idx).reset_index(drop=True)
    y_department = df["service_group"].map(dept_label_to_idx).reset_index(drop=True)

    X_train, X_test, y_diag_train, y_diag_test, y_dept_train, y_dept_test = \
        train_test_split(
            features, y_diagnosis, y_department,
            test_size=0.2, random_state=42,
            stratify=y_diagnosis,
        )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 3. Load saved models
    # ------------------------------------------------------------------
    print_section("LOADING SAVED MODELS")
    diagnosis_model = joblib.load(DOCTOR_MODELS_DIR / "diagnosis_model.joblib")
    department_model = joblib.load(DOCTOR_MODELS_DIR / "department_model.joblib")
    print(f"  Loaded diagnosis_model + department_model")
    print(f"  Trained at: {metadata['trained_at']}")

    # ===================================================================
    #  DIAGNOSIS CATEGORY MODEL BENCHMARK
    # ===================================================================
    print_section("DIAGNOSIS CATEGORY MODEL — 14 CLASSES")

    y_pred_diag = diagnosis_model.predict(X_test)
    y_prob_diag = diagnosis_model.predict_proba(X_test)

    diag_acc = accuracy_score(y_diag_test, y_pred_diag)

    # Top-k accuracy
    top3_acc = top_k_accuracy_score(y_diag_test, y_prob_diag, k=3)
    top5_acc = top_k_accuracy_score(y_diag_test, y_prob_diag, k=5)

    print(f"\n  Overall Metrics:")
    print(f"    Top-1 accuracy (exact):  {diag_acc:.4f}  ({diag_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:          {top3_acc:.4f}  ({top3_acc*100:.2f}%)")
    print(f"    Top-5 accuracy:          {top5_acc:.4f}  ({top5_acc*100:.2f}%)")

    # Kappa
    kappa = cohen_kappa_score(y_diag_test, y_pred_diag)
    print(f"    Cohen's kappa:           {kappa:.4f}")

    # Exclude catch-all accuracy
    catchall_idx = diag_label_to_idx.get("Symptoms, Signs, Ill-Defined")
    if catchall_idx is not None:
        mask_no_catchall = y_diag_test != catchall_idx
        acc_no_catchall = accuracy_score(
            y_diag_test[mask_no_catchall], y_pred_diag[mask_no_catchall]
        )
        print(f"\n    Accuracy EXCLUDING 'Symptoms/Signs/Ill-Defined':")
        print(f"      {acc_no_catchall:.4f}  ({acc_no_catchall*100:.2f}%) on {mask_no_catchall.sum():,} samples")

    # Per-class report
    print(f"\n  Per-Class Classification Report:")
    # Shorten names for display
    short_diag = [l[:30] for l in diagnosis_labels]
    print(classification_report(
        y_diag_test, y_pred_diag,
        target_names=short_diag, digits=4, zero_division=0,
    ))

    # Per-class accuracy
    print(f"  Per-Class Accuracy:")
    for idx, label in enumerate(diagnosis_labels):
        mask = y_diag_test == idx
        if mask.sum() > 0:
            acc = accuracy_score(y_diag_test[mask], y_pred_diag[mask])
            correct = (y_pred_diag[mask] == idx).sum()
            total = mask.sum()
            print(f"    {label:42s}  {acc:.4f}  ({correct:,}/{total:,})")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    cm_diag = confusion_matrix(y_diag_test, y_pred_diag)
    # Use abbreviated labels
    abbrev_diag = [
        "Blood", "Circ", "Digest", "Endo", "Genito",
        "Infect", "Injury", "Mental", "Musculo", "Nerv",
        "Other", "Resp", "Skin", "Sx/Ill"
    ]
    print_confusion_matrix(cm_diag, diagnosis_labels, abbrev_diag)

    # Confidence analysis
    max_probs = np.max(y_prob_diag, axis=1)
    correct_mask = y_pred_diag == y_diag_test.values
    print(f"\n  Confidence Analysis:")
    print(f"    Mean confidence (correct):   {max_probs[correct_mask].mean():.4f}")
    print(f"    Mean confidence (incorrect): {max_probs[~correct_mask].mean():.4f}")

    # Top misclassification pairs
    print(f"\n  Top 15 Misclassification Pairs (True -> Predicted, count):")
    errors = []
    for i in range(len(diagnosis_labels)):
        for j in range(len(diagnosis_labels)):
            if i != j and cm_diag[i][j] > 0:
                errors.append((cm_diag[i][j], diagnosis_labels[i], diagnosis_labels[j]))
    errors.sort(reverse=True)
    for count, true_l, pred_l in errors[:15]:
        true_total = (y_diag_test == diag_label_to_idx[true_l]).sum()
        print(f"    {true_l:35s} -> {pred_l:35s}  {count:>5,}  ({100*count/true_total:5.1f}% of true class)")

    # Feature importance (top 20)
    print_section("TOP 20 FEATURE IMPORTANCES — DIAGNOSIS MODEL")
    tfidf = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.joblib")
    tfidf_vocab_inv = {v: k for k, v in tfidf.vocabulary_.items()}
    feature_names = list(X_test.columns)
    importances = diagnosis_model.feature_importances_
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

    # ===================================================================
    #  DEPARTMENT MODEL BENCHMARK
    # ===================================================================
    print_section("DEPARTMENT MODEL — 11 CLASSES")

    X_test_dept = X_test.copy()
    X_test_dept["predicted_diagnosis"] = y_pred_diag  # cascading

    y_pred_dept = department_model.predict(X_test_dept)
    y_prob_dept = department_model.predict_proba(X_test_dept)

    dept_acc = accuracy_score(y_dept_test, y_pred_dept)
    dept_top3 = top_k_accuracy_score(y_dept_test, y_prob_dept, k=3)
    dept_kappa = cohen_kappa_score(y_dept_test, y_pred_dept)

    print(f"\n  Overall Metrics:")
    print(f"    Top-1 accuracy (exact):  {dept_acc:.4f}  ({dept_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:          {dept_top3:.4f}  ({dept_top3*100:.2f}%)")
    print(f"    Cohen's kappa:           {dept_kappa:.4f}")

    # Exclude MED accuracy (majority class)
    med_idx = dept_label_to_idx.get("MED")
    if med_idx is not None:
        mask_no_med = y_dept_test != med_idx
        acc_no_med = accuracy_score(
            y_dept_test[mask_no_med], y_pred_dept[mask_no_med]
        )
        print(f"\n    Accuracy EXCLUDING 'MED' (majority class):")
        print(f"      {acc_no_med:.4f}  ({acc_no_med*100:.2f}%) on {mask_no_med.sum():,} samples")

    # Per-class report
    print(f"\n  Per-Class Classification Report:")
    dept_display = [f"{l} ({DEPARTMENT_NAMES.get(l, l)[:25]})" for l in department_labels]
    print(classification_report(
        y_dept_test, y_pred_dept,
        target_names=department_labels, digits=4, zero_division=0,
    ))

    # Per-class accuracy
    print(f"  Per-Class Accuracy:")
    for idx, label in enumerate(department_labels):
        mask = y_dept_test == idx
        if mask.sum() > 0:
            acc = accuracy_score(y_dept_test[mask], y_pred_dept[mask])
            correct = (y_pred_dept[mask] == idx).sum()
            total = mask.sum()
            full_name = DEPARTMENT_NAMES.get(label, label)
            print(f"    {label:12s} ({full_name:42s})  {acc:.4f}  ({correct:,}/{total:,})")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    cm_dept = confusion_matrix(y_dept_test, y_pred_dept)
    print_confusion_matrix(cm_dept, department_labels)

    # Department accuracy by diagnosis category
    print(f"\n  Department Accuracy by True Diagnosis Category:")
    y_diag_test_vals = y_diag_test.values
    for idx, label in enumerate(diagnosis_labels):
        mask = y_diag_test_vals == idx
        if mask.sum() >= 50:
            acc = accuracy_score(y_dept_test[mask], y_pred_dept[mask])
            print(f"    {label:42s}  acc={acc:.4f}  (n={mask.sum():,})")

    # ===================================================================
    #  BASELINE COMPARISONS
    # ===================================================================
    print_section("BASELINE COMPARISONS")

    # Majority class baseline
    diag_majority = y_diag_train.mode()[0]
    diag_majority_acc = accuracy_score(y_diag_test, [diag_majority] * len(y_diag_test))
    print(f"  Diagnosis — majority class baseline: {diag_majority_acc:.4f} "
          f"(always predict '{diagnosis_labels[diag_majority]}')")
    print(f"  Diagnosis — model accuracy:          {diag_acc:.4f}  "
          f"(+{diag_acc - diag_majority_acc:.4f} over baseline)")

    dept_majority = y_dept_train.mode()[0]
    dept_majority_acc = accuracy_score(y_dept_test, [dept_majority] * len(y_dept_test))
    print(f"\n  Department — majority class baseline: {dept_majority_acc:.4f} "
          f"(always predict '{department_labels[dept_majority]}')")
    print(f"  Department — model accuracy:          {dept_acc:.4f}  "
          f"(+{dept_acc - dept_majority_acc:.4f} over baseline)")

    # Random weighted baseline
    rng = np.random.RandomState(42)
    diag_priors = y_diag_train.value_counts(normalize=True).sort_index()
    random_diag = rng.choice(len(diagnosis_labels), size=len(y_diag_test), p=diag_priors.values)
    random_diag_acc = accuracy_score(y_diag_test, random_diag)
    print(f"\n  Diagnosis — weighted random baseline: {random_diag_acc:.4f}")
    print(f"  Diagnosis — model lift over random:   {diag_acc / random_diag_acc:.2f}x")

    dept_priors = y_dept_train.value_counts(normalize=True).sort_index()
    random_dept = rng.choice(len(department_labels), size=len(y_dept_test), p=dept_priors.values)
    random_dept_acc = accuracy_score(y_dept_test, random_dept)
    print(f"\n  Department — weighted random baseline: {random_dept_acc:.4f}")
    print(f"  Department — model lift over random:   {dept_acc / random_dept_acc:.2f}x")

    # ===================================================================
    #  SUMMARY
    # ===================================================================
    print_section("BENCHMARK SUMMARY")
    print(f"  Test set size:                    {len(y_diag_test):,}")
    print(f"  ---")
    print(f"  DIAGNOSIS MODEL (14 classes):")
    print(f"    Top-1 accuracy:                 {diag_acc:.4f}  ({diag_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:                 {top3_acc:.4f}  ({top3_acc*100:.2f}%)")
    print(f"    Top-5 accuracy:                 {top5_acc:.4f}  ({top5_acc*100:.2f}%)")
    print(f"    Cohen's kappa:                  {kappa:.4f}")
    print(f"    Accuracy excl. catch-all:       {acc_no_catchall:.4f}  ({acc_no_catchall*100:.2f}%)")
    print(f"    Majority baseline:              {diag_majority_acc:.4f}")
    print(f"  ---")
    print(f"  DEPARTMENT MODEL (11 classes):")
    print(f"    Top-1 accuracy:                 {dept_acc:.4f}  ({dept_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:                 {dept_top3:.4f}  ({dept_top3*100:.2f}%)")
    print(f"    Cohen's kappa:                  {dept_kappa:.4f}")
    print(f"    Accuracy excl. MED:             {acc_no_med:.4f}  ({acc_no_med*100:.2f}%)")
    print(f"    Majority baseline:              {dept_majority_acc:.4f}")

    print("\n" + "#" * 70)
    print("  BENCHMARK COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
