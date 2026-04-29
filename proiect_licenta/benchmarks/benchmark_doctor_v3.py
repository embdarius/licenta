"""
Benchmark script for the Doctor v3 base model (catch-all excluded, 13 classes).

Reproduces the exact data loading and train/test split from training
(random_state=42, stratified, NO sub-sampling), loads saved v3_base
artifacts, and evaluates with the same metrics as benchmark_doctor.py.

The v3 label space drops the "Symptoms, Signs, Ill-Defined" catch-all class,
so per-class metrics here are not directly comparable to v1's 14-class
numbers. Use benchmarks/compare_all_versions.py for the four-way overview.
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
# Make the proiect_licenta package importable from this script
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import (
    TRIAGE_V1_DIR as TRIAGE_MODELS_DIR,
    DOCTOR_V3_BASE_DIR as DOCTOR_MODELS_DIR,
)
from proiect_licenta.training.train_doctor import (
    DIAGNOSIS_GROUP_MAP, SERVICE_GROUP_MAP, DEPARTMENT_NAMES,
    CATCH_ALL_LABEL,
)
from proiect_licenta.training.train_doctor_v3 import (
    load_and_clean_data, build_features,
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
    print("  DOCTOR v3 BASE MODEL BENCHMARK")
    print(f"  Catch-all class '{CATCH_ALL_LABEL}' EXCLUDED — 13-class label space")
    print("  Held-out test set (20%)")
    print("#" * 70)

    # ------------------------------------------------------------------
    # 1. Load data (catch-all filter applied inside)
    # ------------------------------------------------------------------
    df = load_and_clean_data()

    # ------------------------------------------------------------------
    # 2. Reproduce exact split (NO sub-sampling for v3)
    # ------------------------------------------------------------------
    print_section("TRAIN/TEST SPLIT (80/20, stratified — full filtered dataset)")
    print(f"  Full filtered dataset: {len(df):,}")

    features = build_features(df)

    # Encode labels
    metadata = json.loads((DOCTOR_MODELS_DIR / "metadata.json").read_text())
    diagnosis_labels = metadata["diagnosis_labels"]
    department_labels = metadata["department_labels"]
    diag_label_to_idx = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_label_to_idx = {l: i for i, l in enumerate(department_labels)}

    assert CATCH_ALL_LABEL not in diagnosis_labels, (
        f"v3 metadata still contains catch-all '{CATCH_ALL_LABEL}' — bug"
    )

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
    # 3. Load saved v3 base models
    # ------------------------------------------------------------------
    print_section("LOADING SAVED v3 BASE MODELS")
    diagnosis_model = joblib.load(DOCTOR_MODELS_DIR / "diagnosis_model.joblib")
    department_model = joblib.load(DOCTOR_MODELS_DIR / "department_model.joblib")
    print(f"  Loaded diagnosis_model + department_model from {DOCTOR_MODELS_DIR.name}/")
    print(f"  Trained at: {metadata['trained_at']}")
    print(f"  Catch-all excluded: {metadata.get('catch_all_excluded', '?')}")

    # ===================================================================
    #  DIAGNOSIS MODEL BENCHMARK
    # ===================================================================
    print_section(f"DIAGNOSIS CATEGORY MODEL — {len(diagnosis_labels)} CLASSES")

    y_pred_diag = diagnosis_model.predict(X_test)
    y_prob_diag = diagnosis_model.predict_proba(X_test)

    diag_acc = accuracy_score(y_diag_test, y_pred_diag)
    top3_acc = top_k_accuracy_score(y_diag_test, y_prob_diag, k=3)
    top5_acc = top_k_accuracy_score(y_diag_test, y_prob_diag, k=5)
    kappa = cohen_kappa_score(y_diag_test, y_pred_diag)

    print(f"\n  Overall Metrics:")
    print(f"    Top-1 accuracy:          {diag_acc:.4f}  ({diag_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:          {top3_acc:.4f}  ({top3_acc*100:.2f}%)")
    print(f"    Top-5 accuracy:          {top5_acc:.4f}  ({top5_acc*100:.2f}%)")
    print(f"    Cohen's kappa:           {kappa:.4f}")

    # Per-class report
    print(f"\n  Per-Class Classification Report:")
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

    # Confusion matrix (auto-truncate labels for display)
    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    cm_diag = confusion_matrix(y_diag_test, y_pred_diag)
    abbrev_diag = [l[:6] for l in diagnosis_labels]
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
        print(f"    {true_l:35s} -> {pred_l:35s}  {count:>5,}  "
              f"({100*count/true_total:5.1f}% of true class)")

    # Feature importance (top 20)
    print_section("TOP 20 FEATURE IMPORTANCES — DIAGNOSIS MODEL")
    tfidf = joblib.load(TRIAGE_MODELS_DIR / "tfidf_vectorizer.joblib")
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
    print_section(f"DEPARTMENT MODEL — {len(department_labels)} CLASSES")

    X_test_dept = X_test.copy()
    X_test_dept["predicted_diagnosis"] = y_pred_diag  # cascading

    y_pred_dept = department_model.predict(X_test_dept)
    y_prob_dept = department_model.predict_proba(X_test_dept)

    dept_acc = accuracy_score(y_dept_test, y_pred_dept)
    dept_top3 = top_k_accuracy_score(y_dept_test, y_prob_dept, k=3)
    dept_kappa = cohen_kappa_score(y_dept_test, y_pred_dept)

    print(f"\n  Overall Metrics:")
    print(f"    Top-1 accuracy:          {dept_acc:.4f}  ({dept_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:          {dept_top3:.4f}  ({dept_top3*100:.2f}%)")
    print(f"    Cohen's kappa:           {dept_kappa:.4f}")

    # Exclude MED accuracy (majority class)
    med_idx = dept_label_to_idx.get("MED")
    acc_no_med = None
    if med_idx is not None:
        mask_no_med = y_dept_test != med_idx
        acc_no_med = accuracy_score(
            y_dept_test[mask_no_med], y_pred_dept[mask_no_med]
        )
        print(f"\n    Accuracy EXCLUDING 'MED' (majority class):")
        print(f"      {acc_no_med:.4f}  ({acc_no_med*100:.2f}%) on {mask_no_med.sum():,} samples")

    # Per-class report
    print(f"\n  Per-Class Classification Report:")
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

    diag_majority = y_diag_train.mode()[0]
    diag_majority_acc = accuracy_score(y_diag_test, [diag_majority] * len(y_diag_test))
    print(f"  Diagnosis - majority class baseline: {diag_majority_acc:.4f} "
          f"(always predict '{diagnosis_labels[diag_majority]}')")
    print(f"  Diagnosis - model accuracy:          {diag_acc:.4f}  "
          f"(+{diag_acc - diag_majority_acc:.4f} over baseline)")

    dept_majority = y_dept_train.mode()[0]
    dept_majority_acc = accuracy_score(y_dept_test, [dept_majority] * len(y_dept_test))
    print(f"\n  Department - majority class baseline: {dept_majority_acc:.4f} "
          f"(always predict '{department_labels[dept_majority]}')")
    print(f"  Department - model accuracy:          {dept_acc:.4f}  "
          f"(+{dept_acc - dept_majority_acc:.4f} over baseline)")

    rng = np.random.RandomState(42)
    diag_priors = y_diag_train.value_counts(normalize=True).sort_index()
    random_diag = rng.choice(len(diagnosis_labels), size=len(y_diag_test), p=diag_priors.values)
    random_diag_acc = accuracy_score(y_diag_test, random_diag)
    print(f"\n  Diagnosis - weighted random baseline: {random_diag_acc:.4f}")
    print(f"  Diagnosis - model lift over random:   {diag_acc / random_diag_acc:.2f}x")

    dept_priors = y_dept_train.value_counts(normalize=True).sort_index()
    random_dept = rng.choice(len(department_labels), size=len(y_dept_test), p=dept_priors.values)
    random_dept_acc = accuracy_score(y_dept_test, random_dept)
    print(f"\n  Department - weighted random baseline: {random_dept_acc:.4f}")
    print(f"  Department - model lift over random:   {dept_acc / random_dept_acc:.2f}x")

    # ===================================================================
    #  SUMMARY
    # ===================================================================
    print_section("BENCHMARK SUMMARY (v3 base)")
    print(f"  Test set size:                    {len(y_diag_test):,}")
    print(f"  Catch-all excluded:               {metadata.get('catch_all_excluded', '?')}")
    print(f"  ---")
    print(f"  DIAGNOSIS MODEL ({len(diagnosis_labels)} classes):")
    print(f"    Top-1 accuracy:                 {diag_acc:.4f}  ({diag_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:                 {top3_acc:.4f}  ({top3_acc*100:.2f}%)")
    print(f"    Top-5 accuracy:                 {top5_acc:.4f}  ({top5_acc*100:.2f}%)")
    print(f"    Cohen's kappa:                  {kappa:.4f}")
    print(f"    Majority baseline:              {diag_majority_acc:.4f}")
    print(f"  ---")
    print(f"  DEPARTMENT MODEL ({len(department_labels)} classes):")
    print(f"    Top-1 accuracy:                 {dept_acc:.4f}  ({dept_acc*100:.2f}%)")
    print(f"    Top-3 accuracy:                 {dept_top3:.4f}  ({dept_top3*100:.2f}%)")
    print(f"    Cohen's kappa:                  {dept_kappa:.4f}")
    if acc_no_med is not None:
        print(f"    Accuracy excl. MED:             {acc_no_med:.4f}  ({acc_no_med*100:.2f}%)")
    print(f"    Majority baseline:              {dept_majority_acc:.4f}")

    print("\n" + "#" * 70)
    print("  BENCHMARK COMPLETE (v3 base)")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
