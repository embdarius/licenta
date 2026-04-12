"""
Benchmark script for the trained triage models.

Reproduces the exact train/test split from training (random_state=42, stratified),
loads saved model artifacts, and evaluates with detailed metrics.
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths (same as data_pipeline.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent / "src" / "proiect_licenta"
DATASET_DIR = BASE_DIR / "datasets" / "datasets_mimic-iv" / "mimic-iv-ed"
HOSP_DIR = BASE_DIR / "datasets" / "datasets_mimic-iv" / "mimic-iv" / "hosp"
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Import shared preprocessing from data_pipeline
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(BASE_DIR.parent))
from proiect_licenta.data_pipeline import (
    load_and_clean_data, build_features, normalize_complaint_text,
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_confusion_matrix(cm, labels):
    """Pretty-print a confusion matrix."""
    col_width = max(8, max(len(str(l)) for l in labels) + 2)
    header = " " * (col_width + 4) + "".join(f"{'Pred ' + str(l):>{col_width}}" for l in labels)
    print(header)
    print(" " * (col_width + 4) + "-" * (col_width * len(labels)))
    for i, label in enumerate(labels):
        row_vals = "".join(f"{cm[i][j]:>{col_width},}" for j in range(len(labels)))
        print(f"  {'True ' + str(label):>{col_width}} |{row_vals}")


def main():
    print("\n" + "#" * 70)
    print("  TRIAGE MODEL BENCHMARK")
    print("  Evaluating saved models on held-out test set (20%)")
    print("#" * 70)

    # ------------------------------------------------------------------
    # 1. Load data (same as training)
    # ------------------------------------------------------------------
    df = load_and_clean_data()

    # ------------------------------------------------------------------
    # 2. Reproduce exact train/test split
    # ------------------------------------------------------------------
    print_section("TRAIN/TEST SPLIT (80/20, stratified, random_state=42)")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["acuity"],
    )
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # ------------------------------------------------------------------
    # 3. Build features using saved artifacts
    # ------------------------------------------------------------------
    print_section("LOADING SAVED ARTIFACTS")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
    acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
    disposition_model = joblib.load(MODELS_DIR / "disposition_model.joblib")
    print("  Loaded: tfidf_vectorizer, severity_map, acuity_model, disposition_model")

    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    print(f"  Model version: {metadata['version']}")
    print(f"  Trained at: {metadata['trained_at']}")

    # Build features for train (fit=True to get severity_map) and test (fit=False)
    import io, contextlib
    # We need the train features to compute severity map consistently
    # But we use saved artifacts for test set
    with contextlib.redirect_stdout(io.StringIO()):
        X_test, _, _ = build_features(test_df, tfidf=tfidf, severity_map=severity_map, fit=False)

    y_acuity_test = test_df["acuity"].reset_index(drop=True)
    y_admit_test = test_df["admitted"].reset_index(drop=True)

    # ===================================================================
    #  ACUITY MODEL BENCHMARK
    # ===================================================================
    print_section("ACUITY MODEL — ESI 1-5 PREDICTION")

    y_pred_acuity = acuity_model.predict(X_test) + 1  # model outputs 0-4
    y_prob_acuity = acuity_model.predict_proba(X_test)

    # --- Overall accuracy ---
    exact_acc = accuracy_score(y_acuity_test, y_pred_acuity)
    within_1 = float(np.mean(np.abs(y_pred_acuity - y_acuity_test.values) <= 1))
    within_2 = float(np.mean(np.abs(y_pred_acuity - y_acuity_test.values) <= 2))

    print(f"\n  Overall Metrics:")
    print(f"    Exact accuracy:           {exact_acc:.4f}  ({exact_acc*100:.2f}%)")
    print(f"    Within-1-level accuracy:  {within_1:.4f}  ({within_1*100:.2f}%)")
    print(f"    Within-2-level accuracy:  {within_2:.4f}  ({within_2*100:.2f}%)")

    # --- Mean Absolute Error ---
    mae = np.mean(np.abs(y_pred_acuity - y_acuity_test.values))
    print(f"    Mean Absolute Error:      {mae:.4f} ESI levels")

    # --- Per-class metrics ---
    print(f"\n  Per-Class Classification Report:")
    report = classification_report(
        y_acuity_test, y_pred_acuity,
        labels=[1, 2, 3, 4, 5],
        target_names=["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"],
        digits=4,
        zero_division=0,
    )
    print(report)

    # --- Per-class accuracy (how often each true class is predicted correctly) ---
    print(f"  Per-Class Accuracy (recall = correct predictions / total in class):")
    for esi in [1, 2, 3, 4, 5]:
        mask = y_acuity_test == esi
        if mask.sum() > 0:
            class_acc = accuracy_score(y_acuity_test[mask], y_pred_acuity[mask])
            total = mask.sum()
            correct = (y_pred_acuity[mask] == esi).sum()
            print(f"    ESI {esi}: {class_acc:.4f}  ({correct:,}/{total:,})")

    # --- Confusion Matrix ---
    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    cm_acuity = confusion_matrix(y_acuity_test, y_pred_acuity, labels=[1, 2, 3, 4, 5])
    print_confusion_matrix(cm_acuity, [1, 2, 3, 4, 5])

    # --- Error distribution: where do misclassifications go? ---
    print(f"\n  Error Analysis — Misclassification Direction:")
    errors = y_pred_acuity - y_acuity_test.values
    total_errors = (errors != 0).sum()
    over_triage = (errors < 0).sum()   # predicted more severe (lower number)
    under_triage = (errors > 0).sum()  # predicted less severe (higher number)
    print(f"    Total misclassified:  {total_errors:,} / {len(y_acuity_test):,} ({100*total_errors/len(y_acuity_test):.1f}%)")
    print(f"    Over-triaged  (pred more severe):  {over_triage:,} ({100*over_triage/total_errors:.1f}% of errors)")
    print(f"    Under-triaged (pred less severe):   {under_triage:,} ({100*under_triage/total_errors:.1f}% of errors)")

    # --- Per-error-magnitude breakdown ---
    print(f"\n  Error Magnitude Distribution:")
    for delta in range(-4, 5):
        count = (errors == delta).sum()
        if count > 0:
            bar = "#" * max(1, int(50 * count / len(y_acuity_test)))
            label = "CORRECT" if delta == 0 else f"{'over' if delta < 0 else 'under'}-triage by {abs(delta)}"
            print(f"    {delta:+d} ({label:25s}): {count:>6,} ({100*count/len(y_acuity_test):5.1f}%) {bar}")

    # --- Weighted Kappa (ordinal agreement) ---
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa_linear = cohen_kappa_score(y_acuity_test, y_pred_acuity, weights="linear")
        kappa_quadratic = cohen_kappa_score(y_acuity_test, y_pred_acuity, weights="quadratic")
        print(f"\n  Ordinal Agreement (Cohen's Kappa):")
        print(f"    Linear-weighted kappa:    {kappa_linear:.4f}")
        print(f"    Quadratic-weighted kappa: {kappa_quadratic:.4f}")
    except ImportError:
        pass

    # --- Multiclass AUC ---
    try:
        auc_ovr = roc_auc_score(y_acuity_test, y_prob_acuity, multi_class="ovr", labels=[0,1,2,3,4])
        auc_ovo = roc_auc_score(y_acuity_test, y_prob_acuity, multi_class="ovo", labels=[0,1,2,3,4])
        print(f"\n  ROC AUC (multiclass):")
        print(f"    One-vs-Rest: {auc_ovr:.4f}")
        print(f"    One-vs-One:  {auc_ovo:.4f}")
    except Exception as e:
        print(f"\n  ROC AUC: could not compute ({e})")

    # --- Mean confidence for correct vs incorrect ---
    max_probs = np.max(y_prob_acuity, axis=1)
    correct_mask = y_pred_acuity == y_acuity_test.values
    print(f"\n  Confidence Analysis:")
    print(f"    Mean confidence (correct preds):   {max_probs[correct_mask].mean():.4f}")
    print(f"    Mean confidence (incorrect preds): {max_probs[~correct_mask].mean():.4f}")
    print(f"    Overall mean confidence:           {max_probs.mean():.4f}")

    # ===================================================================
    #  DISPOSITION MODEL BENCHMARK
    # ===================================================================
    print_section("DISPOSITION MODEL — ADMITTED vs NOT ADMITTED")

    X_test_disp = X_test.copy()
    X_test_disp["predicted_acuity"] = y_pred_acuity  # cascading

    y_pred_disp = disposition_model.predict(X_test_disp)
    y_prob_disp = disposition_model.predict_proba(X_test_disp)[:, 1]

    disp_acc = accuracy_score(y_admit_test, y_pred_disp)

    print(f"\n  Overall Accuracy: {disp_acc:.4f} ({disp_acc*100:.2f}%)")

    # --- Classification report ---
    print(f"\n  Classification Report:")
    print(classification_report(
        y_admit_test, y_pred_disp,
        target_names=["NOT ADMITTED", "ADMITTED"],
        digits=4,
    ))

    # --- Confusion matrix ---
    cm_disp = confusion_matrix(y_admit_test, y_pred_disp)
    print(f"  Confusion Matrix:")
    print_confusion_matrix(cm_disp, ["NOT ADM", "ADMITTED"])

    # --- Binary AUC ---
    try:
        auc_disp = roc_auc_score(y_admit_test, y_prob_disp)
        print(f"\n  ROC AUC: {auc_disp:.4f}")
    except Exception as e:
        print(f"\n  ROC AUC: could not compute ({e})")

    # --- Disposition accuracy by acuity level ---
    print(f"\n  Disposition Accuracy by True Acuity Level:")
    for esi in [1, 2, 3, 4, 5]:
        mask = y_acuity_test == esi
        if mask.sum() > 0:
            acc = accuracy_score(y_admit_test[mask], y_pred_disp[mask])
            n = mask.sum()
            admit_rate = y_admit_test[mask].mean()
            print(f"    ESI {esi}: accuracy={acc:.4f}  (n={n:,}, actual admit rate={admit_rate:.1%})")

    # ===================================================================
    #  FEATURE IMPORTANCE (Top 30)
    # ===================================================================
    print_section("TOP 30 FEATURE IMPORTANCES — ACUITY MODEL")

    feature_names = list(X_test.columns)
    importances = acuity_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    # Map tfidf_N back to actual words
    tfidf_vocab_inv = {v: k for k, v in tfidf.vocabulary_.items()}

    for rank, idx in enumerate(sorted_idx[:30], 1):
        name = feature_names[idx]
        if name.startswith("tfidf_"):
            tfidf_idx = int(name.split("_")[1])
            word = tfidf_vocab_inv.get(tfidf_idx, "?")
            display = f"{name} ({word})"
        else:
            display = name
        bar = "#" * max(1, int(40 * importances[idx] / importances[sorted_idx[0]]))
        print(f"  {rank:>2}. {display:<45s} {importances[idx]:.5f}  {bar}")

    # ===================================================================
    #  SUMMARY
    # ===================================================================
    print_section("BENCHMARK SUMMARY")
    print(f"  Test set size:              {len(y_acuity_test):,}")
    print(f"  ---")
    print(f"  Acuity exact accuracy:      {exact_acc:.4f}  ({exact_acc*100:.2f}%)")
    print(f"  Acuity within-1 accuracy:   {within_1:.4f}  ({within_1*100:.2f}%)")
    print(f"  Acuity MAE:                 {mae:.4f} ESI levels")
    print(f"  Acuity kappa (quadratic):   {kappa_quadratic:.4f}")
    print(f"  ---")
    print(f"  Disposition accuracy:       {disp_acc:.4f}  ({disp_acc*100:.2f}%)")
    try:
        print(f"  Disposition ROC AUC:        {auc_disp:.4f}")
    except:
        pass
    print(f"  ---")
    print(f"  Over-triage rate:           {100*over_triage/len(y_acuity_test):.1f}%")
    print(f"  Under-triage rate:          {100*under_triage/len(y_acuity_test):.1f}%")

    print("\n" + "#" * 70)
    print("  BENCHMARK COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
