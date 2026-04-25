"""
Benchmark script for triage v2 models (with vital signs).

Reproduces the exact train/test split from training (random_state=42, stratified,
capped at 100K rows), loads saved v2 model artifacts, and evaluates with detailed metrics.
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
    roc_auc_score, cohen_kappa_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Make the proiect_licenta package importable from this script
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import TRIAGE_V2_DIR as MODELS_DIR
from proiect_licenta.training.train_triage_v2 import (
    load_and_clean_data, build_features, VITAL_COLS, ABNORMALITY_THRESHOLDS,
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_confusion_matrix(cm, labels):
    col_width = max(8, max(len(str(l)) for l in labels) + 2)
    header = " " * (col_width + 4) + "".join(f"{'Pred ' + str(l):>{col_width}}" for l in labels)
    print(header)
    print(" " * (col_width + 4) + "-" * (col_width * len(labels)))
    for i, label in enumerate(labels):
        row_vals = "".join(f"{cm[i][j]:>{col_width},}" for j in range(len(labels)))
        print(f"  {'True ' + str(label):>{col_width}} |{row_vals}")


def main():
    print("\n" + "#" * 70)
    print("  TRIAGE v2 MODEL BENCHMARK (with vital signs)")
    print("  Evaluating saved models on held-out test set (20%)")
    print("#" * 70)

    # ------------------------------------------------------------------
    # 1. Load data (same as training — includes 100K cap)
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
    # 3. Load saved artifacts
    # ------------------------------------------------------------------
    print_section("LOADING SAVED v2 ARTIFACTS")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
    vital_medians = joblib.load(MODELS_DIR / "vital_medians.joblib")
    acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
    disposition_model = joblib.load(MODELS_DIR / "disposition_model.joblib")
    print("  Loaded: tfidf, severity_map, vital_medians, acuity_model, disposition_model")

    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    print(f"  Model version: {metadata['version']}")
    print(f"  Trained at: {metadata['trained_at']}")
    print(f"  Total features: {metadata['n_total_features']}")

    # Build test features
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        X_test, _, _, _ = build_features(
            test_df, tfidf=tfidf, severity_map=severity_map,
            vital_medians=vital_medians, fit=False,
        )

    y_acuity_test = test_df["acuity"].reset_index(drop=True)
    y_admit_test = test_df["admitted"].reset_index(drop=True)
    arrival_test = test_df["arrival_transport"].reset_index(drop=True)

    # ===================================================================
    #  ACUITY MODEL BENCHMARK
    # ===================================================================
    print_section("ACUITY MODEL — ESI 1-5 PREDICTION")

    y_pred_acuity = acuity_model.predict(X_test) + 1
    y_prob_acuity = acuity_model.predict_proba(X_test)

    exact_acc = accuracy_score(y_acuity_test, y_pred_acuity)
    within_1 = float(np.mean(np.abs(y_pred_acuity - y_acuity_test.values) <= 1))
    within_2 = float(np.mean(np.abs(y_pred_acuity - y_acuity_test.values) <= 2))

    print(f"\n  Overall Metrics:")
    print(f"    Exact accuracy:           {exact_acc:.4f}  ({exact_acc*100:.2f}%)")
    print(f"    Within-1-level accuracy:  {within_1:.4f}  ({within_1*100:.2f}%)")
    print(f"    Within-2-level accuracy:  {within_2:.4f}  ({within_2*100:.2f}%)")

    mae = np.mean(np.abs(y_pred_acuity - y_acuity_test.values))
    print(f"    Mean Absolute Error:      {mae:.4f} ESI levels")

    print(f"\n  Per-Class Classification Report:")
    print(classification_report(
        y_acuity_test, y_pred_acuity,
        labels=[1, 2, 3, 4, 5],
        target_names=["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"],
        digits=4,
        zero_division=0,
    ))

    print(f"  Per-Class Accuracy:")
    for esi in [1, 2, 3, 4, 5]:
        mask = y_acuity_test == esi
        if mask.sum() > 0:
            class_acc = accuracy_score(y_acuity_test[mask], y_pred_acuity[mask])
            total = mask.sum()
            correct = (y_pred_acuity[mask] == esi).sum()
            print(f"    ESI {esi}: {class_acc:.4f}  ({correct:,}/{total:,})")

    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    cm_acuity = confusion_matrix(y_acuity_test, y_pred_acuity, labels=[1, 2, 3, 4, 5])
    print_confusion_matrix(cm_acuity, [1, 2, 3, 4, 5])

    errors = y_pred_acuity - y_acuity_test.values
    total_errors = (errors != 0).sum()
    over_triage = (errors < 0).sum()
    under_triage = (errors > 0).sum()
    print(f"\n  Error Analysis:")
    print(f"    Total misclassified:  {total_errors:,} / {len(y_acuity_test):,} "
          f"({100*total_errors/len(y_acuity_test):.1f}%)")
    print(f"    Over-triaged  (pred more severe):  {over_triage:,} "
          f"({100*over_triage/total_errors:.1f}% of errors)")
    print(f"    Under-triaged (pred less severe):   {under_triage:,} "
          f"({100*under_triage/total_errors:.1f}% of errors)")

    print(f"\n  Error Magnitude Distribution:")
    for delta in range(-4, 5):
        count = (errors == delta).sum()
        if count > 0:
            bar = "#" * max(1, int(50 * count / len(y_acuity_test)))
            label = ("CORRECT" if delta == 0
                     else f"{'over' if delta < 0 else 'under'}-triage by {abs(delta)}")
            print(f"    {delta:+d} ({label:25s}): {count:>6,} "
                  f"({100*count/len(y_acuity_test):5.1f}%) {bar}")

    kappa_linear = cohen_kappa_score(y_acuity_test, y_pred_acuity, weights="linear")
    kappa_quadratic = cohen_kappa_score(y_acuity_test, y_pred_acuity, weights="quadratic")
    print(f"\n  Cohen's Kappa:")
    print(f"    Linear:    {kappa_linear:.4f}")
    print(f"    Quadratic: {kappa_quadratic:.4f}")

    max_probs = np.max(y_prob_acuity, axis=1)
    correct_mask = y_pred_acuity == y_acuity_test.values
    print(f"\n  Confidence Analysis:")
    print(f"    Mean confidence (correct):   {max_probs[correct_mask].mean():.4f}")
    print(f"    Mean confidence (incorrect): {max_probs[~correct_mask].mean():.4f}")

    # ===================================================================
    #  DISPOSITION MODEL BENCHMARK
    # ===================================================================
    print_section("DISPOSITION MODEL — ADMITTED vs NOT ADMITTED")

    X_test_disp = X_test.copy()
    X_test_disp["predicted_acuity"] = y_pred_acuity

    y_pred_disp = disposition_model.predict(X_test_disp)
    y_prob_disp = disposition_model.predict_proba(X_test_disp)[:, 1]

    disp_acc = accuracy_score(y_admit_test, y_pred_disp)
    print(f"\n  Overall Accuracy: {disp_acc:.4f} ({disp_acc*100:.2f}%)")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_admit_test, y_pred_disp,
        target_names=["NOT ADMITTED", "ADMITTED"],
        digits=4,
    ))

    cm_disp = confusion_matrix(y_admit_test, y_pred_disp)
    print(f"  Confusion Matrix:")
    print_confusion_matrix(cm_disp, ["NOT ADM", "ADMITTED"])

    try:
        auc_disp = roc_auc_score(y_admit_test, y_prob_disp)
        print(f"\n  ROC AUC: {auc_disp:.4f}")
    except Exception as e:
        print(f"\n  ROC AUC: could not compute ({e})")
        auc_disp = None

    print(f"\n  Disposition Accuracy by True Acuity Level:")
    for esi in [1, 2, 3, 4, 5]:
        mask = y_acuity_test == esi
        if mask.sum() > 0:
            acc = accuracy_score(y_admit_test[mask], y_pred_disp[mask])
            n = mask.sum()
            admit_rate = y_admit_test[mask].mean()
            print(f"    ESI {esi}: accuracy={acc:.4f}  "
                  f"(n={n:,}, actual admit rate={admit_rate:.1%})")

    # ===================================================================
    #  ACUITY — BY ARRIVAL TRANSPORT
    # ===================================================================
    print_section("ACUITY — BY ARRIVAL TRANSPORT")

    for group_name, group_mask in [
        ("AMBULANCE/HELICOPTER (real vitals)", arrival_test.isin(["AMBULANCE", "HELICOPTER"])),
        ("WALK-IN (vitals masked at training)", arrival_test == "WALK IN"),
        ("OTHER/UNKNOWN (vitals masked at training)", ~arrival_test.isin(["AMBULANCE", "HELICOPTER", "WALK IN"])),
    ]:
        n = group_mask.sum()
        if n == 0:
            continue
        y_true_g = y_acuity_test[group_mask]
        y_pred_g = y_pred_acuity[group_mask]

        acc_g = accuracy_score(y_true_g, y_pred_g)
        w1_g = float(np.mean(np.abs(y_pred_g - y_true_g.values) <= 1))
        mae_g = np.mean(np.abs(y_pred_g - y_true_g.values))

        print(f"\n  {group_name} (n={n:,}):")
        print(f"    Exact accuracy:  {acc_g:.4f}  ({acc_g*100:.2f}%)")
        print(f"    Within-1:        {w1_g:.4f}  ({w1_g*100:.2f}%)")
        print(f"    MAE:             {mae_g:.4f}")

        for esi in [1, 2, 3, 4, 5]:
            esi_mask = y_true_g == esi
            if esi_mask.sum() > 0:
                esi_correct = (y_pred_g[esi_mask] == esi).sum()
                esi_total = esi_mask.sum()
                print(f"    ESI {esi}: {esi_correct}/{esi_total} "
                      f"({100*esi_correct/esi_total:.1f}%)")

    # ===================================================================
    #  FEATURE IMPORTANCE (Top 30)
    # ===================================================================
    print_section("TOP 30 FEATURE IMPORTANCES — ACUITY MODEL")

    feature_names = list(X_test.columns)
    importances = acuity_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

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
    print(f"  Total features:             {X_test.shape[1]}")
    print(f"  ---")
    print(f"  Acuity exact accuracy:      {exact_acc:.4f}  ({exact_acc*100:.2f}%)")
    print(f"  Acuity within-1 accuracy:   {within_1:.4f}  ({within_1*100:.2f}%)")
    print(f"  Acuity MAE:                 {mae:.4f} ESI levels")
    print(f"  Acuity kappa (quadratic):   {kappa_quadratic:.4f}")
    print(f"  ---")
    print(f"  Disposition accuracy:       {disp_acc:.4f}  ({disp_acc*100:.2f}%)")
    if auc_disp:
        print(f"  Disposition ROC AUC:        {auc_disp:.4f}")
    print(f"  ---")
    print(f"  Over-triage rate:           {100*over_triage/len(y_acuity_test):.1f}%")
    print(f"  Under-triage rate:          {100*under_triage/len(y_acuity_test):.1f}%")

    print("\n" + "#" * 70)
    print("  BENCHMARK COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
