"""
Realistic benchmark for triage v2 models.

Mirrors actual inference behavior:
  - Ambulance/helicopter patients: real vitals from triage.csv
  - Walk-in/other patients: vitals masked to missing (median-imputed, flags=1)

This reflects how the system actually runs: only ambulance/helicopter patients
have EMS vitals at triage. Walk-in patients haven't had vitals measured yet.
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
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent / "src" / "proiect_licenta"
MODELS_DIR = BASE_DIR / "models_v2"

# ---------------------------------------------------------------------------
# Import from v2 pipeline
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(BASE_DIR.parent))
from proiect_licenta.triage_pipeline_v2 import (
    load_and_clean_data, build_features,
    VITAL_COLS, ABNORMALITY_THRESHOLDS,
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


def mask_vitals_for_walkins(X: pd.DataFrame, test_df: pd.DataFrame,
                            vital_medians: dict) -> pd.DataFrame:
    """Mask vital sign features to 'missing' for non-ambulance/helicopter patients.

    For walk-in/other/unknown patients:
      - Raw vitals → set to median (as if imputed from missing)
      - Missing flags → set to 1
      - Abnormality flags → set to 0 (median values are normal)
      - abnormal_vital_count → set to 0
      - Vital interaction features → set to 0
    Ambulance/helicopter patients keep their real values.
    """
    X = X.copy()

    # Build mask: True for patients who should NOT have vitals
    arrival = test_df["arrival_transport"].reset_index(drop=True)
    walkin_mask = ~arrival.isin(["AMBULANCE", "HELICOPTER"])

    n_masked = walkin_mask.sum()
    n_total = len(walkin_mask)
    print(f"  Masking vitals for {n_masked:,}/{n_total:,} non-ambulance/helicopter patients "
          f"({100*n_masked/n_total:.1f}%)")

    # Raw vitals → median
    for col in VITAL_COLS:
        X.loc[walkin_mask, col] = vital_medians[col]

    # Missing flags → 1
    for col in VITAL_COLS:
        X.loc[walkin_mask, f"{col}_missing"] = 1

    # Abnormality flags → 0
    abnormality_flag_names = list(ABNORMALITY_THRESHOLDS.keys())
    for flag in abnormality_flag_names:
        X.loc[walkin_mask, flag] = 0

    # Abnormal count → 0
    X.loc[walkin_mask, "abnormal_vital_count"] = 0

    # Vital-transport interactions → 0 (walk-ins aren't ambulance anyway, but be explicit)
    for col in ["tachycardic_ambulance", "hypoxic_ambulance",
                "hypotensive_ambulance", "fever_ambulance"]:
        X.loc[walkin_mask, col] = 0

    # Vital-age interactions → 0
    for col in ["tachycardic_elderly", "hypoxic_elderly", "hypotensive_elderly"]:
        X.loc[walkin_mask, col] = 0

    return X


def main():
    print("\n" + "#" * 70)
    print("  TRIAGE v2 — REALISTIC BENCHMARK")
    print("  Vitals only for ambulance/helicopter; masked for walk-ins")
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
    print("  Loaded all artifacts")

    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    print(f"  Model version: {metadata['version']}")
    print(f"  Total features: {metadata['n_total_features']}")

    # Build test features (with real vitals for everyone)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        X_test, _, _, _ = build_features(
            test_df, tfidf=tfidf, severity_map=severity_map,
            vital_medians=vital_medians, fit=False,
        )

    y_acuity_test = test_df["acuity"].reset_index(drop=True)
    y_admit_test = test_df["admitted"].reset_index(drop=True)
    arrival_test = test_df["arrival_transport"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Mask vitals for walk-in patients
    # ------------------------------------------------------------------
    print_section("MASKING VITALS FOR WALK-IN PATIENTS")
    X_test_realistic = mask_vitals_for_walkins(X_test, test_df, vital_medians)

    # ===================================================================
    #  ACUITY MODEL — OVERALL
    # ===================================================================
    print_section("ACUITY MODEL — OVERALL (realistic inference)")

    y_pred_acuity = acuity_model.predict(X_test_realistic) + 1
    y_prob_acuity = acuity_model.predict_proba(X_test_realistic)

    exact_acc = accuracy_score(y_acuity_test, y_pred_acuity)
    within_1 = float(np.mean(np.abs(y_pred_acuity - y_acuity_test.values) <= 1))
    mae = np.mean(np.abs(y_pred_acuity - y_acuity_test.values))

    print(f"\n  Exact accuracy:           {exact_acc:.4f}  ({exact_acc*100:.2f}%)")
    print(f"  Within-1-level accuracy:  {within_1:.4f}  ({within_1*100:.2f}%)")
    print(f"  Mean Absolute Error:      {mae:.4f} ESI levels")

    print(f"\n  Per-Class Classification Report:")
    print(classification_report(
        y_acuity_test, y_pred_acuity,
        labels=[1, 2, 3, 4, 5],
        target_names=["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"],
        digits=4, zero_division=0,
    ))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_acuity_test, y_pred_acuity, labels=[1, 2, 3, 4, 5])
    print_confusion_matrix(cm, [1, 2, 3, 4, 5])

    errors = y_pred_acuity - y_acuity_test.values
    total_errors = (errors != 0).sum()
    over_triage = (errors < 0).sum()
    under_triage = (errors > 0).sum()
    print(f"\n  Over-triage rate:  {100*over_triage/len(y_acuity_test):.1f}%")
    print(f"  Under-triage rate: {100*under_triage/len(y_acuity_test):.1f}%")

    kappa_q = cohen_kappa_score(y_acuity_test, y_pred_acuity, weights="quadratic")
    print(f"  Cohen's kappa (quadratic): {kappa_q:.4f}")

    # ===================================================================
    #  ACUITY — BREAKDOWN BY ARRIVAL TRANSPORT
    # ===================================================================
    print_section("ACUITY — BY ARRIVAL TRANSPORT (ambulance/helicopter vs walk-in)")

    for group_name, group_mask in [
        ("AMBULANCE/HELICOPTER (real vitals)", arrival_test.isin(["AMBULANCE", "HELICOPTER"])),
        ("WALK-IN (vitals masked)", arrival_test == "WALK IN"),
        ("OTHER/UNKNOWN (vitals masked)", ~arrival_test.isin(["AMBULANCE", "HELICOPTER", "WALK IN"])),
    ]:
        n = group_mask.sum()
        if n == 0:
            continue
        y_true = y_acuity_test[group_mask]
        y_pred = y_pred_acuity[group_mask]

        acc = accuracy_score(y_true, y_pred)
        w1 = float(np.mean(np.abs(y_pred - y_true.values) <= 1))
        group_mae = np.mean(np.abs(y_pred - y_true.values))

        print(f"\n  {group_name} (n={n:,}):")
        print(f"    Exact accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    Within-1:        {w1:.4f}  ({w1*100:.2f}%)")
        print(f"    MAE:             {group_mae:.4f}")

        # Per-class recall for this group
        for esi in [1, 2, 3, 4, 5]:
            esi_mask = y_true == esi
            if esi_mask.sum() > 0:
                esi_acc = (y_pred[esi_mask] == esi).sum()
                esi_total = esi_mask.sum()
                print(f"    ESI {esi}: {esi_acc}/{esi_total} "
                      f"({100*esi_acc/esi_total:.1f}%)")

    # ===================================================================
    #  DISPOSITION MODEL
    # ===================================================================
    print_section("DISPOSITION MODEL — OVERALL (realistic inference)")

    X_test_disp = X_test_realistic.copy()
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

    try:
        auc_disp = roc_auc_score(y_admit_test, y_prob_disp)
        print(f"  ROC AUC: {auc_disp:.4f}")
    except Exception:
        auc_disp = None

    # Disposition by arrival transport
    print_section("DISPOSITION — BY ARRIVAL TRANSPORT")
    for group_name, group_mask in [
        ("AMBULANCE/HELICOPTER (real vitals)", arrival_test.isin(["AMBULANCE", "HELICOPTER"])),
        ("WALK-IN (vitals masked)", arrival_test == "WALK IN"),
    ]:
        n = group_mask.sum()
        if n == 0:
            continue
        y_true = y_admit_test[group_mask]
        y_pred = y_pred_disp[group_mask]
        acc = accuracy_score(y_true, y_pred)
        print(f"\n  {group_name} (n={n:,}):")
        print(f"    Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # ===================================================================
    #  SUMMARY
    # ===================================================================
    print_section("BENCHMARK SUMMARY — REALISTIC INFERENCE")
    print(f"  Test set size:              {len(y_acuity_test):,}")
    print(f"  Patients with real vitals:  {arrival_test.isin(['AMBULANCE', 'HELICOPTER']).sum():,} "
          f"({100*arrival_test.isin(['AMBULANCE', 'HELICOPTER']).sum()/len(arrival_test):.1f}%)")
    print(f"  Patients with masked vitals:{(~arrival_test.isin(['AMBULANCE', 'HELICOPTER'])).sum():,} "
          f"({100*(~arrival_test.isin(['AMBULANCE', 'HELICOPTER'])).sum()/len(arrival_test):.1f}%)")
    print(f"  ---")
    print(f"  Acuity exact accuracy:      {exact_acc:.4f}  ({exact_acc*100:.2f}%)")
    print(f"  Acuity within-1 accuracy:   {within_1:.4f}  ({within_1*100:.2f}%)")
    print(f"  Acuity MAE:                 {mae:.4f}")
    print(f"  Acuity kappa (quadratic):   {kappa_q:.4f}")
    print(f"  ---")
    print(f"  Disposition accuracy:       {disp_acc:.4f}  ({disp_acc*100:.2f}%)")
    if auc_disp:
        print(f"  Disposition ROC AUC:        {auc_disp:.4f}")
    print(f"  ---")
    print(f"  Over-triage rate:           {100*over_triage/len(y_acuity_test):.1f}%")
    print(f"  Under-triage rate:          {100*under_triage/len(y_acuity_test):.1f}%")

    print("\n" + "#" * 70)
    print("  REALISTIC BENCHMARK COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
