"""Benchmark for the doctor disposition model.

Reproduces the training train/test split and evaluates the disposition model
against the triage v3 cascade baseline encoded in the soft-cascade features.
Reports accuracy/ROC AUC/Brier/ECE deltas, the confusion matrix and
over/under-triage at 0.5, calibration curve points, per-subgroup analysis
(elderly, polypharmacy, prior-ED-visit), and a feature-importance-by-group audit.
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import DOCTOR_V3_DIR
from proiect_licenta.training.train_doctor_disposition import (
    load_and_clean_data, build_features, SOFT_CASCADE_COLS,
)
from proiect_licenta.pmh_features import PMH_FEATURE_COLS
from proiect_licenta.training.train_nurse import MED_CATEGORY_KEYWORDS
from proiect_licenta.training.train_nurse_v3 import LONG_VITAL_FEATURE_COLS


# Reporting helpers
def print_section(title: str):
    print(f"  {title}")


def report_binary(name: str, y_true, y_proba, threshold: float = 0.5) -> dict:
    """Compact metric block for a binary admit/discharge prediction."""
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    over = fp / max(tn + fp, 1)
    under = fn / max(fn + tp, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    print(f"\n  [{name}] @ threshold={threshold:.2f}")
    print(f"    Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    ROC AUC         : {auc:.4f}")
    print(f"    Brier score     : {brier:.4f}  (lower is better)")
    print(f"    Sensitivity     : {sens:.4f}  (admit recall)")
    print(f"    Specificity     : {spec:.4f}  (discharge recall)")
    print(f"    Over-triage     : {over:.4f}  (discharge -> admit)")
    print(f"    Under-triage    : {under:.4f}  (admit -> discharge)")
    print(f"    Confusion matrix (rows=truth, cols=pred):")
    print(f"      truth=discharge: pred=disch {tn:>7,}  pred=admit {fp:>7,}")
    print(f"      truth=admit    : pred=disch {fn:>7,}  pred=admit {tp:>7,}")
    return {
        "accuracy": float(acc), "roc_auc": float(auc), "brier": float(brier),
        "sensitivity": float(sens), "specificity": float(spec),
        "over_triage": float(over), "under_triage": float(under),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_curve(y_true, y_proba, n_bins: int = 10):
    """Return (bin_centers, fraction_positive, mean_predicted) for plotting."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_proba, bins[1:-1])
    centers, fracs, preds, counts = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        centers.append((bins[b] + bins[b + 1]) / 2)
        fracs.append(float(y_true[mask].mean()))
        preds.append(float(y_proba[mask].mean()))
        counts.append(n)
    return centers, fracs, preds, counts


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_proba, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = y_proba[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_proba)) * abs(conf - acc)
    return float(ece)


# Main
def main():
    print("  DOCTOR DISPOSITION v3 - BENCHMARK")
    print("  Head-to-head vs triage v3 cascade baseline")

    # Load + rebuild the exact training pipeline
    df = load_and_clean_data()
    features = build_features(df)
    y = df["admitted"].astype(int).reset_index(drop=True)

    # Same split as training
    print_section("TRAIN/TEST SPLIT (80/20, stratified, random_state=42)")
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.20, random_state=42, stratify=y,
    )
    X_test = features.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(train_idx):,} | Test: {len(test_idx):,}")
    print(f"  Test admit rate: {y_test.mean():.4f}")

    # Load the trained doctor disposition model
    model_path = DOCTOR_V3_DIR / "disposition_model.joblib"
    raw_path = DOCTOR_V3_DIR / "disposition_model_raw.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Calibrated model not found at {model_path}. Run "
            f"`uv run train_doctor_disposition` first."
        )
    calibrated = joblib.load(model_path)
    raw = joblib.load(raw_path) if raw_path.exists() else None

    # Baseline 1: triage v3 cascade dispo probability
    print_section("BASELINE - TRIAGE v3 CASCADE DISPOSITION")
    triage_v1_proba = X_test["triage_disposition_proba_admit"].values.astype(float)
    base_metrics = report_binary("triage v3 cascade", y_test, triage_v1_proba)
    base_ece = expected_calibration_error(y_test.values, triage_v1_proba)
    print(f"    ECE (10 bins)   : {base_ece:.4f}")

    # Doctor disposition v3 (calibrated, deployment)
    print_section("DOCTOR DISPOSITION v3 - CALIBRATED (deployment model)")
    doctor_proba = calibrated.predict_proba(X_test)[:, 1]
    doc_metrics = report_binary("doctor dispo v3 (cal)", y_test, doctor_proba)
    doc_ece = expected_calibration_error(y_test.values, doctor_proba)
    print(f"    ECE (10 bins)   : {doc_ece:.4f}")

    if raw is not None:
        print_section("DOCTOR DISPOSITION v3 - UNCALIBRATED (audit only)")
        raw_proba = raw.predict_proba(X_test)[:, 1]
        raw_metrics = report_binary("doctor dispo v3 (raw)", y_test, raw_proba)
        raw_ece = expected_calibration_error(y_test.values, raw_proba)
        print(f"    ECE (10 bins)   : {raw_ece:.4f}")
        print(f"    [calibration delta: ECE {raw_ece:.4f} -> {doc_ece:.4f}  "
              f"(lower is better)]")

    # Headline delta
    print_section("HEADLINE DELTA - TRIAGE v3 CASCADE -> DOCTOR DISPOSITION v3")
    print(f"  Accuracy   :  {base_metrics['accuracy']:.4f} -> "
          f"{doc_metrics['accuracy']:.4f}   delta {doc_metrics['accuracy']-base_metrics['accuracy']:+.4f}")
    print(f"  ROC AUC    :  {base_metrics['roc_auc']:.4f} -> "
          f"{doc_metrics['roc_auc']:.4f}   delta {doc_metrics['roc_auc']-base_metrics['roc_auc']:+.4f}")
    print(f"  Brier      :  {base_metrics['brier']:.4f} -> "
          f"{doc_metrics['brier']:.4f}   delta {doc_metrics['brier']-base_metrics['brier']:+.4f}  (lower=better)")
    print(f"  ECE        :  {base_ece:.4f} -> {doc_ece:.4f}   "
          f"delta {doc_ece-base_ece:+.4f}  (lower=better)")
    print(f"  Sensitivity:  {base_metrics['sensitivity']:.4f} -> "
          f"{doc_metrics['sensitivity']:.4f}   delta {doc_metrics['sensitivity']-base_metrics['sensitivity']:+.4f}")
    print(f"  Specificity:  {base_metrics['specificity']:.4f} -> "
          f"{doc_metrics['specificity']:.4f}   delta {doc_metrics['specificity']-base_metrics['specificity']:+.4f}")
    print(f"  Under-tri  :  {base_metrics['under_triage']:.4f} -> "
          f"{doc_metrics['under_triage']:.4f}   delta {doc_metrics['under_triage']-base_metrics['under_triage']:+.4f}  (lower=better)")
    print(f"  Over-tri   :  {base_metrics['over_triage']:.4f} -> "
          f"{doc_metrics['over_triage']:.4f}   delta {doc_metrics['over_triage']-base_metrics['over_triage']:+.4f}  (lower=better)")

    # Calibration curve
    print_section("CALIBRATION CURVE (10 reliability bins, calibrated model)")
    centers, fracs, preds, counts = calibration_curve(y_test.values, doctor_proba)
    print(f"  {'bin center':>11s}  {'predicted mean':>15s}  {'observed admit':>15s}  {'n':>6s}")
    for c, p, f, n in zip(centers, preds, fracs, counts):
        bar_w = int(round(40 * f))
        bar = "#" * bar_w + "." * (40 - bar_w)
        print(f"  {c:>11.2f}  {p:>15.4f}  {f:>15.4f}  {n:>6,}  |{bar}|")

    # Per-subgroup analysis
    print_section("PER-SUBGROUP DELTAS  (plan section 3: lift concentrates in elderly / polypharmacy / repeat-visits)")
    subgroups = {
        "elderly (age >= 65)":
            (df_test["age"] >= 65),
        "polypharmacy (n_meds >= 5)":
            (df_test["n_medications"] >= 5),
        "abnormal vitals (any 1+ flag)":
            ((df_test["fever"] + df_test["tachycardia"] + df_test["tachypnea"]
              + df_test["hypoxia"] + df_test["hypotension"]) >= 1),
        "repeat visitor (prior ED visit)":
            (df_test["n_prior_ed_visits"] > 0)
            if "n_prior_ed_visits" in df_test.columns else None,
        "prior admission":
            (df_test["n_prior_admissions"] > 0)
            if "n_prior_admissions" in df_test.columns else None,
        "non-sinus rhythm":
            (df_test["rhythm_irregular"] == 1)
            if "rhythm_irregular" in df_test.columns else None,
    }
    print(f"  {'subgroup':40s}  {'n':>6s}  {'base acc':>9s}  {'doc acc':>9s}  {'delta':>8s}   "
          f"{'base AUC':>9s}  {'doc AUC':>9s}  {'delta':>8s}")
    for name, mask in subgroups.items():
        if mask is None:
            continue
        mask = mask.astype(bool)
        n = int(mask.sum())
        if n < 200:
            print(f"  {name:40s}  {n:>6,}  (subgroup too small for stable estimate)")
            continue
        y_sub = y_test.values[mask]
        if y_sub.sum() == 0 or y_sub.sum() == n:
            print(f"  {name:40s}  {n:>6,}  (single-class subgroup, skipping)")
            continue
        b_acc = accuracy_score(y_sub, (triage_v1_proba[mask] >= 0.5).astype(int))
        d_acc = accuracy_score(y_sub, (doctor_proba[mask] >= 0.5).astype(int))
        b_auc = roc_auc_score(y_sub, triage_v1_proba[mask])
        d_auc = roc_auc_score(y_sub, doctor_proba[mask])
        print(f"  {name:40s}  {n:>6,}  {b_acc:>9.4f}  {d_acc:>9.4f}  "
              f"{d_acc-b_acc:>+8.4f}   {b_auc:>9.4f}  {d_auc:>9.4f}  "
              f"{d_auc-b_auc:>+8.4f}")

    # Feature importance audit
    if raw is not None:
        print_section("FEATURE IMPORTANCE AUDIT (uncalibrated XGBoost, by gain)")
        booster = raw.get_booster()
        importance = booster.get_score(importance_type="gain")
        # XGBoost names features by their column name when passed a DataFrame.
        cols = list(X_test.columns)
        # Map: any column with non-zero importance
        items = []
        for name, gain in importance.items():
            if name in cols:
                items.append((name, gain))
            else:
                # XGBoost sometimes uses 'fN' instead of column names
                try:
                    idx = int(name[1:])
                    items.append((cols[idx], gain))
                except Exception:
                    items.append((name, gain))
        items.sort(key=lambda kv: kv[1], reverse=True)

        # Group classification
        def group_of(col: str) -> str:
            if col.startswith("tfidf_"):
                return "tfidf"
            if col in SOFT_CASCADE_COLS:
                return "soft_cascade"
            if col in PMH_FEATURE_COLS:
                return "pmh"
            if col in LONG_VITAL_FEATURE_COLS:
                return "longitudinal"
            if col in MED_CATEGORY_KEYWORDS or col in ("n_medications", "meds_unknown"):
                return "medications"
            if col in {"temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
                       "temperature_missing", "heartrate_missing", "resprate_missing",
                       "o2sat_missing", "sbp_missing", "dbp_missing",
                       "fever", "tachycardia", "bradycardia", "tachypnea",
                       "hypoxia", "hypertension", "hypotension", "map"}:
                return "snapshot_vitals"
            return "structured"

        # Group-level totals
        group_totals = {}
        for col, g in items:
            grp = group_of(col)
            group_totals[grp] = group_totals.get(grp, 0.0) + g
        total = sum(group_totals.values()) or 1.0
        print(f"\n  Gain by feature group:")
        for grp in sorted(group_totals, key=group_totals.get, reverse=True):
            print(f"    {grp:18s}  {group_totals[grp]:>10.1f}   "
                  f"({100*group_totals[grp]/total:5.1f}%)")

        print(f"\n  Top-25 columns by gain:")
        print(f"    {'rank':>4s}  {'gain':>10s}  {'group':18s}  column")
        for r, (col, gain) in enumerate(items[:25], start=1):
            print(f"    {r:>4d}  {gain:>10.1f}  {group_of(col):18s}  {col}")

        # PMH-specific check: how many of the 19 PMH columns earn non-zero gain
        pmh_with_gain = sum(1 for col, _ in items if col in PMH_FEATURE_COLS)
        print(f"\n  PMH wiring audit: {pmh_with_gain}/{len(PMH_FEATURE_COLS)} "
              f"PMH columns earned non-zero gain")
        # Soft cascade: should be high-importance
        cascade_with_gain = [(c, g) for c, g in items if c in SOFT_CASCADE_COLS]
        print(f"  Soft-cascade audit ({len(cascade_with_gain)}/6 with gain):")
        for c, g in cascade_with_gain:
            print(f"    {c:38s}  gain={g:.1f}")

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
