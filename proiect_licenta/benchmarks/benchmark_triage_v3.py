"""
Benchmark script for triage v3 models (v2 features + PMH).

Reproduces the exact train/test split from training (random_state=42, stratified)
and evaluates v3 against the same v2 baseline on the same held-out rows so the
PMH delta can be read cleanly.
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import (
    TRIAGE_V2_DIR, TRIAGE_V3_DIR,
)
from proiect_licenta.training.train_triage_v3 import (
    load_and_clean_data as load_v3_data,
    build_features as build_v3_features,
)
from proiect_licenta.training.train_triage_v2 import (
    build_features as build_v2_features,
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_confusion_matrix(cm, labels):
    col_width = max(8, max(len(str(l)) for l in labels) + 2)
    header = " " * (col_width + 4) + "".join(
        f"{'Pred ' + str(l):>{col_width}}" for l in labels
    )
    print(header)
    print(" " * (col_width + 4) + "-" * (col_width * len(labels)))
    for i, label in enumerate(labels):
        row_vals = "".join(f"{cm[i][j]:>{col_width},}" for j in range(len(labels)))
        print(f"  {'True ' + str(label):>{col_width}} |{row_vals}")


def evaluate_acuity(name, y_true, y_pred, y_prob=None):
    """Compact metric block for one (acuity) model."""
    exact = accuracy_score(y_true, y_pred)
    w1 = float(np.mean(np.abs(y_pred - y_true.values) <= 1))
    w2 = float(np.mean(np.abs(y_pred - y_true.values) <= 2))
    mae = float(np.mean(np.abs(y_pred - y_true.values)))
    kappa_q = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    kappa_l = cohen_kappa_score(y_true, y_pred, weights="linear")
    errors = y_pred - y_true.values
    over = int((errors < 0).sum())
    under = int((errors > 0).sum())
    print(f"\n  [{name}] Overall:")
    print(f"    Exact accuracy:           {exact:.4f}  ({exact*100:.2f}%)")
    print(f"    Within-1-level accuracy:  {w1:.4f}  ({w1*100:.2f}%)")
    print(f"    Within-2-level accuracy:  {w2:.4f}  ({w2*100:.2f}%)")
    print(f"    Mean Absolute Error:      {mae:.4f}")
    print(f"    Cohen's κ (linear):       {kappa_l:.4f}")
    print(f"    Cohen's κ (quadratic):    {kappa_q:.4f}")
    print(f"    Over-triage rate:         {100*over/len(y_true):.2f}%")
    print(f"    Under-triage rate:        {100*under/len(y_true):.2f}%")
    return {
        "exact": exact, "within_1": w1, "within_2": w2, "mae": mae,
        "kappa_linear": kappa_l, "kappa_quadratic": kappa_q,
        "over_rate": over / len(y_true), "under_rate": under / len(y_true),
    }


def main():
    print("\n" + "#" * 70)
    print("  TRIAGE v3 MODEL BENCHMARK (v2 features + PMH)")
    print("  Head-to-head vs v2 on the same held-out test rows")
    print("#" * 70)

    # ------------------------------------------------------------------
    # 1. Load v3 data (which already includes the PMH aggregation step)
    # ------------------------------------------------------------------
    df, _edstays = load_v3_data()

    # ------------------------------------------------------------------
    # 2. Reproduce the same train/test split used at training
    # ------------------------------------------------------------------
    print_section("TRAIN/TEST SPLIT (80/20, stratified, random_state=42)")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["acuity"],
    )
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    y_acuity_test = test_df["acuity"].reset_index(drop=True)
    y_admit_test = test_df["admitted"].reset_index(drop=True)
    arrival_test = test_df["arrival_transport"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Load v3 artifacts and build v3 features on the test rows
    # ------------------------------------------------------------------
    print_section("LOADING v3 ARTIFACTS")
    v3_tfidf = joblib.load(TRIAGE_V3_DIR / "tfidf_vectorizer.joblib")
    v3_severity = joblib.load(TRIAGE_V3_DIR / "severity_map.joblib")
    v3_vital_medians = joblib.load(TRIAGE_V3_DIR / "vital_medians.joblib")
    v3_acuity = joblib.load(TRIAGE_V3_DIR / "acuity_model.joblib")
    v3_disp = joblib.load(TRIAGE_V3_DIR / "disposition_model.joblib")
    v3_meta = json.loads((TRIAGE_V3_DIR / "model_metadata.json").read_text())
    print(f"  v3 version:  {v3_meta['version']}")
    print(f"  v3 trained:  {v3_meta['trained_at']}")
    print(f"  v3 features: {v3_meta['n_total_features']}")

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        X_test_v3, _, _, _ = build_v3_features(
            test_df, tfidf=v3_tfidf, severity_map=v3_severity,
            vital_medians=v3_vital_medians, fit=False,
        )

    y_pred_v3_acuity = v3_acuity.predict(X_test_v3) + 1

    # Disposition cascades on predicted_acuity
    X_test_v3_disp = X_test_v3.copy()
    X_test_v3_disp["predicted_acuity"] = y_pred_v3_acuity
    y_pred_v3_disp = v3_disp.predict(X_test_v3_disp)
    y_prob_v3_disp = v3_disp.predict_proba(X_test_v3_disp)[:, 1]

    # ------------------------------------------------------------------
    # 4. Load v2 artifacts and predict on the same test rows
    # ------------------------------------------------------------------
    print_section("LOADING v2 ARTIFACTS (baseline)")
    v2_tfidf = joblib.load(TRIAGE_V2_DIR / "tfidf_vectorizer.joblib")
    v2_severity = joblib.load(TRIAGE_V2_DIR / "severity_map.joblib")
    v2_vital_medians = joblib.load(TRIAGE_V2_DIR / "vital_medians.joblib")
    v2_acuity = joblib.load(TRIAGE_V2_DIR / "acuity_model.joblib")
    v2_disp = joblib.load(TRIAGE_V2_DIR / "disposition_model.joblib")
    v2_meta = json.loads((TRIAGE_V2_DIR / "model_metadata.json").read_text())
    print(f"  v2 version:  {v2_meta['version']}")
    print(f"  v2 trained:  {v2_meta['trained_at']}")
    print(f"  v2 features: {v2_meta['n_total_features']}")

    with contextlib.redirect_stdout(io.StringIO()):
        X_test_v2, _, _, _ = build_v2_features(
            test_df, tfidf=v2_tfidf, severity_map=v2_severity,
            vital_medians=v2_vital_medians, fit=False,
        )

    y_pred_v2_acuity = v2_acuity.predict(X_test_v2) + 1
    X_test_v2_disp = X_test_v2.copy()
    X_test_v2_disp["predicted_acuity"] = y_pred_v2_acuity
    y_pred_v2_disp = v2_disp.predict(X_test_v2_disp)
    y_prob_v2_disp = v2_disp.predict_proba(X_test_v2_disp)[:, 1]

    # ===================================================================
    #  ACUITY HEAD-TO-HEAD (v2 vs v3 on identical test rows)
    # ===================================================================
    print_section("ACUITY — HEAD-TO-HEAD v2 vs v3")

    v2_metrics = evaluate_acuity("v2", y_acuity_test, y_pred_v2_acuity)
    v3_metrics = evaluate_acuity("v3", y_acuity_test, y_pred_v3_acuity)

    print("\n  Δ v3 − v2:")
    print(f"    Exact accuracy:           "
          f"{(v3_metrics['exact'] - v2_metrics['exact'])*100:+.2f}pp")
    print(f"    Within-1-level accuracy:  "
          f"{(v3_metrics['within_1'] - v2_metrics['within_1'])*100:+.2f}pp")
    print(f"    MAE:                      "
          f"{v3_metrics['mae'] - v2_metrics['mae']:+.4f}")
    print(f"    Cohen's κ (quadratic):    "
          f"{v3_metrics['kappa_quadratic'] - v2_metrics['kappa_quadratic']:+.4f}")
    print(f"    Under-triage rate:        "
          f"{(v3_metrics['under_rate'] - v2_metrics['under_rate'])*100:+.2f}pp")

    print(f"\n  v3 per-class report:")
    print(classification_report(
        y_acuity_test, y_pred_v3_acuity,
        labels=[1, 2, 3, 4, 5],
        target_names=["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"],
        digits=4, zero_division=0,
    ))
    print(f"\n  v3 confusion matrix (rows=True, cols=Predicted):")
    cm_v3 = confusion_matrix(y_acuity_test, y_pred_v3_acuity, labels=[1, 2, 3, 4, 5])
    print_confusion_matrix(cm_v3, [1, 2, 3, 4, 5])

    # ===================================================================
    #  DISPOSITION HEAD-TO-HEAD
    # ===================================================================
    print_section("DISPOSITION — HEAD-TO-HEAD v2 vs v3")

    v2_disp_acc = accuracy_score(y_admit_test, y_pred_v2_disp)
    v3_disp_acc = accuracy_score(y_admit_test, y_pred_v3_disp)
    try:
        v2_auc = roc_auc_score(y_admit_test, y_prob_v2_disp)
        v3_auc = roc_auc_score(y_admit_test, y_prob_v3_disp)
    except Exception:
        v2_auc = v3_auc = None

    print(f"\n  Accuracy:  v2 {v2_disp_acc:.4f}  →  v3 {v3_disp_acc:.4f}  "
          f"(Δ {(v3_disp_acc - v2_disp_acc)*100:+.2f}pp)")
    if v2_auc is not None:
        print(f"  ROC AUC:   v2 {v2_auc:.4f}  →  v3 {v3_auc:.4f}  "
              f"(Δ {v3_auc - v2_auc:+.4f})")

    print(f"\n  v3 classification report:")
    print(classification_report(
        y_admit_test, y_pred_v3_disp,
        target_names=["NOT ADMITTED", "ADMITTED"], digits=4,
    ))
    cm_disp_v3 = confusion_matrix(y_admit_test, y_pred_v3_disp)
    print(f"  v3 confusion matrix:")
    print_confusion_matrix(cm_disp_v3, ["NOT ADM", "ADMITTED"])

    print(f"\n  v3 disposition accuracy by true acuity:")
    for esi in [1, 2, 3, 4, 5]:
        mask = y_acuity_test == esi
        if mask.sum() > 0:
            acc = accuracy_score(y_admit_test[mask], y_pred_v3_disp[mask])
            n = int(mask.sum())
            admit_rate = float(y_admit_test[mask].mean())
            print(f"    ESI {esi}: accuracy={acc:.4f}  "
                  f"(n={n:,}, actual admit rate={admit_rate:.1%})")

    # ===================================================================
    #  ACUITY BY ARRIVAL TRANSPORT (v3 only — confirms walk-in lift)
    # ===================================================================
    print_section("v3 ACUITY — BY ARRIVAL TRANSPORT")

    for group_name, group_mask in [
        ("AMBULANCE/HELICOPTER (real vitals)",
         arrival_test.isin(["AMBULANCE", "HELICOPTER"])),
        ("WALK-IN (vitals masked at training)",
         arrival_test == "WALK IN"),
        ("OTHER/UNKNOWN (vitals masked at training)",
         ~arrival_test.isin(["AMBULANCE", "HELICOPTER", "WALK IN"])),
    ]:
        n = int(group_mask.sum())
        if n == 0:
            continue
        y_true_g = y_acuity_test[group_mask]
        y_pred_g_v2 = y_pred_v2_acuity[group_mask]
        y_pred_g_v3 = y_pred_v3_acuity[group_mask]

        acc_v2 = accuracy_score(y_true_g, y_pred_g_v2)
        acc_v3 = accuracy_score(y_true_g, y_pred_g_v3)
        w1_v3 = float(np.mean(np.abs(y_pred_g_v3 - y_true_g.values) <= 1))
        mae_v3 = float(np.mean(np.abs(y_pred_g_v3 - y_true_g.values)))

        print(f"\n  {group_name} (n={n:,}):")
        print(f"    v2 exact:        {acc_v2:.4f}  ({acc_v2*100:.2f}%)")
        print(f"    v3 exact:        {acc_v3:.4f}  ({acc_v3*100:.2f}%)  "
              f"(Δ {(acc_v3-acc_v2)*100:+.2f}pp)")
        print(f"    v3 within-1:     {w1_v3:.4f}  ({w1_v3*100:.2f}%)")
        print(f"    v3 MAE:          {mae_v3:.4f}")

    # ===================================================================
    #  PMH-FEATURE COVERAGE ON THE TEST SET
    # ===================================================================
    print_section("PMH FEATURE COVERAGE — TEST SET")

    no_hist = (test_df["no_history"] == 1).mean()
    any_pmh = (test_df[[c for c in test_df.columns
                        if c.startswith("pmh_")]].sum(axis=1) > 0).mean()
    print(f"  Test rows with no_history=1 (first-time MIMIC patients): "
          f"{100*no_hist:.1f}%")
    print(f"  Test rows with ≥1 PMH flag set:                          "
          f"{100*any_pmh:.1f}%")
    print(f"  Mean n_prior_admissions:  {test_df['n_prior_admissions'].mean():.2f}")
    print(f"  Mean n_prior_ed_visits:   {test_df['n_prior_ed_visits'].mean():.2f}")

    # ===================================================================
    #  FEATURE IMPORTANCE — verify PMH features are non-zero on v3 acuity
    # ===================================================================
    print_section("v3 ACUITY — TOP 30 FEATURE IMPORTANCES + PMH AUDIT")

    feature_names = list(X_test_v3.columns)
    importances = v3_acuity.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    tfidf_vocab_inv = {v: k for k, v in v3_tfidf.vocabulary_.items()}

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

    pmh_feature_idxs = [i for i, n in enumerate(feature_names)
                        if n.startswith("pmh_") or n in {
                            "n_prior_admissions", "n_prior_ed_visits",
                            "days_since_last_admission", "days_since_last_ed",
                            "same_complaint_as_prior", "no_history",
                        }]
    pmh_nonzero = sum(1 for i in pmh_feature_idxs if importances[i] > 0)
    print(f"\n  PMH features with non-zero importance: "
          f"{pmh_nonzero}/{len(pmh_feature_idxs)}")
    pmh_in_top50 = sum(1 for i in sorted_idx[:50] if i in pmh_feature_idxs)
    print(f"  PMH features in top 50: {pmh_in_top50}")

    # ===================================================================
    #  RED-FLAG FEATURE AUDIT (section 1.5)
    # ===================================================================
    print_section("v3 ACUITY — RED-FLAG FEATURE AUDIT (section 1.5)")

    rf_feature_idxs = [i for i, n in enumerate(feature_names)
                       if n.startswith("rf_")]
    rf_nonzero = sum(1 for i in rf_feature_idxs if importances[i] > 0)
    rf_in_top50 = sum(1 for i in sorted_idx[:50] if i in rf_feature_idxs)
    rf_in_top100 = sum(1 for i in sorted_idx[:100] if i in rf_feature_idxs)
    print(f"  Red-flag features in model:                    {len(rf_feature_idxs)}")
    print(f"  Red-flag features with non-zero gain (acuity): "
          f"{rf_nonzero}/{len(rf_feature_idxs)}")
    print(f"  Red-flag features in top 50 by gain (acuity):  {rf_in_top50}")
    print(f"  Red-flag features in top 100 by gain (acuity): {rf_in_top100}")

    # Coverage on the test set — how often does any red flag fire?
    rf_cols_in_test = [c for c in test_df.columns if c.startswith("rf_")] \
        if any(c.startswith("rf_") for c in test_df.columns) else []
    # rf_* columns aren't on test_df directly (red-flag features are built
    # inside build_features), so compute coverage off X_test_v3 instead.
    rf_any_col = "rf_any"
    if rf_any_col in X_test_v3.columns:
        rf_any_rate = float((X_test_v3[rf_any_col] == 1).mean())
        rf_any_count_mean = float(X_test_v3["rf_any_count"].mean())
        print(f"  rf_any positive rate on test set:              {100*rf_any_rate:.2f}%")
        print(f"  Mean rf_any_count per row:                     {rf_any_count_mean:.3f}")

    # Top-N red flags by gain on the ACUITY model (the head that ESI
    # extreme-class weighting should make most receptive to this signal).
    rf_imps = sorted(
        [(feature_names[i], float(importances[i])) for i in rf_feature_idxs],
        key=lambda kv: kv[1], reverse=True,
    )
    print(f"\n  Top 10 red flags by gain (acuity):")
    for rank, (name, gain) in enumerate(rf_imps[:10], 1):
        # Overall feature rank for context.
        overall_rank = int(np.where(sorted_idx == feature_names.index(name))[0][0]) + 1
        print(f"    {rank:>2}. rank={overall_rank:<4d}  {name:<28s}  gain={gain:.5f}")

    # Same audit on the DISPOSITION head — chronic-condition-like red flags
    # (anaphylaxis, sepsis, GI bleed) may matter more for admit/discharge.
    print(f"\n  Disposition head:")
    disp_feature_names = list(X_test_v3_disp.columns)
    disp_imps_arr = v3_disp.feature_importances_
    disp_rf_idxs = [i for i, n in enumerate(disp_feature_names)
                    if n.startswith("rf_")]
    disp_rf_nonzero = sum(1 for i in disp_rf_idxs if disp_imps_arr[i] > 0)
    disp_sorted = np.argsort(disp_imps_arr)[::-1]
    disp_rf_top50 = sum(1 for i in disp_sorted[:50] if i in disp_rf_idxs)
    print(f"    Red-flag features with non-zero gain: "
          f"{disp_rf_nonzero}/{len(disp_rf_idxs)}")
    print(f"    Red-flag features in top 50 by gain:  {disp_rf_top50}")
    disp_rf_imps = sorted(
        [(disp_feature_names[i], float(disp_imps_arr[i])) for i in disp_rf_idxs],
        key=lambda kv: kv[1], reverse=True,
    )
    print(f"\n    Top 5 red flags by gain (disposition):")
    for rank, (name, gain) in enumerate(disp_rf_imps[:5], 1):
        overall_rank = int(np.where(disp_sorted == disp_feature_names.index(name))[0][0]) + 1
        print(f"      {rank:>2}. rank={overall_rank:<4d}  {name:<28s}  gain={gain:.5f}")

    # ===================================================================
    #  SUMMARY
    # ===================================================================
    print_section("BENCHMARK SUMMARY")
    print(f"  Test set size:                {len(y_acuity_test):,}")
    print(f"  v3 total features:            {X_test_v3.shape[1]}")
    print(f"  ---")
    print(f"  ACUITY")
    print(f"    v2 exact accuracy:          {v2_metrics['exact']:.4f}")
    print(f"    v3 exact accuracy:          {v3_metrics['exact']:.4f}  "
          f"(Δ {(v3_metrics['exact'] - v2_metrics['exact'])*100:+.2f}pp)")
    print(f"    v2 within-1-level:          {v2_metrics['within_1']:.4f}")
    print(f"    v3 within-1-level:          {v3_metrics['within_1']:.4f}  "
          f"(Δ {(v3_metrics['within_1'] - v2_metrics['within_1'])*100:+.2f}pp)")
    print(f"    v2 quadratic κ:             {v2_metrics['kappa_quadratic']:.4f}")
    print(f"    v3 quadratic κ:             {v3_metrics['kappa_quadratic']:.4f}  "
          f"(Δ {v3_metrics['kappa_quadratic'] - v2_metrics['kappa_quadratic']:+.4f})")
    print(f"  ---")
    print(f"  DISPOSITION")
    print(f"    v2 accuracy:                {v2_disp_acc:.4f}")
    print(f"    v3 accuracy:                {v3_disp_acc:.4f}  "
          f"(Δ {(v3_disp_acc - v2_disp_acc)*100:+.2f}pp)")
    if v2_auc is not None:
        print(f"    v2 ROC AUC:                 {v2_auc:.4f}")
        print(f"    v3 ROC AUC:                 {v3_auc:.4f}  "
              f"(Δ {v3_auc - v2_auc:+.4f})")

    print("\n" + "#" * 70)
    print("  BENCHMARK COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
