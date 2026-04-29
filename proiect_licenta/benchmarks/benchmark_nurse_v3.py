"""
Benchmark: Doctor v3 base vs v3 with-nurse comparison

Evaluates both v3 models on the same held-out test data to quantify the
impact of nurse-collected vital signs and medication features ON THE NEW
13-class label space (catch-all class excluded, full filtered dataset).

  v3 base       : complaints + demographics + triage predictions (no nurse)
  v3 with nurse : v3 base + snapshot vitals + medications (Phase A);
                  Phase B will add longitudinal vitals + rhythm here.

This is the v3 analogue of benchmark_nurse.py (which compares v1 vs v2).
The two scripts are kept side-by-side so the thesis can show v1-vs-v2
(14 classes) AND v3-base-vs-v3-with-nurse (13 classes) tables.
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import (
    TRIAGE_V1_DIR as MODELS_DIR,
    DOCTOR_V3_BASE_DIR, DOCTOR_V3_DIR,
)
from proiect_licenta.training.train_doctor import (
    DEPARTMENT_NAMES, CATCH_ALL_LABEL,
)
from proiect_licenta.training.train_nurse_v3 import (
    load_and_clean_data as load_v3_data,
    build_features as build_v3_with_nurse_features,
)
from proiect_licenta.training.train_doctor_v3 import (
    build_features as build_v3_base_features,
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print("\n" + "#" * 70)
    print("  DOCTOR v3 BASE vs v3 WITH-NURSE BENCHMARK COMPARISON")
    print(f"  Catch-all '{CATCH_ALL_LABEL}' EXCLUDED — 13-class label space")
    print("  Impact of nurse-collected vitals + medications on the cleaner labels")
    print("#" * 70)

    # ------------------------------------------------------------------
    # 1. Load data (v3 with-nurse loader is the superset; both v3 base
    #    and v3 with-nurse use the same row filter so this is a single
    #    source of truth for the test set).
    # ------------------------------------------------------------------
    df = load_v3_data()

    # ------------------------------------------------------------------
    # 2. Build features for both versions (NO sub-sampling for v3)
    # ------------------------------------------------------------------
    print_section("BUILDING FEATURES (full filtered dataset, no sub-sample)")
    print(f"  Full filtered dataset: {len(df):,}")

    print("\n  Building v3 base features (no nurse data)...")
    features_v3_base = build_v3_base_features(df)
    print(f"  v3 base features: {features_v3_base.shape[1]}")

    print("\n  Building v3 with-nurse features (vitals + meds)...")
    features_v3_nurse = build_v3_with_nurse_features(df)
    print(f"  v3 with-nurse features: {features_v3_nurse.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Encode labels + split (same seed for both)
    # ------------------------------------------------------------------
    meta_base = json.loads((DOCTOR_V3_BASE_DIR / "metadata.json").read_text())
    meta_nurse = json.loads((DOCTOR_V3_DIR / "metadata.json").read_text())

    diagnosis_labels = meta_base["diagnosis_labels"]
    department_labels = meta_base["department_labels"]

    # Both v3 metadata files must agree on the label space.
    assert meta_nurse["diagnosis_labels"] == diagnosis_labels, (
        "v3 base and v3 with-nurse have different diagnosis label spaces"
    )
    assert CATCH_ALL_LABEL not in diagnosis_labels, (
        f"v3 metadata still contains catch-all '{CATCH_ALL_LABEL}' — bug"
    )

    diag_map = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_map = {l: i for i, l in enumerate(department_labels)}

    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)
    y_dept = df["service_group"].map(dept_map).reset_index(drop=True)

    # v3 base split
    Xb_train, Xb_test, y_diag_train, y_diag_test, y_dept_train, y_dept_test = \
        train_test_split(features_v3_base, y_diag, y_dept,
                         test_size=0.2, random_state=42, stratify=y_diag)

    # v3 with-nurse split (same indices because same seed + same y_diag stratify)
    Xn_train, Xn_test, _, _, _, _ = \
        train_test_split(features_v3_nurse, y_diag, y_dept,
                         test_size=0.2, random_state=42, stratify=y_diag)

    print(f"  Train: {len(Xb_train):,} | Test: {len(Xb_test):,}")

    # ------------------------------------------------------------------
    # 4. Load models
    # ------------------------------------------------------------------
    print_section("LOADING MODELS")
    base_diag = joblib.load(DOCTOR_V3_BASE_DIR / "diagnosis_model.joblib")
    base_dept = joblib.load(DOCTOR_V3_BASE_DIR / "department_model.joblib")
    nurse_diag = joblib.load(DOCTOR_V3_DIR / "diagnosis_model.joblib")
    nurse_dept = joblib.load(DOCTOR_V3_DIR / "department_model.joblib")
    print(f"  v3 base trained:       {meta_base['trained_at']}")
    print(f"  v3 with-nurse trained: {meta_nurse['trained_at']}")

    # ------------------------------------------------------------------
    # 5. Evaluate DIAGNOSIS models
    # ------------------------------------------------------------------
    print_section(f"DIAGNOSIS CATEGORY — v3 base vs v3 with-nurse "
                  f"({len(diagnosis_labels)} classes)")

    yb_pred = base_diag.predict(Xb_test)
    yb_prob = base_diag.predict_proba(Xb_test)
    yn_pred = nurse_diag.predict(Xn_test)
    yn_prob = nurse_diag.predict_proba(Xn_test)

    metrics_base = {
        "accuracy": accuracy_score(y_diag_test, yb_pred),
        "top3": top_k_accuracy_score(y_diag_test, yb_prob, k=3),
        "top5": top_k_accuracy_score(y_diag_test, yb_prob, k=min(5, len(diagnosis_labels))),
        "kappa": cohen_kappa_score(y_diag_test, yb_pred),
    }
    metrics_nurse = {
        "accuracy": accuracy_score(y_diag_test, yn_pred),
        "top3": top_k_accuracy_score(y_diag_test, yn_prob, k=3),
        "top5": top_k_accuracy_score(y_diag_test, yn_prob, k=min(5, len(diagnosis_labels))),
        "kappa": cohen_kappa_score(y_diag_test, yn_pred),
    }

    print(f"\n  {'Metric':<30s}  {'base':>10s}  {'nurse':>10s}  {'Delta':>10s}")
    print(f"  {'-'*65}")
    for metric, label in [("accuracy", "Top-1 Accuracy"), ("top3", "Top-3 Accuracy"),
                          ("top5", "Top-5 Accuracy"), ("kappa", "Cohen's Kappa")]:
        v_b = metrics_base[metric]
        v_n = metrics_nurse[metric]
        delta = v_n - v_b
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<30s}  {v_b:>9.4f}  {v_n:>9.4f}  {sign}{delta:>9.4f}")

    # Per-class
    print(f"\n  Per-Class Accuracy Comparison (Diagnosis):")
    print(f"  {'Category':<40s}  {'base':>8s}  {'nurse':>8s}  {'Delta':>8s}  {'n':>6s}")
    print(f"  {'-'*75}")
    for idx, label in enumerate(diagnosis_labels):
        mask = y_diag_test == idx
        if mask.sum() > 0:
            a_b = accuracy_score(y_diag_test[mask], yb_pred[mask])
            a_n = accuracy_score(y_diag_test[mask], yn_pred[mask])
            delta = a_n - a_b
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<40s}  {a_b:>7.3f}  {a_n:>7.3f}  {sign}{delta:>7.3f}  {mask.sum():>6,}")

    print(f"\n  v3 with-nurse Classification Report (Diagnosis):")
    short_diag = [l[:30] for l in diagnosis_labels]
    print(classification_report(
        y_diag_test, yn_pred,
        target_names=short_diag, digits=4, zero_division=0,
    ))

    # ------------------------------------------------------------------
    # 6. Evaluate DEPARTMENT models
    # ------------------------------------------------------------------
    print_section(f"DEPARTMENT — v3 base vs v3 with-nurse "
                  f"({len(department_labels)} classes)")

    Xb_test_dept = Xb_test.copy()
    Xb_test_dept["predicted_diagnosis"] = base_diag.predict(Xb_test)

    Xn_test_dept = Xn_test.copy()
    Xn_test_dept["predicted_diagnosis"] = nurse_diag.predict(Xn_test)

    db_pred = base_dept.predict(Xb_test_dept)
    db_prob = base_dept.predict_proba(Xb_test_dept)
    dn_pred = nurse_dept.predict(Xn_test_dept)
    dn_prob = nurse_dept.predict_proba(Xn_test_dept)

    db_acc = accuracy_score(y_dept_test, db_pred)
    db_top3 = top_k_accuracy_score(y_dept_test, db_prob, k=3)
    db_kappa = cohen_kappa_score(y_dept_test, db_pred)
    dn_acc = accuracy_score(y_dept_test, dn_pred)
    dn_top3 = top_k_accuracy_score(y_dept_test, dn_prob, k=3)
    dn_kappa = cohen_kappa_score(y_dept_test, dn_pred)

    print(f"\n  {'Metric':<30s}  {'base':>10s}  {'nurse':>10s}  {'Delta':>10s}")
    print(f"  {'-'*65}")
    for label, v_b, v_n in [
        ("Top-1 Accuracy", db_acc, dn_acc),
        ("Top-3 Accuracy", db_top3, dn_top3),
        ("Cohen's Kappa", db_kappa, dn_kappa),
    ]:
        delta = v_n - v_b
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<30s}  {v_b:>9.4f}  {v_n:>9.4f}  {sign}{delta:>9.4f}")

    # Per-class
    print(f"\n  Per-Class Accuracy Comparison (Department):")
    print(f"  {'Department':<20s}  {'Full Name':<35s}  {'base':>7s}  {'nurse':>7s}  {'Delta':>7s}  {'n':>6s}")
    print(f"  {'-'*90}")
    for idx, label in enumerate(department_labels):
        mask = y_dept_test == idx
        if mask.sum() > 0:
            a_b = accuracy_score(y_dept_test[mask], db_pred[mask])
            a_n = accuracy_score(y_dept_test[mask], dn_pred[mask])
            delta = a_n - a_b
            sign = "+" if delta >= 0 else ""
            full = DEPARTMENT_NAMES.get(label, label)[:35]
            print(f"  {label:<20s}  {full:<35s}  {a_b:>6.3f}  {a_n:>6.3f}  {sign}{delta:>6.3f}  {mask.sum():>6,}")

    # ------------------------------------------------------------------
    # 7. Feature importance — v3 with-nurse top features
    # ------------------------------------------------------------------
    print_section("TOP 20 FEATURE IMPORTANCES — v3 WITH-NURSE DIAGNOSIS MODEL")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_vocab_inv = {v: k for k, v in tfidf.vocabulary_.items()}
    feature_names = list(Xn_test.columns)
    importances = nurse_diag.feature_importances_
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

    # Highlight nurse-specific features in top 50
    print(f"\n  Nurse-specific features in top 50:")
    nurse_features = {
        # Snapshot vitals (Phase A)
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
        "temperature_missing", "heartrate_missing", "resprate_missing",
        "o2sat_missing", "sbp_missing", "dbp_missing",
        "fever", "tachycardia", "bradycardia", "tachypnea",
        "hypoxia", "hypertension", "hypotension", "map",
        # Medications (Phase A)
        "n_medications", "meds_unknown",
        "has_cardiac_meds", "has_diabetes_meds", "has_psych_meds",
        "has_respiratory_meds", "has_opioid_meds", "has_anticoagulant_meds",
        "has_gi_meds", "has_thyroid_meds", "has_anticonvulsant_meds",
    }
    # Phase B will add: <vital>_min/_max/_last/_delta, n_<flag>_readings,
    # rhythm_<value>, rhythm_irregular. Those will surface here automatically.
    for rank, idx in enumerate(sorted_idx[:50], 1):
        name = feature_names[idx]
        if name in nurse_features or name.startswith("rhythm_") or any(
            name.startswith(f"{v}_") for v in ["temperature", "heartrate",
                                               "resprate", "o2sat", "sbp", "dbp"]
            if not name.endswith("_missing")
        ) or name.startswith("n_") and name.endswith("_readings"):
            print(f"    #{rank:>2}: {name:<35s}  importance={importances[idx]:.5f}")

    # ------------------------------------------------------------------
    # 8. Baseline comparisons
    # ------------------------------------------------------------------
    print_section("BASELINE COMPARISONS")

    diag_majority = y_diag_train.mode()[0]
    diag_majority_acc = accuracy_score(y_diag_test, [diag_majority] * len(y_diag_test))
    dept_majority = y_dept_train.mode()[0]
    dept_majority_acc = accuracy_score(y_dept_test, [dept_majority] * len(y_dept_test))

    print(f"  Diagnosis majority baseline:   {diag_majority_acc:.4f} "
          f"(always '{diagnosis_labels[diag_majority]}')")
    print(f"  Diagnosis v3 base:             {metrics_base['accuracy']:.4f}  "
          f"(+{metrics_base['accuracy']-diag_majority_acc:.4f})")
    print(f"  Diagnosis v3 with-nurse:       {metrics_nurse['accuracy']:.4f}  "
          f"(+{metrics_nurse['accuracy']-diag_majority_acc:.4f})")
    print(f"")
    print(f"  Department majority baseline:  {dept_majority_acc:.4f} "
          f"(always '{department_labels[dept_majority]}')")
    print(f"  Department v3 base:            {db_acc:.4f}  (+{db_acc-dept_majority_acc:.4f})")
    print(f"  Department v3 with-nurse:      {dn_acc:.4f}  (+{dn_acc-dept_majority_acc:.4f})")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print_section("BENCHMARK SUMMARY")
    print(f"  Test set size:               {len(y_diag_test):,}")
    print(f"  Catch-all excluded:          {meta_base.get('catch_all_excluded', '?')}")
    print(f"  v3 base features:            {features_v3_base.shape[1]}")
    print(f"  v3 with-nurse features:      {features_v3_nurse.shape[1]} "
          f"(+{features_v3_nurse.shape[1]-features_v3_base.shape[1]} nurse features)")
    print(f"")
    print(f"  DIAGNOSIS ({len(diagnosis_labels)} classes):")
    print(f"    {'':20s}  {'base':>8s}  {'nurse':>8s}  {'Improve':>8s}")
    print(f"    {'Top-1 accuracy':<20s}  {metrics_base['accuracy']:>7.2%}  "
          f"{metrics_nurse['accuracy']:>7.2%}  "
          f"{metrics_nurse['accuracy']-metrics_base['accuracy']:>+7.2%}")
    print(f"    {'Top-3 accuracy':<20s}  {metrics_base['top3']:>7.2%}  "
          f"{metrics_nurse['top3']:>7.2%}  "
          f"{metrics_nurse['top3']-metrics_base['top3']:>+7.2%}")
    print(f"    {'Cohen kappa':<20s}  {metrics_base['kappa']:>7.4f}  "
          f"{metrics_nurse['kappa']:>7.4f}  "
          f"{metrics_nurse['kappa']-metrics_base['kappa']:>+7.4f}")
    print(f"")
    print(f"  DEPARTMENT ({len(department_labels)} classes):")
    print(f"    {'':20s}  {'base':>8s}  {'nurse':>8s}  {'Improve':>8s}")
    print(f"    {'Top-1 accuracy':<20s}  {db_acc:>7.2%}  {dn_acc:>7.2%}  {dn_acc-db_acc:>+7.2%}")
    print(f"    {'Top-3 accuracy':<20s}  {db_top3:>7.2%}  {dn_top3:>7.2%}  {dn_top3-db_top3:>+7.2%}")
    print(f"    {'Cohen kappa':<20s}  {db_kappa:>7.4f}  {dn_kappa:>7.4f}  {dn_kappa-db_kappa:>+7.4f}")

    print("\n" + "#" * 70)
    print("  BENCHMARK COMPLETE (v3 base vs v3 with-nurse)")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
