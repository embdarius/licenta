"""Detailed tabular re-benchmark of every live v3 model — one auditable JSON + CSVs.

Reuses the EXACT deterministic splits and feature builders of the existing
per-agent benchmarks (so the headline numbers must reproduce ``ceiling.json``),
and computes the full metric suite from :mod:`_metrics` for:

  * Triage acuity (ESI 1-5, ordinal) + triage disposition (binary, raw)
  * Doctor diagnosis (13-class) + department (11-class), v3_base AND v3_nurse
  * Doctor disposition v3 (binary, isotonic-calibrated) — per-threshold table at
    {0.15, 0.20, 0.30, 0.40-LIVE, 0.50} + 0.10-0.90 sweep, subgroups, calibration,
    lift vs the triage-v3 cascade baseline, feature-importance-by-group audit
  * Stage-2 exact-ICD resolver (delegates to ``run_icd_benchmark``)

Outputs (all under ``--out-dir``, default
``artifacts/benchmarks/audit/<date>/tabular/``):
  tabular_full.json          master: every metric + metadata
  regression_check.json      each headline metric vs ceiling.json (delta + PASS/FAIL)
  *_per_class.csv / *_confusion_*.csv / *_thresholds.csv / *_calibration.csv / ...
  icd_resolution_full.json + icd_resolution_summary.csv

Each section loads its own data (the discharge.csv PMH parse is re-streamed per
loader — there is no shared disk cache), so the full run is long; sections free
their memory before the next. Use ``--skip-*`` to run a subset.

Run:  uv run python benchmarks/benchmark_tabular_full.py [--out-dir DIR] [--skip-icd ...]
"""

# Force UTF-8 stdout/stderr BEFORE the v3 loaders (their PMH step prints non-cp1252
# chars) — mirrors the guard in the other benchmark scripts.
import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import gc
import io
import contextlib
import json
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling _metrics / icd benchmark

import _metrics as M
from proiect_licenta.paths import (
    ARTIFACTS_DIR, TRIAGE_V3_DIR, DOCTOR_V3_BASE_DIR, DOCTOR_V3_DIR,
)

# Disposition operating points: the discrete key thresholds (incl. the LIVE 0.40
# gate) plus a finer sweep, so the under/over-triage trade-off is legible.
LIVE_THRESHOLD = 0.40
KEY_THRESHOLDS = [0.15, 0.20, 0.30, 0.40, 0.50]
SWEEP_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.10, 0.901, 0.05)]
ALL_THRESHOLDS = sorted(float(t) for t in set(KEY_THRESHOLDS) | set(SWEEP_THRESHOLDS))


def hdr(title):
    print(f"\n{'='*74}\n  {title}\n{'='*74}")


def _quiet_build(fn, *args, **kwargs):
    """Run a feature builder while swallowing its verbose stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def write_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"    csv -> {path.name}  ({len(df)} rows)")


def write_confusion(cf: dict, counts_path: Path, norm_path: Path):
    names = cf["label_names"]
    pd.DataFrame(cf["counts"], index=names, columns=names).to_csv(
        counts_path, encoding="utf-8")
    pd.DataFrame(cf["row_normalized"], index=names, columns=names).to_csv(
        norm_path, encoding="utf-8")
    print(f"    csv -> {counts_path.name} + {norm_path.name}")


def per_class_compare_df(base_pc, nurse_pc):
    """Combine two per_class tables into base|nurse|delta rows (matched by label)."""
    bn = {r["label"]: r for r in base_pc}
    rows = []
    for r in nurse_pc:
        b = bn.get(r["label"], {})
        rows.append({
            "label": r["label"], "name": r["name"], "support": r["support"],
            "base_precision": b.get("precision"), "nurse_precision": r["precision"],
            "base_recall": b.get("recall"), "nurse_recall": r["recall"],
            "delta_recall": (None if b.get("recall") is None
                             else r["recall"] - b["recall"]),
            "base_f1": b.get("f1"), "nurse_f1": r["f1"],
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Section: TRIAGE (acuity + disposition)
# ===========================================================================
def run_triage(out_dir: Path) -> dict:
    hdr("TRIAGE v3 — acuity (ESI 1-5) + disposition (admit/discharge)")
    from proiect_licenta.training.train_triage_v3 import (
        load_and_clean_data as load_v3, build_features as build_v3,
    )
    df, _eds = load_v3()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["acuity"])
    y_acuity = test_df["acuity"].reset_index(drop=True)
    y_admit = test_df["admitted"].astype(int).reset_index(drop=True)
    arrival = test_df["arrival_transport"].reset_index(drop=True)

    tfidf = joblib.load(TRIAGE_V3_DIR / "tfidf_vectorizer.joblib")
    severity = joblib.load(TRIAGE_V3_DIR / "severity_map.joblib")
    vmed = joblib.load(TRIAGE_V3_DIR / "vital_medians.joblib")
    acuity_model = joblib.load(TRIAGE_V3_DIR / "acuity_model.joblib")
    disp_model = joblib.load(TRIAGE_V3_DIR / "disposition_model.joblib")
    meta = json.loads((TRIAGE_V3_DIR / "model_metadata.json").read_text())

    X, _, _, _ = _quiet_build(build_v3, test_df, tfidf=tfidf, severity_map=severity,
                              vital_medians=vmed, fit=False)
    acu_prob = acuity_model.predict_proba(X)          # cols = ESI 1..5 ascending
    acu_pred = acuity_model.predict(X) + 1
    Xd = X.copy()
    Xd["predicted_acuity"] = acu_pred
    p_admit = disp_model.predict_proba(Xd)[:, 1]

    acuity_rep = M.ordinal_report(y_acuity.values, acu_pred, acu_prob,
                                  levels=(1, 2, 3, 4, 5))
    # Per-arrival-transport acuity breakdown (confirms walk-in lift).
    by_arrival = []
    for label, mask in [
        ("ambulance/helicopter", arrival.isin(["AMBULANCE", "HELICOPTER"])),
        ("walk_in", arrival == "WALK IN"),
        ("other/unknown", ~arrival.isin(["AMBULANCE", "HELICOPTER", "WALK IN"])),
    ]:
        m = mask.values
        if m.sum() == 0:
            continue
        yt, yp = y_acuity.values[m], acu_pred[m]
        by_arrival.append({
            "group": label, "n": int(m.sum()),
            "exact": M._f(np.mean(yt == yp)),
            "within_1": M._f(np.mean(np.abs(yp - yt) <= 1)),
            "mae": M._f(np.mean(np.abs(yp - yt))),
            "under_triage_rate": M._f(np.mean(yp > yt)),
            "over_triage_rate": M._f(np.mean(yp < yt)),
        })
    disp_rep = M.binary_report(y_admit.values, p_admit, thresholds=ALL_THRESHOLDS)

    report = {
        "artifact": str(TRIAGE_V3_DIR),
        "trained_at": meta.get("trained_at"),
        "n_features": int(X.shape[1]),
        "n_test": int(len(y_acuity)),
        "operating_threshold_disposition": 0.5,
        "acuity": acuity_rep,
        "acuity_by_arrival": by_arrival,
        "disposition": disp_rep,
    }

    # CSVs
    write_df(pd.DataFrame(acuity_rep["per_class"]),
             out_dir / "triage_acuity_per_class.csv")
    cf = M.confusion_frame(y_acuity.values, acu_pred, [1, 2, 3, 4, 5],
                           [f"ESI {l}" for l in (1, 2, 3, 4, 5)])
    write_confusion(cf, out_dir / "triage_acuity_confusion_counts.csv",
                    out_dir / "triage_acuity_confusion_norm.csv")
    write_df(pd.DataFrame(by_arrival), out_dir / "triage_acuity_by_arrival.csv")
    thr_df = pd.DataFrame(disp_rep["thresholds"])
    thr_df["is_key"] = thr_df["threshold"].isin(KEY_THRESHOLDS)
    thr_df["is_operating"] = thr_df["threshold"] == 0.5
    write_df(thr_df, out_dir / "triage_disposition_thresholds.csv")
    write_df(pd.DataFrame(disp_rep["reliability_bins"]),
             out_dir / "triage_disposition_calibration.csv")

    del df, train_df, test_df, X, Xd, acu_prob
    gc.collect()
    return report


# ===========================================================================
# Section: DOCTOR diagnosis + department (v3_base vs v3_nurse)
# ===========================================================================
def run_doctor_heads(out_dir: Path) -> dict:
    hdr("DOCTOR v3 — diagnosis (13c) + department (11c): base vs nurse")
    from proiect_licenta.training.train_doctor import DEPARTMENT_NAMES
    from proiect_licenta.training.train_nurse_v3 import (
        load_and_clean_data as load_v3, build_features as build_nurse,
    )
    from proiect_licenta.training.train_doctor_v3 import build_features as build_base

    df = load_v3()
    feat_base = _quiet_build(build_base, df)
    feat_nurse = _quiet_build(build_nurse, df)

    meta_base = json.loads((DOCTOR_V3_BASE_DIR / "metadata.json").read_text())
    meta_nurse = json.loads((DOCTOR_V3_DIR / "metadata.json").read_text())
    diag_labels = meta_base["diagnosis_labels"]
    dept_labels = meta_base["department_labels"]
    diag_map = {l: i for i, l in enumerate(diag_labels)}
    dept_map = {l: i for i, l in enumerate(dept_labels)}
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)
    y_dept = df["service_group"].map(dept_map).reset_index(drop=True)

    Xb_tr, Xb_te, y_diag_tr, y_diag_te, y_dept_tr, y_dept_te = train_test_split(
        feat_base, y_diag, y_dept, test_size=0.2, random_state=42, stratify=y_diag)
    Xn_tr, Xn_te, _, _, _, _ = train_test_split(
        feat_nurse, y_diag, y_dept, test_size=0.2, random_state=42, stratify=y_diag)

    base_diag = joblib.load(DOCTOR_V3_BASE_DIR / "diagnosis_model.joblib")
    base_dept = joblib.load(DOCTOR_V3_BASE_DIR / "department_model.joblib")
    nurse_diag = joblib.load(DOCTOR_V3_DIR / "diagnosis_model.joblib")
    nurse_dept = joblib.load(DOCTOR_V3_DIR / "department_model.joblib")

    diag_idx = list(range(len(diag_labels)))
    dept_idx = list(range(len(dept_labels)))

    # ---- Diagnosis ----
    diag_base = M.multiclass_report(
        y_diag_te.values, base_diag.predict(Xb_te), base_diag.predict_proba(Xb_te),
        diag_idx, diag_labels, topk=(1, 3, 5), y_train=y_diag_tr.values)
    nd_pred = nurse_diag.predict(Xn_te)
    nd_prob = nurse_diag.predict_proba(Xn_te)
    diag_nurse = M.multiclass_report(
        y_diag_te.values, nd_pred, nd_prob, diag_idx, diag_labels,
        topk=(1, 3, 5), y_train=y_diag_tr.values)

    # ---- Department (cascade) ----
    Xb_te_d = Xb_te.copy()
    Xb_te_d["predicted_diagnosis"] = base_diag.predict(Xb_te)
    db_prob = base_dept.predict_proba(Xb_te_d)
    db_pred = np.argmax(db_prob, axis=1)
    dept_base = M.multiclass_report(
        y_dept_te.values, db_pred, db_prob, dept_idx,
        [DEPARTMENT_NAMES.get(l, l) for l in dept_labels],
        topk=(1, 3, 5), y_train=y_dept_tr.values)

    Xn_te_d = Xn_te.copy()
    casc = meta_nurse.get("diag_cascade_cols")
    if casc:
        for k, col in enumerate(casc):
            Xn_te_d[col] = nd_prob[:, k]
    else:
        Xn_te_d["predicted_diagnosis"] = nd_pred
    dn_prob = nurse_dept.predict_proba(Xn_te_d)
    dn_pred = np.argmax(dn_prob, axis=1)
    dept_nurse = M.multiclass_report(
        y_dept_te.values, dn_pred, dn_prob, dept_idx,
        [DEPARTMENT_NAMES.get(l, l) for l in dept_labels],
        topk=(1, 3, 5), y_train=y_dept_tr.values)

    def deltas(b, n):
        keys = ["accuracy", "top3", "top5", "mrr", "cohen_kappa",
                "balanced_accuracy", "f1_macro", "macro_auc_ovr"]
        return {k: (None if b.get(k) is None or n.get(k) is None
                    else n[k] - b[k]) for k in keys}

    report = {
        "artifact_base": str(DOCTOR_V3_BASE_DIR),
        "artifact_nurse": str(DOCTOR_V3_DIR),
        "trained_at_base": meta_base.get("trained_at"),
        "trained_at_nurse": meta_nurse.get("trained_at"),
        "n_test": int(len(y_diag_te)),
        "diagnosis_labels": diag_labels,
        "department_labels": dept_labels,
        "diagnosis": {"v3_base": diag_base, "v3_nurse": diag_nurse,
                      "delta_nurse_minus_base": deltas(diag_base, diag_nurse)},
        "department": {"v3_base": dept_base, "v3_nurse": dept_nurse,
                       "delta_nurse_minus_base": deltas(dept_base, dept_nurse)},
    }

    # CSVs
    write_df(per_class_compare_df(diag_base["per_class"], diag_nurse["per_class"]),
             out_dir / "doctor_diagnosis_per_class.csv")
    write_df(per_class_compare_df(dept_base["per_class"], dept_nurse["per_class"]),
             out_dir / "doctor_department_per_class.csv")
    write_confusion(
        M.confusion_frame(y_diag_te.values, base_diag.predict(Xb_te), diag_idx, diag_labels),
        out_dir / "doctor_diagnosis_confusion_base_counts.csv",
        out_dir / "doctor_diagnosis_confusion_base_norm.csv")
    write_confusion(
        M.confusion_frame(y_diag_te.values, nd_pred, diag_idx, diag_labels),
        out_dir / "doctor_diagnosis_confusion_nurse_counts.csv",
        out_dir / "doctor_diagnosis_confusion_nurse_norm.csv")
    write_confusion(
        M.confusion_frame(y_dept_te.values, db_pred, dept_idx, dept_labels),
        out_dir / "doctor_department_confusion_base_counts.csv",
        out_dir / "doctor_department_confusion_base_norm.csv")
    write_confusion(
        M.confusion_frame(y_dept_te.values, dn_pred, dept_idx, dept_labels),
        out_dir / "doctor_department_confusion_nurse_counts.csv",
        out_dir / "doctor_department_confusion_nurse_norm.csv")

    del df, feat_base, feat_nurse, Xb_te, Xn_te, Xb_te_d, Xn_te_d
    gc.collect()
    return report


# ===========================================================================
# Section: DOCTOR DISPOSITION v3 (calibrated binary)
# ===========================================================================
def run_disposition(out_dir: Path) -> dict:
    hdr("DOCTOR DISPOSITION v3 — calibrated binary (LIVE threshold 0.40)")
    from proiect_licenta.training.train_doctor_disposition import (
        load_and_clean_data, build_features, SOFT_CASCADE_COLS,
    )
    from proiect_licenta.pmh_features import PMH_FEATURE_COLS
    from proiect_licenta.training.train_nurse import MED_CATEGORY_KEYWORDS
    from proiect_licenta.training.train_nurse_v3 import LONG_VITAL_FEATURE_COLS

    df = load_and_clean_data()
    features = build_features(df)
    y = df["admitted"].astype(int).reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.20, random_state=42, stratify=y)
    X_test = features.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    calibrated = joblib.load(DOCTOR_V3_DIR / "disposition_model.joblib")
    raw_path = DOCTOR_V3_DIR / "disposition_model_raw.joblib"
    raw = joblib.load(raw_path) if raw_path.exists() else None

    doctor_proba = calibrated.predict_proba(X_test)[:, 1]
    triage_proba = X_test["triage_disposition_proba_admit"].values.astype(float)
    raw_proba = raw.predict_proba(X_test)[:, 1] if raw is not None else None

    doc_rep = M.binary_report(y_test.values, doctor_proba, thresholds=ALL_THRESHOLDS)
    triage_rep = M.binary_report(y_test.values, triage_proba, thresholds=ALL_THRESHOLDS)
    raw_cal = (M.calibration_report(y_test.values, raw_proba) if raw_proba is not None
               else None)

    # Lift vs triage cascade baseline at the LIVE threshold.
    def at(rep, t):
        return next(r for r in rep["thresholds"] if abs(r["threshold"] - t) < 1e-9)
    d_live, t_live = at(doc_rep, LIVE_THRESHOLD), at(triage_rep, LIVE_THRESHOLD)
    lift = {
        "threshold": LIVE_THRESHOLD,
        "accuracy": d_live["accuracy"] - t_live["accuracy"],
        "roc_auc": (doc_rep["roc_auc"] - triage_rep["roc_auc"]),
        "brier": doc_rep["brier"] - triage_rep["brier"],
        "ece": doc_rep["ece"] - triage_rep["ece"],
        "under_triage": d_live["under_triage"] - t_live["under_triage"],
        "over_triage": d_live["over_triage"] - t_live["over_triage"],
        "sensitivity": d_live["sensitivity"] - t_live["sensitivity"],
        "specificity": d_live["specificity"] - t_live["specificity"],
    }

    # Per-subgroup at the LIVE threshold.
    from sklearn.metrics import accuracy_score, roc_auc_score
    subgroups = {
        "elderly (age>=65)": df_test["age"] >= 65,
        "polypharmacy (n_meds>=5)": df_test["n_medications"] >= 5,
        "abnormal vitals (>=1 flag)": (
            df_test["fever"] + df_test["tachycardia"] + df_test["tachypnea"]
            + df_test["hypoxia"] + df_test["hypotension"]) >= 1,
        "repeat visitor (prior ED)": (df_test["n_prior_ed_visits"] > 0
                                      if "n_prior_ed_visits" in df_test else None),
        "prior admission": (df_test["n_prior_admissions"] > 0
                            if "n_prior_admissions" in df_test else None),
        "non-sinus rhythm": (df_test["rhythm_irregular"] == 1
                             if "rhythm_irregular" in df_test else None),
    }
    subgroup_rows = []
    for name, mask in subgroups.items():
        if mask is None:
            continue
        m = mask.astype(bool).values
        n = int(m.sum())
        ys = y_test.values[m]
        if n < 200 or ys.sum() in (0, n):
            subgroup_rows.append({"subgroup": name, "n": n, "note": "too small / single-class"})
            continue
        dp = (doctor_proba[m] >= LIVE_THRESHOLD).astype(int)
        tp = (triage_proba[m] >= LIVE_THRESHOLD).astype(int)
        subgroup_rows.append({
            "subgroup": name, "n": n, "admit_rate": M._f(ys.mean()),
            "base_acc": M._f(accuracy_score(ys, tp)),
            "doc_acc": M._f(accuracy_score(ys, dp)),
            "delta_acc": M._f(accuracy_score(ys, dp) - accuracy_score(ys, tp)),
            "base_auc": M._f(roc_auc_score(ys, triage_proba[m])),
            "doc_auc": M._f(roc_auc_score(ys, doctor_proba[m])),
            "doc_under_triage": M._f(np.mean((ys == 1) & (dp == 0)) / max(ys.mean(), 1e-9)),
            "doc_over_triage": M._f(np.mean((ys == 0) & (dp == 1)) / max(1 - ys.mean(), 1e-9)),
        })

    # Feature-importance-by-group audit (uncalibrated model).
    feat_groups = None
    if raw is not None:
        booster = raw.get_booster()
        importance = booster.get_score(importance_type="gain")
        cols = list(X_test.columns)
        items = []
        for name, gain in importance.items():
            col = name
            if name not in cols:
                try:
                    col = cols[int(name[1:])]
                except Exception:
                    col = name
            items.append((col, float(gain)))

        def group_of(col):
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

        gtot = {}
        for col, g in items:
            gtot[group_of(col)] = gtot.get(group_of(col), 0.0) + g
        total = sum(gtot.values()) or 1.0
        feat_groups = {
            "by_group": {g: {"gain": M._f(v), "pct": M._f(100 * v / total)}
                         for g, v in sorted(gtot.items(), key=lambda kv: -kv[1])},
            "pmh_cols_with_gain": int(sum(1 for c, _ in items if c in PMH_FEATURE_COLS)),
            "n_pmh_cols": len(PMH_FEATURE_COLS),
            "soft_cascade_cols_with_gain": int(sum(1 for c, _ in items
                                                   if c in SOFT_CASCADE_COLS)),
            "n_soft_cascade_cols": len(SOFT_CASCADE_COLS),
            "top25": [{"rank": r, "column": c, "group": group_of(c), "gain": M._f(g)}
                      for r, (c, g) in enumerate(sorted(items, key=lambda kv: -kv[1])[:25], 1)],
        }

    report = {
        "artifact": str(DOCTOR_V3_DIR / "disposition_model.joblib"),
        "live_threshold": LIVE_THRESHOLD,
        "n_features": int(X_test.shape[1]),
        "n_test": int(len(y_test)),
        "calibrated": doc_rep,
        "uncalibrated_calibration": raw_cal,
        "triage_v3_cascade_baseline": triage_rep,
        "lift_vs_triage_at_live": lift,
        "subgroups_at_live": subgroup_rows,
        "feature_importance_groups": feat_groups,
    }

    # CSVs
    thr_df = pd.DataFrame(doc_rep["thresholds"])
    thr_df["is_key"] = thr_df["threshold"].isin(KEY_THRESHOLDS)
    thr_df["is_live"] = thr_df["threshold"] == LIVE_THRESHOLD
    write_df(thr_df, out_dir / "doctor_disposition_thresholds.csv")
    write_df(pd.DataFrame(doc_rep["reliability_bins"]),
             out_dir / "doctor_disposition_calibration.csv")
    write_df(pd.DataFrame(subgroup_rows), out_dir / "doctor_disposition_subgroups.csv")
    if feat_groups is not None:
        fg = pd.DataFrame([{"group": g, **v} for g, v in feat_groups["by_group"].items()])
        write_df(fg, out_dir / "doctor_disposition_feature_groups.csv")

    del df, features, X_test, df_test
    gc.collect()
    return report


# ===========================================================================
# Section: ICD resolver (delegates to run_icd_benchmark)
# ===========================================================================
def run_icd(out_dir: Path) -> dict:
    hdr("STAGE-2 EXACT-ICD RESOLVER (oracle + e2e + 3 graded engines)")
    from benchmark_icd_resolution import run_icd_benchmark
    results = run_icd_benchmark(out_path=out_dir / "icd_resolution_full.json")

    # Long-format summary CSV: section | model | granularity | variant | metric | value
    rows = []
    for g, vmap in results["oracle"].items():
        for variant, cell in vmap.items():
            for k, v in cell["strict"].items():
                rows.append(["oracle", "-", g, variant, k, v])
            for eng, d in cell["graded"].items():
                for k, v in d.items():
                    rows.append(["oracle", "-", g, variant, f"graded:{eng}:{k}", v])
    for model, gmap in results["end_to_end"].items():
        s1 = gmap.get("stage1_category_recall", {})
        for k, v in s1.items():
            rows.append(["end_to_end", model, "-", "-", f"stage1_cat_recall:{k}", v])
        for g, vmap in gmap.items():
            if g == "stage1_category_recall":
                continue
            for variant, cell in vmap.items():
                for k, v in cell["strict"].items():
                    rows.append(["end_to_end", model, g, variant, k, v])
                for eng, d in cell["graded"].items():
                    for k, v in d.items():
                        rows.append(["end_to_end", model, g, variant, f"graded:{eng}:{k}", v])
    for g, vmap in results.get("before_after_nurse", {}).items():
        for k, d in vmap.get("strict", {}).items():
            for who, v in d.items():
                rows.append(["before_after_nurse", who, g, "blend+vitals", k, v])
    write_df(pd.DataFrame(rows, columns=["section", "model", "granularity",
                                         "variant", "metric", "value"]),
             out_dir / "icd_resolution_summary.csv")
    return results


# ===========================================================================
# Regression check vs ceiling.json
# ===========================================================================
def regression_check(master: dict, tol: float = 2e-3) -> dict:
    ceiling_path = ARTIFACTS_DIR / "benchmarks" / "tabular" / "ceiling.json"
    if not ceiling_path.exists():
        return {"status": "SKIPPED", "reason": f"ceiling.json not found at {ceiling_path}"}
    ceiling = json.loads(ceiling_path.read_text(encoding="utf-8"))

    def disp_at(rep, t, key):
        if rep is None:
            return None
        for r in rep["thresholds"]:
            if abs(r["threshold"] - t) < 1e-9:
                return r[key]
        return None

    checks = []   # (name, ceiling_value, new_value)

    tri = master.get("triage")
    if tri:
        a, d = tri["acuity"], tri["disposition"]
        c = ceiling.get("triage_v3", {})
        checks += [
            ("triage.acuity.exact", c.get("acuity", {}).get("exact"), a["exact"]),
            ("triage.acuity.within_1", c.get("acuity", {}).get("within_1"), a["within_1"]),
            ("triage.acuity.kappa_quadratic",
             c.get("acuity", {}).get("kappa_quadratic"), a["kappa_quadratic"]),
            ("triage.disposition.accuracy@0.5",
             c.get("disposition", {}).get("accuracy"), disp_at(d, 0.5, "accuracy")),
            ("triage.disposition.roc_auc",
             c.get("disposition", {}).get("roc_auc"), d["roc_auc"]),
        ]

    doc = master.get("doctor_heads")
    if doc:
        db = doc["diagnosis"]["v3_base"]
        dn = doc["diagnosis"]["v3_nurse"]
        pb = doc["department"]["v3_base"]
        pn = doc["department"]["v3_nurse"]
        cb = ceiling.get("doctor_v3_base", {})
        cn = ceiling.get("doctor_v3_nurse", {})
        checks += [
            ("doctor.diag_base.top1", cb.get("diagnosis_13c", {}).get("top1"), db["accuracy"]),
            ("doctor.diag_base.top3", cb.get("diagnosis_13c", {}).get("top3"), db["top3"]),
            ("doctor.diag_base.top5", cb.get("diagnosis_13c", {}).get("top5"), db["top5"]),
            ("doctor.dept_base.top1", cb.get("department_11c", {}).get("top1"), pb["accuracy"]),
            ("doctor.dept_base.top3", cb.get("department_11c", {}).get("top3"), pb["top3"]),
            ("doctor.diag_nurse.top1", cn.get("diagnosis_13c", {}).get("top1"), dn["accuracy"]),
            ("doctor.diag_nurse.top3", cn.get("diagnosis_13c", {}).get("top3"), dn["top3"]),
            ("doctor.diag_nurse.kappa", cn.get("diagnosis_13c", {}).get("kappa"), dn["cohen_kappa"]),
            ("doctor.dept_nurse.top1", cn.get("department_11c", {}).get("top1"), pn["accuracy"]),
            ("doctor.dept_nurse.top3", cn.get("department_11c", {}).get("top3"), pn["top3"]),
            ("doctor.dept_nurse.kappa", cn.get("department_11c", {}).get("kappa"), pn["cohen_kappa"]),
        ]

    dispo = master.get("doctor_disposition")
    if dispo:
        cal = dispo["calibrated"]
        cc = ceiling.get("doctor_disposition_v3", {})
        checks += [
            ("dispo.accuracy@0.5", cc.get("accuracy"), disp_at(cal, 0.5, "accuracy")),
            ("dispo.roc_auc", cc.get("roc_auc"), cal["roc_auc"]),
            ("dispo.brier", cc.get("brier"), cal["brier"]),
            ("dispo.ece_10bin", cc.get("ece_10bin"), cal["ece"]),
            ("dispo.sensitivity@0.5", cc.get("sensitivity_admit"), disp_at(cal, 0.5, "sensitivity")),
            ("dispo.specificity@0.5", cc.get("specificity_disch"), disp_at(cal, 0.5, "specificity")),
            ("dispo.under_triage@0.5", cc.get("under_triage"), disp_at(cal, 0.5, "under_triage")),
            ("dispo.over_triage@0.5", cc.get("over_triage"), disp_at(cal, 0.5, "over_triage")),
        ]

    icd = master.get("icd")
    if icd:
        ci = ceiling.get("icd_resolver_stage2", {})
        oroll = icd["oracle"]["rollup"]
        e2e_n = icd["end_to_end"].get("v3 with-nurse", {})
        checks += [
            ("icd.oracle@5.blend",
             ci.get("oracle_at5_rollup", {}).get("blend"),
             oroll["blend"]["strict"]["oracle@5"]),
            ("icd.oracle@5.blend_vitals",
             ci.get("oracle_at5_rollup", {}).get("blend_vitals"),
             oroll["blend+vitals"]["strict"]["oracle@5"]),
            ("icd.e2e_union.blend_vitals.v3nurse",
             ci.get("e2e_union_rollup_v3nurse", {}).get("blend_vitals"),
             e2e_n.get("rollup", {}).get("blend+vitals", {}).get("strict", {}).get("union")),
            ("icd.stage1.top1.v3nurse",
             ci.get("stage1_category_recall_v3nurse", {}).get("top1"),
             e2e_n.get("stage1_category_recall", {}).get("top1")),
            ("icd.stage1.top5.v3nurse",
             ci.get("stage1_category_recall_v3nurse", {}).get("top5"),
             e2e_n.get("stage1_category_recall", {}).get("top5")),
        ]

    results = []
    n_fail = n_checked = 0
    for name, ceil_v, new_v in checks:
        if ceil_v is None or new_v is None:
            results.append({"metric": name, "ceiling": ceil_v, "new": new_v,
                            "delta": None, "status": "NO_REF"})
            continue
        n_checked += 1
        delta = new_v - ceil_v
        ok = abs(delta) <= tol
        n_fail += not ok
        results.append({"metric": name, "ceiling": ceil_v, "new": round(new_v, 6),
                        "delta": round(delta, 6), "status": "PASS" if ok else "FAIL"})
    return {
        "status": "PASS" if n_fail == 0 else "FAIL",
        "tolerance": tol, "n_checked": n_checked, "n_failed": n_fail,
        "checks": results,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default artifacts/benchmarks/audit/<date>/tabular/)")
    ap.add_argument("--skip-triage", action="store_true")
    ap.add_argument("--skip-doctor", action="store_true")
    ap.add_argument("--skip-disposition", action="store_true")
    ap.add_argument("--skip-icd", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        ARTIFACTS_DIR / "benchmarks" / "audit" / date.today().isoformat() / "tabular")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 74)
    print("  DETAILED TABULAR RE-BENCHMARK — all live v3 models")
    print(f"  Output: {out_dir}")
    print("#" * 74)

    master = {
        "_meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "split": "80/20 stratified, random_state=42",
            "disposition_thresholds": ALL_THRESHOLDS,
            "disposition_key_thresholds": KEY_THRESHOLDS,
            "disposition_live_threshold": LIVE_THRESHOLD,
        },
    }

    if not args.skip_triage:
        master["triage"] = run_triage(out_dir)
    if not args.skip_doctor:
        master["doctor_heads"] = run_doctor_heads(out_dir)
    if not args.skip_disposition:
        master["doctor_disposition"] = run_disposition(out_dir)
    if not args.skip_icd:
        master["icd"] = run_icd(out_dir)

    # Regression check vs ceiling.json
    hdr("REGRESSION CHECK vs ceiling.json")
    reg = regression_check(master)
    master["_meta"]["regression_check_status"] = reg["status"]
    (out_dir / "regression_check.json").write_text(
        json.dumps(reg, indent=2), encoding="utf-8")
    print(f"  Overall: {reg['status']}  "
          f"({reg.get('n_checked', 0)} checked, {reg.get('n_failed', 0)} failed, "
          f"tol={reg.get('tolerance')})")
    for c in reg.get("checks", []):
        if c["status"] in ("FAIL", "NO_REF"):
            print(f"    [{c['status']}] {c['metric']}: ceiling={c['ceiling']} "
                  f"new={c['new']} delta={c['delta']}")
    if reg["status"] == "PASS":
        print("  All headline metrics reproduce ceiling.json within tolerance.")

    (out_dir / "tabular_full.json").write_text(
        json.dumps(master, indent=2), encoding="utf-8")
    print(f"\n  Master JSON -> {out_dir / 'tabular_full.json'}")
    print("\n" + "#" * 74)
    print("  TABULAR BENCHMARK COMPLETE")
    print("#" * 74 + "\n")


if __name__ == "__main__":
    main()
