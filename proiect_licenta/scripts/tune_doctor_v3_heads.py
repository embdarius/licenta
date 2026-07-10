"""Constrained Optuna sweep for the doctor v3 department and disposition heads.

Reporting-only. Mirrors scripts/tune_triage_v3.py: it searches only the inherited
"Group 2" XGBoost regularization knobs (max_depth, subsample, colsample_*,
min_child_weight, gamma, reg_alpha, reg_lambda). Each head's
documented "Group 1" config is FROZEN - the values are the hand-picked defaults
baked into the training pipelines:

  * department  (train_nurse_v3.train_model):
        n_estimators=3000, learning_rate=0.02, early_stopping=100,
        eval_metric=mlogloss, multi:softprob, sqrt-inverse class weighting
        (class_weight_exponent=0.5), diagnosis-softmax cascade (13 cols),
        isotonic calibration (applied in the report stage only).
  * disposition (train_doctor_disposition.train_disposition):
        n_estimators=5000, learning_rate=0.02, early_stopping=150,
        eval_metric=logloss, binary:logistic, scale_pos_weight=sqrt(N_neg/N_pos),
        isotonic calibration (applied in the report stage only).

It never writes the live doctor v3 joblibs. All output goes to a dedicated
`artifacts/doctor/v3/hpo/` subdir + (for disposition) a dedicated feature cache,
so nothing produced by the training notebooks or the older write-mode diagnosis
sweep (scripts/tune_doctor_v3.py) is overwritten.

Objective design:
  * department  - maximize macro-F1 across the 11 service classes (matches the
                  diagnosis sweep's "useful across categories" thesis). No
                  constraint. The diagnosis-softmax cascade is built once per
                  run from a diagnosis model trained on the inner-train split
                  (using the existing tuned_params.json if present) and reused
                  across all trials, so only the department Group-2 varies.
  * disposition - maximize ROC AUC SUBJECT TO a hard constraint: under-triage
                  rate (admit predicted as discharge, threshold 0.5) <= the
                  incumbent's. The current (hand-tuned) Group-2 values are
                  enqueued as trial 0 and define that baseline in-sample on the
                  inner-val split. The dangerous error (missing an admission)
                  can never get worse than the deployed model. This is the
                  direct analog of the triage *acuity* under-triage constraint.

Per-trial fits during the search are LEAN (no calibration). Calibration is a
monotone transform that barely moves the argmax / ranking; it is applied only
in the report stage so incumbent-vs-best is apples-to-apples.

Resume is built in (mirrors scripts/tune_triage_v3.py): every trial is committed
to a SQLite study on Drive; `load_if_exists=True` means a fresh Colab session
re-running the tune cell continues where it left off. A human-readable JSON log
is rewritten after every completed trial.

Stages:
    --stage department    tune the department head (macro-F1)
    --stage disposition   tune the disposition head (constrained ROC AUC)
    --stage report        evaluate incumbent vs best on the OUTER test split
                          and write doctor_hpo_results.json (thesis table)

Pre-flight:
    --selftest            synthetic, CPU-only, no MIMIC/GPU - validates all
                          plumbing (study + JSON log + resume + constraint flag)
                          in seconds, for BOTH heads
    --smoke               real-data subsample, throwaway paths, fast trees -
                          run on Colab GPU BEFORE committing to a full sweep

Usage on Colab (after symlinking artifacts/ + data/ to Drive):
    !XGB_DEVICE=cuda XGB_TREE_METHOD=hist python scripts/tune_doctor_v3_heads.py \\
        --stage department --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# Python 3.12 / pydantic / optuna compatibility shim (copied verbatim from
# scripts/tune_triage_v3.py). Pydantic (via CrewAI) monkey-patches
# warnings.warn with a filter that predates 3.12's skip_file_prefixes kwarg,
# which optuna passes. Wrap it kwargs-tolerantly. Must run before `import optuna`.
import warnings as _w
if not getattr(_w, "_optuna_compat_patched", False):
    _patched_warn = _w.warn

    def _compatible_warn(message, category=Warning, stacklevel=1, source=None, **kwargs):
        try:
            return _patched_warn(message, category=category, stacklevel=stacklevel,
                                 source=source, **kwargs)
        except TypeError:
            return _patched_warn(message, category=category, stacklevel=stacklevel,
                                 source=source)

    _compatible_warn._kwargs_tolerant = True
    _w.warn = _compatible_warn
    _w._optuna_compat_patched = True


def _ensure_warn_shim():
    """Re-wrap ``warnings.warn`` so it tolerates Optuna's py3.12
    ``skip_file_prefixes`` kwarg. The top-level shim runs at import time, but
    building the feature caches imports the training modules, which pull in
    crewai/pydantic - and pydantic RE-patches ``warnings.warn`` with a wrapper
    that rejects unknown kwargs, clobbering our shim. Calling this right before
    any ``optuna.create_study`` re-installs a tolerant wrapper around whatever
    ``warnings.warn`` is current. Idempotent (marked via ``_kwargs_tolerant``)."""
    cur = _w.warn
    if getattr(cur, "_kwargs_tolerant", False):
        return

    def _tolerant(message, category=Warning, stacklevel=1, source=None, **kwargs):
        try:
            return cur(message, category=category, stacklevel=stacklevel,
                       source=source, **kwargs)
        except TypeError:
            return cur(message, category=category, stacklevel=stacklevel,
                       source=source)

    _tolerant._kwargs_tolerant = True
    _w.warn = _tolerant


import numpy as np
import pandas as pd
import joblib

# Run with the project on the path even when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    brier_score_loss, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
    _HAS_FROZEN = True
except ImportError:  # pragma: no cover
    FrozenEstimator = None  # type: ignore[assignment]
    _HAS_FROZEN = False

from proiect_licenta.paths import (
    DOCTOR_V3_DIR, DOCTOR_V3_HPO_DIR,
    DERIVED_DIR, DOCTOR_DISPO_TUNE_CACHE_DIR,
)

# Read the GPU device at RUNTIME, not import-time (see the same note in
# tune_triage_v3.py). The training modules cache their own XGB_DEVICE at first
# import; reading os.environ here keeps the device correct regardless of which
# Colab cell imported what first.
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")
XGB_TREE_METHOD = os.environ.get("XGB_TREE_METHOD", "hist")


# Constants
DEPT_STUDY_NAME = "doctor_department_macro_f1"
DISP_STUDY_NAME = "doctor_disposition_auc"

# Frozen "Group 1" per head. These mirror the hand-picked values baked inline
# into the training pipelines (single source of truth lives there; we cannot
# import them as named constants because they are inline kwargs):
#   department  -> train_nurse_v3.train_model (lines ~855-859, cw_exponent 0.5)
#   disposition -> train_doctor_disposition.train_disposition (lines ~519-525)
DEPT_N_ESTIMATORS = 3000
DEPT_LEARNING_RATE = 0.02
DEPT_EARLY_STOPPING = 100
DEPT_CW_EXPONENT = 0.5  # sqrt-inverse-frequency class weighting

DISP_N_ESTIMATORS = 5000
DISP_LEARNING_RATE = 0.02
DISP_EARLY_STOPPING = 150

# The current hand-tuned Group-2 values (identical across the doctor heads and
# the triage heads - they were all inherited from v1/v2). Enqueued as trial 0
# of each study so Optuna always evaluates the incumbent first; for disposition
# its inner-val under-triage rate becomes the feasibility threshold.
FROZEN_GROUP2 = {
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "colsample_bylevel": 0.7,
    "min_child_weight": 3,
    "gamma": 0.05,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
}


# Group-2 search space (regularization box; identical to
# tune_triage_v3.suggest_group2).
def suggest_group2(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
    }


# Group-1 search (the per-head cost/weighting config). SINGLE-OBJECTIVE per the
# doctor Group-1 plan - lower priority than triage's, mostly a cost/fidelity
# envelope (the class-weight exponent was already validated ~0.52 by the
# diagnosis sweep). Group-2 stays frozen at FROZEN_GROUP2 throughout. Department
# keeps the unconstrained macro-F1 objective; disposition keeps the under-triage
# constraint - i.e. the same objectives/constraints as the Group-2 sweep.
DEPT_G1_STUDY_NAME = "doctor_department_g1"
DISP_G1_STUDY_NAME = "doctor_disposition_g1"

FROZEN_GROUP1_DEPT = {
    "learning_rate": DEPT_LEARNING_RATE,        # 0.02
    "n_estimators": DEPT_N_ESTIMATORS,          # 3000
    "class_weight_exponent": DEPT_CW_EXPONENT,  # 0.5 (sqrt-inverse frequency)
}
FROZEN_GROUP1_DISP = {
    "learning_rate": DISP_LEARNING_RATE,        # 0.02
    "n_estimators": DISP_N_ESTIMATORS,          # 5000
    "scale_pos_weight_exponent": 0.5,           # live = sqrt(N_neg/N_pos)
}


def suggest_group1_department(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "n_estimators": trial.suggest_categorical("n_estimators", [2000, 3000, 5000]),
        "class_weight_exponent": trial.suggest_float("class_weight_exponent", 0.3, 0.8),
    }


def suggest_group1_disposition(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "n_estimators": trial.suggest_categorical("n_estimators", [3000, 5000, 8000]),
        "scale_pos_weight_exponent": trial.suggest_float("scale_pos_weight_exponent", 0.3, 1.0),
    }


def _g1_dept_gcfg(base: dict, p: dict) -> dict:
    g = dict(base)
    g["dept_learning_rate"] = p["learning_rate"]
    g["dept_n_estimators"] = int(p["n_estimators"])
    g["dept_cw_exponent"] = p["class_weight_exponent"]
    return g


def _g1_disp_gcfg(base: dict, p: dict) -> dict:
    g = dict(base)
    g["disp_learning_rate"] = p["learning_rate"]
    g["disp_n_estimators"] = int(p["n_estimators"])
    g["disp_scale_pos_weight_exponent"] = p["scale_pos_weight_exponent"]
    return g


# Per-trial XGBoost iteration progress bar (tqdm). Optional; silent fallback.
def _make_trial_progress_callback(n_total: int, desc: str):
    try:
        from tqdm.auto import tqdm as _tqdm
        from xgboost.callback import TrainingCallback
    except ImportError:
        return None

    class _TqdmTrialCallback(TrainingCallback):  # type: ignore[misc]
        def __init__(self):
            self.pbar = _tqdm(total=n_total, desc=desc, unit="iter", leave=False)

        def after_iteration(self, model, epoch, evals_log):
            self.pbar.update(1)
            last = None
            for ds in evals_log:
                for metric in evals_log[ds]:
                    vals = evals_log[ds][metric]
                    if vals:
                        last = vals[-1]
            if last is not None and hasattr(self.pbar, "set_postfix"):
                self.pbar.set_postfix(eval=f"{last:.4f}", refresh=False)
            return False

        def after_training(self, model):
            self.pbar.close()
            return model

    return _TqdmTrialCallback()


# Lean per-head trainers (frozen Group-1 + given Group-2; no calibration).
def _sqrt_inverse_weights(y_train: pd.Series, n_classes: int,
                          exponent: float) -> pd.Series:
    """weight = (N / (K * count[c])) ** exponent - identical to
    train_nurse_v3.train_model (Group-1 frozen at exponent=0.5)."""
    class_counts = y_train.value_counts()
    total = len(y_train)
    return y_train.map(lambda x: (total / (n_classes * class_counts[x])) ** exponent)


def train_department(X_tr, y_tr, X_val, y_val, group2, gcfg, n_classes, desc):
    """Train one department model (frozen multiclass G1 + given Group-2).
    Returns the fitted (uncalibrated) XGBClassifier.

    The Group-1 sweep injects a candidate class_weight_exponent via
    gcfg["dept_cw_exponent"]; Group-2 callers leave it absent -> frozen 0.5."""
    cw_exp = gcfg.get("dept_cw_exponent", DEPT_CW_EXPONENT)
    sample_weights = _sqrt_inverse_weights(y_tr, n_classes, cw_exp)
    cb = _make_trial_progress_callback(gcfg["dept_n_estimators"], desc)
    model = XGBClassifier(
        n_estimators=gcfg["dept_n_estimators"],
        learning_rate=gcfg["dept_learning_rate"],
        early_stopping_rounds=gcfg["dept_early_stopping"],
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
        device=XGB_DEVICE, tree_method=XGB_TREE_METHOD,
        callbacks=[cb] if cb is not None else None,
        **group2,
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weights,
              eval_set=[(X_val, y_val)], verbose=False)
    try:
        model.set_params(callbacks=None)
    except Exception:
        pass
    return model


def train_disposition(X_tr, y_tr, X_val, y_val, group2, gcfg, desc):
    """Train one disposition model (frozen binary G1 + given Group-2). Binary
    admit/discharge with scale_pos_weight=(N_neg/N_pos)**exponent; the live
    exponent is 0.5 (sqrt). The Group-1 sweep injects a candidate exponent via
    gcfg["disp_scale_pos_weight_exponent"]; absent -> frozen sqrt."""
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    exponent = gcfg.get("disp_scale_pos_weight_exponent", 0.5)
    scale = float((neg / max(pos, 1)) ** exponent)
    cb = _make_trial_progress_callback(gcfg["disp_n_estimators"], desc)
    model = XGBClassifier(
        n_estimators=gcfg["disp_n_estimators"],
        learning_rate=gcfg["disp_learning_rate"],
        early_stopping_rounds=gcfg["disp_early_stopping"],
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale,
        random_state=42, n_jobs=-1, verbosity=0,
        device=XGB_DEVICE, tree_method=XGB_TREE_METHOD,
        callbacks=[cb] if cb is not None else None,
        **group2,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    try:
        model.set_params(callbacks=None)
    except Exception:
        pass
    return model


def _predict_labels(model, X) -> np.ndarray:
    """argmax over predict_proba - robust to the XGBoost GPU multi:softprob
    quirk where predict() can return a 2-D matrix (same helper as nurse_v3)."""
    return np.argmax(model.predict_proba(X), axis=1)


# Metrics
def department_metrics(y_true, y_pred, labels=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
    }
    if labels is not None:
        rec = recall_score(y_true, y_pred, labels=labels, average=None,
                           zero_division=0)
        out["per_class_recall"] = {str(c): float(r) for c, r in zip(labels, rec)}
    return out


def _ece(y, p, n_bins=10) -> float:
    """10-bin Expected Calibration Error (copied from
    train_doctor_disposition.train_disposition._ece)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def disposition_metrics(y_true, y_pred, y_prob, full: bool = False) -> dict:
    """Binary admit/discharge metrics. `under_rate` = admit predicted as
    discharge (the constrained, dangerous error). `full=True` adds Brier/ECE/
    over-triage/sensitivity/specificity for the report stage."""
    yt = np.asarray(y_true)
    out = {
        "accuracy": float(accuracy_score(yt, y_pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(yt, y_prob))
    except Exception:
        out["roc_auc"] = None
    cm = confusion_matrix(yt, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["under_rate"] = float(fn / max(fn + tp, 1))   # admit -> discharge
    out["over_rate"] = float(fp / max(tn + fp, 1))    # discharge -> admit
    if full:
        out["brier"] = float(brier_score_loss(yt, y_prob))
        out["ece"] = _ece(yt, np.asarray(y_prob))
        out["sensitivity"] = float(tp / max(tp + fn, 1))
        out["specificity"] = float(tn / max(tn + fp, 1))
    return out


# Isotonic calibration (A4 pattern; report stage only).
def _calibrate(raw, X_cal, y_cal):
    if _HAS_FROZEN:
        cal = CalibratedClassifierCV(FrozenEstimator(raw), method="isotonic",
                                     cv=None)
    else:
        cal = CalibratedClassifierCV(raw, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


# Feature caches
# Department: reuse the SAME cache the older write-mode diagnosis sweep
# (scripts/tune_doctor_v3.py) builds - `data/derived/tune_cache/`. Identical
# filenames + columns so the parquet is shared and built only once across both
# sweeps. If absent, build it from the nurse_v3 pipeline.
DEPT_CACHE_DIR = DERIVED_DIR / "tune_cache"


def load_or_build_dept_cache(cache_dir: Path, rebuild: bool, subsample: int | None):
    """Return (X, y_diag, y_dept, diagnosis_labels, department_labels).
    Mirrors scripts/tune_doctor_v3.load_or_build_cache exactly."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fx = cache_dir / "features.parquet"
    fy = cache_dir / "targets.parquet"
    fmeta = cache_dir / "meta.json"

    if not rebuild and fx.exists() and fy.exists() and fmeta.exists():
        print(f"[cache] Loading department features from {cache_dir}")
        X = pd.read_parquet(fx)
        targets = pd.read_parquet(fy)
        meta = json.loads(fmeta.read_text())
        return (X, targets["y_diag"], targets["y_dept"],
                meta["diagnosis_labels"], meta["department_labels"])

    print("[cache] Building department feature matrix from raw data "
          "(slow: parses discharge.csv)")
    from proiect_licenta.training.train_nurse_v3 import (
        load_and_clean_data, build_features, CATCH_ALL_LABEL,
    )
    df = load_and_clean_data()
    if subsample and len(df) > subsample:
        df, _ = train_test_split(
            df, train_size=subsample, random_state=7,
            stratify=df["diagnosis_group"],
        )
        print(f"[cache] Subsampled to {len(df):,} rows (smoke).")
    features = build_features(df)

    diagnosis_labels = sorted(df["diagnosis_group"].unique())
    department_labels = sorted(df["service_group"].unique())
    assert CATCH_ALL_LABEL not in diagnosis_labels
    diag_map = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_map = {l: i for i, l in enumerate(department_labels)}
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True).astype("int32")
    y_dept = df["service_group"].map(dept_map).reset_index(drop=True).astype("int32")

    for c in features.columns:
        if features[c].dtype == "float64":
            features[c] = features[c].astype("float32")

    # Always persist - smoke builds go to their own throwaway `*_smoke` dir
    # (set in main()), so caching them just makes repeat smoke runs fast too.
    features.to_parquet(fx, index=False)
    pd.DataFrame({"y_diag": y_diag, "y_dept": y_dept}).to_parquet(fy, index=False)
    fmeta.write_text(json.dumps({
        "diagnosis_labels": diagnosis_labels,
        "department_labels": department_labels,
        "built_at": datetime.now().isoformat(),
        "n_rows": int(len(features)), "n_cols": int(features.shape[1]),
        "subsample": subsample,
    }, indent=2))
    print(f"[cache] Wrote {features.shape[1]} cols, {len(features):,} rows "
          f"to {cache_dir}")
    return features, y_diag, y_dept, diagnosis_labels, department_labels


def load_or_build_disp_cache(cache_dir: Path, rebuild: bool, subsample: int | None):
    """Return (X, y) for the disposition head. y is the binary admit label.
    Built from the train_doctor_disposition pipeline (full 425K, soft cascade)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fx = cache_dir / "X.parquet"
    fy = cache_dir / "y.parquet"
    fmeta = cache_dir / "meta.json"

    if not rebuild and fx.exists() and fy.exists() and fmeta.exists():
        print(f"[cache] Loading disposition features from {cache_dir}")
        return pd.read_parquet(fx), pd.read_parquet(fy)["admitted"]

    print("[cache] Building disposition feature matrix from raw data "
          "(slow: full 425K, parses discharge.csv + runs triage v3 cascade)")
    from proiect_licenta.training.train_doctor_disposition import (
        load_and_clean_data, build_features,
    )
    df = load_and_clean_data()
    if subsample and len(df) > subsample:
        df, _ = train_test_split(
            df, train_size=subsample, random_state=7, stratify=df["admitted"],
        )
        print(f"[cache] Subsampled to {len(df):,} rows (smoke).")
    features = build_features(df)
    y = df["admitted"].astype(int).reset_index(drop=True)

    for c in features.columns:
        if features[c].dtype == "float64":
            features[c] = features[c].astype("float32")

    # Always persist - smoke builds go to their own throwaway `smoke/` subdir
    # (set in main()), so caching them just makes repeat smoke runs fast too.
    features.to_parquet(fx, index=False)
    pd.DataFrame({"admitted": y}).to_parquet(fy, index=False)
    fmeta.write_text(json.dumps({
        "built_at": datetime.now().isoformat(),
        "n_rows": int(len(features)), "n_cols": int(features.shape[1]),
        "admit_rate": float(y.mean()), "subsample": subsample,
    }, indent=2))
    print(f"[cache] Wrote {features.shape[1]} cols, {len(features):,} rows "
          f"to {cache_dir}")
    return features, y


# Incremental JSON log - rewritten after every completed trial (atomic).
# (copied from tune_triage_v3.py)
def _atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def best_feasible_trial(study):
    feas = [
        t for t in study.trials
        if t.state.name == "COMPLETE"
        and t.user_attrs.get("constraint", [1.0])[0] <= 1e-9
    ]
    return max(feas, key=lambda t: t.value) if feas else None


def _dept_block(study, best) -> dict:
    return {
        "study_name": study.study_name,
        "best_trial": int(best.number),
        "objective": "macro-F1 across department classes",
        "best_macro_f1": float(best.value),
        "best_accuracy": float(best.user_attrs.get("accuracy", -1)),
        "best_iteration": int(best.user_attrs.get("best_iteration", -1)),
        "n_trials": len(study.trials),
        "params": best.params,
        "written_at": datetime.now().isoformat(),
    }


def _disp_block(study, best) -> dict:
    return {
        "study_name": study.study_name,
        "best_trial": int(best.number),
        "objective": "ROC AUC s.t. under_rate <= incumbent",
        "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
        "best_roc_auc": float(best.value),
        "best_accuracy": float(best.user_attrs.get("accuracy", -1)),
        "best_under_rate": float(best.user_attrs.get("under_rate", -1)),
        "best_iteration": int(best.user_attrs.get("best_iteration", -1)),
        "n_trials": len(study.trials),
        "params": best.params,
        "written_at": datetime.now().isoformat(),
    }


def make_trial_logger(log_path: Path, stage: str, out_dir: Path):
    """Optuna callback. After every trial it (1) rewrites the human-readable
    per-trial JSON log and (2) re-writes the best-config block into
    tuned_params_doctor.json (interruption-safe; mirrors tune_triage_v3)."""
    def _log(study, trial):
        rows = []
        for t in study.trials:
            if t.state.name != "COMPLETE":
                continue
            ua = t.user_attrs
            row = {"number": t.number, "value": t.value, "params": t.params}
            for k in ("macro_f1", "accuracy", "roc_auc", "under_rate",
                      "over_rate", "best_iteration"):
                if k in ua:
                    row[k] = ua[k]
            if "constraint" in ua:
                row["feasible"] = bool(ua["constraint"][0] <= 1e-9)
            rows.append(row)
        payload = {
            "stage": stage,
            "study_name": study.study_name,
            "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
            "n_complete": len(rows),
            "updated_at": datetime.now().isoformat(),
            "trials": rows,
        }
        _atomic_write_json(log_path, payload)

        # Persist the best-config block every trial (interruption-safe).
        if stage == "department":
            try:
                best = study.best_trial
            except ValueError:
                best = None
            if best is not None:
                _update_tuned_params(out_dir, "department",
                                     _dept_block(study, best), quiet=True)
        elif stage == "disposition":
            best = best_feasible_trial(study)
            if best is not None:
                _update_tuned_params(out_dir, "disposition",
                                     _disp_block(study, best), quiet=True)

        if trial.state.name == "COMPLETE":
            ua = trial.user_attrs
            if "macro_f1" in ua:
                extra = (f"macro_f1={ua['macro_f1']:.4f} "
                         f"acc={ua.get('accuracy', 0)*100:.2f}%")
            elif "roc_auc" in ua:
                feas = "feasible" if ua.get("constraint", [1])[0] <= 1e-9 else "INFEASIBLE"
                extra = (f"auc={ua['roc_auc']:.4f} "
                         f"under={ua['under_rate']*100:.2f}% [{feas}]")
            else:
                extra = ""
            print(f"  trial #{trial.number:3d} done | {extra} "
                  f"best_iter={ua.get('best_iteration', '?')}", flush=True)
    return _log


# tuned_params_doctor.json read/update (preserve sibling blocks across stages)
def _tuned_path(out_dir: Path, fname: str = "tuned_params_doctor.json") -> Path:
    return out_dir / fname


def _read_tuned_params(out_dir: Path, fname: str = "tuned_params_doctor.json") -> dict | None:
    p = _tuned_path(out_dir, fname)
    return json.loads(p.read_text()) if p.exists() else None


def _update_tuned_params(out_dir: Path, key: str, block: dict, quiet: bool = False,
                         fname: str = "tuned_params_doctor.json"):
    data = _read_tuned_params(out_dir, fname) or {}
    data[key] = block
    _atomic_write_json(_tuned_path(out_dir, fname), data)
    if not quiet:
        print(f"  Updated {_tuned_path(out_dir, fname)} ['{key}']")


# Group-1 tuned-params live in a sibling file so the two studies never collide.
G1_TUNED_FNAME = "tuned_params_doctor_g1.json"


# Diagnosis cascade (department head needs the 13 diag_proba_* columns)
def _build_cascade_diag_model(X_tr, y_diag_tr, X_val, y_diag_val,
                              diagnosis_labels, gcfg, cache_path=None,
                              rebuild=False):
    """Train ONE diagnosis model for the cascade (reused across all department
    trials). Uses the existing write-mode tuned diagnosis params if present
    (artifacts/doctor/v3/tuned_params.json), else hand-picked defaults - exactly
    as train_nurse_v3.main() does.

    The fitted model is cached to `cache_path` (joblib) so repeated department
    chunks across Colab sessions don't retrain it. `--rebuild-features` (passed
    through as `rebuild`) forces a fresh fit (use it if tuned_params.json or the
    feature cache changed)."""
    if cache_path is not None and cache_path.exists() and not rebuild:
        print(f"  Cascade: loading cached diagnosis model from {cache_path.name}")
        return joblib.load(cache_path)

    from proiect_licenta.training.train_nurse_v3 import train_model
    tuned_path = DOCTOR_V3_DIR / "tuned_params.json"
    tuned = None
    if tuned_path.exists():
        try:
            tuned = json.loads(tuned_path.read_text())
            print(f"  Cascade: using tuned diagnosis params "
                  f"(trial #{tuned.get('best_trial', '?')}, "
                  f"macro_f1={tuned.get('best_macro_f1', '?')})")
        except Exception as e:
            print(f"  Cascade: could not parse {tuned_path}: {e}. Using defaults.")
    model = train_model(X_tr, y_diag_tr, X_val, y_diag_val,
                        diagnosis_labels, "DIAGNOSIS (cascade source)",
                        tuned_params=tuned)
    if cache_path is not None:
        try:
            model.set_params(callbacks=None)  # detach any tqdm bar before pickle
        except Exception:
            pass
        joblib.dump(model, cache_path)
        print(f"  Cascade: cached diagnosis model to {cache_path.name}")
    return model


# Stage: department (macro-F1)
def run_department_stage(args, gcfg, optuna, TPESampler, out_dir, cache_dir):
    from proiect_licenta.training.train_nurse_v3 import (
        build_diag_cascade_cols, _attach_diag_cascade,
    )
    X, y_diag, y_dept, diagnosis_labels, department_labels = load_or_build_dept_cache(
        cache_dir, rebuild=args.rebuild_features,
        subsample=20000 if args.smoke else None,
    )
    n_classes = len(department_labels)

    # Outer split (matches train_nurse_v3: stratify on diagnosis).
    X_train, X_test, yd_tr_all, yd_te_all, ydept_tr, ydept_te = train_test_split(
        X, y_diag, y_dept, test_size=0.2, random_state=42, stratify=y_diag,
    )
    # Inner split for tuning (stratify on department).
    X_in_tr, X_in_val, yd_in_tr, yd_in_val, ydept_in_tr, ydept_in_val = \
        train_test_split(X_train, yd_tr_all, ydept_tr, test_size=0.2,
                         random_state=1, stratify=ydept_tr)
    print(f"  Inner split (tuning): train={len(X_in_tr):,}, val={len(X_in_val):,}")

    # Build the diagnosis cascade ONCE (fixed across all department trials);
    # cache it so later department chunks reload instead of retraining.
    diag_model = _build_cascade_diag_model(
        X_in_tr, yd_in_tr, X_in_val, yd_in_val, diagnosis_labels, gcfg,
        cache_path=out_dir / "cascade_diag_model.joblib",
        rebuild=args.rebuild_features,
    )
    cascade_cols = build_diag_cascade_cols(diagnosis_labels)
    X_in_tr_c = _attach_diag_cascade(X_in_tr, diag_model, cascade_cols)
    X_in_val_c = _attach_diag_cascade(X_in_val, diag_model, cascade_cols)
    print(f"  Attached {len(cascade_cols)} diagnosis-softmax cascade cols.")

    def objective(trial):
        group2 = suggest_group2(trial)
        print(f"\n  >>> trial #{trial.number:3d} | "
              f"depth={group2['max_depth']} sub={group2['subsample']:.2f} "
              f"mcw={group2['min_child_weight']} gamma={group2['gamma']:.3f} "
              f"a={group2['reg_alpha']:.3f} l={group2['reg_lambda']:.3f}", flush=True)
        model = train_department(X_in_tr_c, ydept_in_tr, X_in_val_c, ydept_in_val,
                                 group2, gcfg, n_classes,
                                 f"    trial #{trial.number} dept")
        y_pred = _predict_labels(model, X_in_val_c)
        m = department_metrics(ydept_in_val, y_pred)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return m["macro_f1"]

    _ensure_warn_shim()  # crewai (imported during cache build) clobbers the shim
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name or DEPT_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP2)
        print("  Enqueued incumbent Group-2 config as trial 0.")

    log_path = out_dir / "tuning_log_department.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger(log_path, "department", out_dir)],
                   gc_after_trial=True)

    best = study.best_trial
    print(f"  Best department trial: #{best.number}  macro_f1={best.value:.4f}  "
          f"acc={best.user_attrs['accuracy']*100:.2f}%")
    for k, v in best.params.items():
        print(f"    {k:18s} = {v}")
    _update_tuned_params(out_dir, "department", _dept_block(study, best))


# Stage: disposition (constrained ROC AUC)
def run_disposition_stage(args, gcfg, optuna, TPESampler, out_dir, cache_dir):
    X, y = load_or_build_disp_cache(
        cache_dir, rebuild=args.rebuild_features,
        subsample=20000 if args.smoke else None,
    )
    y = y.reset_index(drop=True)
    # Outer split (matches train_doctor_disposition: stratify on admit).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    # Inner split for tuning.
    Xd_tr, Xd_val, yd_tr, yd_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1, stratify=y_train,
    )
    print(f"  Inner split (tuning): train={len(Xd_tr):,}, val={len(Xd_val):,}")

    def objective(trial):
        group2 = suggest_group2(trial)
        print(f"\n  >>> trial #{trial.number:3d} | "
              f"depth={group2['max_depth']} sub={group2['subsample']:.2f} "
              f"mcw={group2['min_child_weight']}", flush=True)
        model = train_disposition(Xd_tr, yd_tr, Xd_val, yd_val, group2, gcfg,
                                  f"    trial #{trial.number} disp")
        y_prob = model.predict_proba(Xd_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = disposition_metrics(yd_val, y_pred, y_prob)

        # Trial 0 (the enqueued incumbent) defines the in-sample baseline.
        baseline = trial.study.user_attrs.get("baseline_under_rate")
        if baseline is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
            baseline = m["under_rate"]

        for k, v in m.items():
            if v is not None:
                trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        # <= 0 means feasible (under-triage no worse than the incumbent).
        trial.set_user_attr("constraint", [m["under_rate"] - baseline])
        return m["roc_auc"] if m["roc_auc"] is not None else 0.0

    def constraints_func(trial):
        return trial.user_attrs.get("constraint", [0.0])

    _ensure_warn_shim()  # crewai (imported during cache build) clobbers the shim
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name or DISP_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True,
                           constraints_func=constraints_func),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP2)  # incumbent = trial 0 = baseline
        print("  Enqueued incumbent Group-2 config as trial 0 (defines baseline).")

    log_path = out_dir / "tuning_log_disposition.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger(log_path, "disposition", out_dir)],
                   gc_after_trial=True)

    best = best_feasible_trial(study)
    if best is None:
        print("  No feasible trial yet (none matched the under-triage baseline).")
        return
    print(f"  Best FEASIBLE disposition trial: #{best.number}  "
          f"auc={best.value:.4f}  under={best.user_attrs['under_rate']*100:.2f}%  "
          f"(baseline under={study.user_attrs['baseline_under_rate']*100:.2f}%)")
    for k, v in best.params.items():
        print(f"    {k:18s} = {v}")
    _update_tuned_params(out_dir, "disposition", _disp_block(study, best))


# Stage: report - incumbent vs best on the OUTER test split.
def run_report_stage(args, gcfg, optuna, out_dir, dept_cache_dir, disp_cache_dir):
    tuned = _read_tuned_params(out_dir) or {}
    results = {"generated_at": datetime.now().isoformat()}

    # Department
    if tuned.get("department"):
        results["department"] = _report_department(args, gcfg, tuned, dept_cache_dir)
    else:
        print("  [department] no tuned block yet - skipping.")

    # Disposition
    if tuned.get("disposition"):
        results["disposition"] = _report_disposition(args, gcfg, tuned, disp_cache_dir)
    else:
        print("  [disposition] no tuned block yet - skipping.")

    out = out_dir / "doctor_hpo_results.json"
    _atomic_write_json(out, results)
    print(f"\n  Wrote {out}")
    _print_report_summary(results)


def _report_department(args, gcfg, tuned, cache_dir):
    from proiect_licenta.training.train_nurse_v3 import (
        build_diag_cascade_cols, _attach_diag_cascade,
    )
    live_diag = DOCTOR_V3_DIR / "diagnosis_model.joblib"
    live_dept = DOCTOR_V3_DIR / "department_model.joblib"
    if not live_diag.exists() or not live_dept.exists():
        print(f"  ERROR: live diagnosis/department models not found in "
              f"{DOCTOR_V3_DIR}.", file=sys.stderr)
        return {"error": "live department/diagnosis model missing"}

    X, y_diag, y_dept, diagnosis_labels, department_labels = load_or_build_dept_cache(
        cache_dir, rebuild=False, subsample=None,
    )
    n_classes = len(department_labels)
    X_train, X_test, _, _, ydept_tr, ydept_te = train_test_split(
        X, y_diag, y_dept, test_size=0.2, random_state=42, stratify=y_diag,
    )

    # Use the LIVE diagnosis model for the cascade in BOTH arms so the
    # comparison isolates the department-Group-2 effect.
    diag_model = joblib.load(live_diag)
    cascade_cols = build_diag_cascade_cols(diagnosis_labels)
    X_train_c = _attach_diag_cascade(X_train, diag_model, cascade_cols)
    X_test_c = _attach_diag_cascade(X_test, diag_model, cascade_cols)

    dept_label_ids = list(range(n_classes))

    print("\n  [department] incumbent (live calibrated model) on outer-test ...")
    inc_model = joblib.load(live_dept)
    inc_pred = _predict_labels(inc_model, X_test_c)
    inc_m = department_metrics(ydept_te, inc_pred, labels=dept_label_ids)

    print("  [department] best config: retrain + calibrate on outer-train ...")
    g2 = tuned["department"]["params"]
    # 10% calibration holdout (A4), then isotonic-calibrate.
    Xc_fit, Xc_cal, yc_fit, yc_cal = train_test_split(
        X_train_c, ydept_tr, test_size=0.1, random_state=42, stratify=ydept_tr,
    )
    best_raw = train_department(Xc_fit, yc_fit, X_test_c, ydept_te, g2, gcfg,
                                n_classes, "    department (best)")
    best_cal = _calibrate(best_raw, Xc_cal, yc_cal)
    best_pred = _predict_labels(best_cal, X_test_c)
    best_m = department_metrics(ydept_te, best_pred, labels=dept_label_ids)

    return {
        "incumbent": inc_m,
        "best": best_m,
        "delta_macro_f1": best_m["macro_f1"] - inc_m["macro_f1"],
        "delta_accuracy": best_m["accuracy"] - inc_m["accuracy"],
        "test_rows": int(len(X_test_c)),
        "best_params": g2,
    }


def _report_disposition(args, gcfg, tuned, cache_dir):
    live_disp = DOCTOR_V3_DIR / "disposition_model.joblib"
    if not live_disp.exists():
        print(f"  ERROR: live disposition model not found in {DOCTOR_V3_DIR}.",
              file=sys.stderr)
        return {"error": "live disposition model missing"}

    X, y = load_or_build_disp_cache(cache_dir, rebuild=False, subsample=None)
    y = y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print("\n  [disposition] incumbent (live calibrated model) on outer-test ...")
    inc_model = joblib.load(live_disp)
    inc_prob = inc_model.predict_proba(X_test)[:, 1]
    inc_pred = (inc_prob >= 0.5).astype(int)
    inc_m = disposition_metrics(y_test, inc_pred, inc_prob, full=True)

    print("  [disposition] best config: retrain + calibrate on outer-train ...")
    g2 = tuned["disposition"]["params"]
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train,
    )
    best_raw = train_disposition(X_fit, y_fit, X_test, y_test, g2, gcfg,
                                 "    disposition (best)")
    best_cal = _calibrate(best_raw, X_cal, y_cal)
    best_prob = best_cal.predict_proba(X_test)[:, 1]
    best_pred = (best_prob >= 0.5).astype(int)
    best_m = disposition_metrics(y_test, best_pred, best_prob, full=True)

    out = {"incumbent": inc_m, "best": best_m, "test_rows": int(len(X_test)),
           "best_params": g2}
    if inc_m.get("roc_auc") and best_m.get("roc_auc"):
        out["delta_roc_auc"] = best_m["roc_auc"] - inc_m["roc_auc"]
    out["delta_under_rate"] = best_m["under_rate"] - inc_m["under_rate"]
    return out


def _print_report_summary(results):
    d = results.get("department")
    if d and "incumbent" in d:
        inc, best = d["incumbent"], d["best"]
        print("\n  ── Department (outer-test) ──")
        print(f"    incumbent:  macro_f1={inc['macro_f1']:.4f}  "
              f"acc={inc['accuracy']*100:.2f}%")
        print(f"    best:       macro_f1={best['macro_f1']:.4f}  "
              f"acc={best['accuracy']*100:.2f}%")
        print(f"    delta:      macro_f1={d['delta_macro_f1']:+.4f}  "
              f"acc={d['delta_accuracy']*100:+.2f}pp")
    p = results.get("disposition")
    if p and "incumbent" in p:
        inc, best = p["incumbent"], p["best"]
        print("  ── Disposition (outer-test) ──")
        print(f"    incumbent:  auc={inc['roc_auc']}  acc={inc['accuracy']*100:.2f}%  "
              f"under={inc['under_rate']*100:.2f}%  ece={inc['ece']:.4f}")
        print(f"    best:       auc={best['roc_auc']}  acc={best['accuracy']*100:.2f}%  "
              f"under={best['under_rate']*100:.2f}%  ece={best['ece']:.4f}")
        if "delta_roc_auc" in p:
            print(f"    delta:      auc={p['delta_roc_auc']:+.4f}  "
                  f"under={p['delta_under_rate']*100:+.2f}pp")


# Group-1 single-objective sweep - the per-head cost/weighting config.
# Same objectives/constraints as Group-2 (department: macro-F1; disposition:
# constrained ROC AUC); Group-2 frozen at FROZEN_GROUP2; the SEARCH varies the
# Group-1 knobs (lr, n_estimators, class-weight / scale_pos_weight exponent).
def _g1_dept_block(study, best) -> dict:
    return {
        "study_name": study.study_name,
        "mode": "single-objective",
        "objective": "macro-F1 across department classes (Group-1)",
        "best_trial": int(best.number),
        "best_macro_f1": float(best.value),
        "best_accuracy": float(best.user_attrs.get("accuracy", -1)),
        "best_iteration": int(best.user_attrs.get("best_iteration", -1)),
        "params": best.params,
        "selected": {"trial": int(best.number), "params": best.params,
                     "metrics": {"macro_f1": float(best.value),
                                 "accuracy": float(best.user_attrs.get("accuracy", -1))}},
        "n_trials": len(study.trials),
        "written_at": datetime.now().isoformat(),
    }


def _g1_disp_block(study, best) -> dict:
    return {
        "study_name": study.study_name,
        "mode": "single-objective (constrained)",
        "objective": "ROC AUC s.t. under_rate <= incumbent (Group-1)",
        "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
        "best_trial": int(best.number),
        "best_roc_auc": float(best.value),
        "best_accuracy": float(best.user_attrs.get("accuracy", -1)),
        "best_under_rate": float(best.user_attrs.get("under_rate", -1)),
        "best_iteration": int(best.user_attrs.get("best_iteration", -1)),
        "params": best.params,
        "selected": {"trial": int(best.number), "params": best.params,
                     "metrics": {"roc_auc": float(best.value),
                                 "under_rate": float(best.user_attrs.get("under_rate", -1))}},
        "n_trials": len(study.trials),
        "written_at": datetime.now().isoformat(),
    }


def make_trial_logger_g1(log_path: Path, stage: str, out_dir: Path):
    """Group-1 Optuna callback - same shape as make_trial_logger but persists to
    tuned_params_doctor_g1.json with the Group-1 block builders. trial.params
    carries the Group-1 knobs (lr/n_estimators/exponent)."""
    def _log(study, trial):
        rows = []
        for t in study.trials:
            if t.state.name != "COMPLETE":
                continue
            ua = t.user_attrs
            row = {"number": t.number, "value": t.value, "params": t.params}
            for k in ("macro_f1", "accuracy", "roc_auc", "under_rate",
                      "over_rate", "best_iteration"):
                if k in ua:
                    row[k] = ua[k]
            if "constraint" in ua:
                row["feasible"] = bool(ua["constraint"][0] <= 1e-9)
            rows.append(row)
        payload = {
            "stage": stage, "study_name": study.study_name, "mode": "Group-1",
            "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
            "n_complete": len(rows),
            "updated_at": datetime.now().isoformat(),
            "trials": rows,
        }
        _atomic_write_json(log_path, payload)

        if stage == "department":
            try:
                best = study.best_trial
            except ValueError:
                best = None
            if best is not None:
                _update_tuned_params(out_dir, "department", _g1_dept_block(study, best),
                                     quiet=True, fname=G1_TUNED_FNAME)
        elif stage == "disposition":
            best = best_feasible_trial(study)
            if best is not None:
                _update_tuned_params(out_dir, "disposition", _g1_disp_block(study, best),
                                     quiet=True, fname=G1_TUNED_FNAME)

        if trial.state.name == "COMPLETE":
            ua, p = trial.user_attrs, trial.params
            if "macro_f1" in ua:
                extra = f"macro_f1={ua['macro_f1']:.4f} acc={ua.get('accuracy', 0)*100:.2f}%"
            elif "roc_auc" in ua:
                feas = "feasible" if ua.get("constraint", [1])[0] <= 1e-9 else "INFEASIBLE"
                extra = f"auc={ua['roc_auc']:.4f} under={ua['under_rate']*100:.2f}% [{feas}]"
            else:
                extra = ""
            print(f"  trial #{trial.number:3d} done | {extra} | "
                  f"lr={p.get('learning_rate')} n_est={p.get('n_estimators')} "
                  f"exp={p.get('class_weight_exponent', p.get('scale_pos_weight_exponent'))} "
                  f"best_iter={ua.get('best_iteration', '?')}", flush=True)
    return _log


def run_department_stage_g1(args, gcfg, optuna, TPESampler, out_dir, cache_dir):
    """Group-1 department: maximize macro-F1 over (lr, n_estimators,
    class_weight_exponent). Group-2 frozen; diagnosis cascade reused."""
    from proiect_licenta.training.train_nurse_v3 import (
        build_diag_cascade_cols, _attach_diag_cascade,
    )
    X, y_diag, y_dept, diagnosis_labels, department_labels = load_or_build_dept_cache(
        cache_dir, rebuild=args.rebuild_features,
        subsample=20000 if args.smoke else None,
    )
    n_classes = len(department_labels)
    X_train, X_test, yd_tr_all, yd_te_all, ydept_tr, ydept_te = train_test_split(
        X, y_diag, y_dept, test_size=0.2, random_state=42, stratify=y_diag,
    )
    X_in_tr, X_in_val, yd_in_tr, yd_in_val, ydept_in_tr, ydept_in_val = \
        train_test_split(X_train, yd_tr_all, ydept_tr, test_size=0.2,
                         random_state=1, stratify=ydept_tr)
    print(f"  Inner split (tuning): train={len(X_in_tr):,}, val={len(X_in_val):,}")

    diag_model = _build_cascade_diag_model(
        X_in_tr, yd_in_tr, X_in_val, yd_in_val, diagnosis_labels, gcfg,
        cache_path=out_dir / "cascade_diag_model.joblib",
        rebuild=args.rebuild_features,
    )
    cascade_cols = build_diag_cascade_cols(diagnosis_labels)
    X_in_tr_c = _attach_diag_cascade(X_in_tr, diag_model, cascade_cols)
    X_in_val_c = _attach_diag_cascade(X_in_val, diag_model, cascade_cols)
    print(f"  Attached {len(cascade_cols)} diagnosis-softmax cascade cols.")

    def objective(trial):
        p = suggest_group1_department(trial)
        g = _g1_dept_gcfg(gcfg, p)
        print(f"\n  >>> trial #{trial.number:3d} | lr={p['learning_rate']:.4f} "
              f"n_est={p['n_estimators']} cw_exp={p['class_weight_exponent']:.2f}",
              flush=True)
        model = train_department(X_in_tr_c, ydept_in_tr, X_in_val_c, ydept_in_val,
                                 FROZEN_GROUP2, g, n_classes,
                                 f"    trial #{trial.number} dept")
        m = department_metrics(ydept_in_val, _predict_labels(model, X_in_val_c))
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return m["macro_f1"]

    _ensure_warn_shim()
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name or DEPT_G1_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP1_DEPT)
        print("  Enqueued incumbent Group-1 config as trial 0.")

    log_path = out_dir / "tuning_log_department_g1.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger_g1(log_path, "department", out_dir)],
                   gc_after_trial=True)
    best = study.best_trial
    print(f"  Best department Group-1 trial: #{best.number}  "
          f"macro_f1={best.value:.4f}  acc={best.user_attrs['accuracy']*100:.2f}%")
    for k, v in best.params.items():
        print(f"    {k:24s} = {v}")
    _update_tuned_params(out_dir, "department", _g1_dept_block(study, best),
                         fname=G1_TUNED_FNAME)


def run_disposition_stage_g1(args, gcfg, optuna, TPESampler, out_dir, cache_dir):
    """Group-1 disposition: maximize ROC AUC subject to under_rate <= incumbent,
    over (lr, n_estimators, scale_pos_weight_exponent). Group-2 frozen."""
    X, y = load_or_build_disp_cache(
        cache_dir, rebuild=args.rebuild_features,
        subsample=20000 if args.smoke else None,
    )
    y = y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    Xd_tr, Xd_val, yd_tr, yd_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1, stratify=y_train,
    )
    print(f"  Inner split (tuning): train={len(Xd_tr):,}, val={len(Xd_val):,}")

    def objective(trial):
        p = suggest_group1_disposition(trial)
        g = _g1_disp_gcfg(gcfg, p)
        print(f"\n  >>> trial #{trial.number:3d} | lr={p['learning_rate']:.4f} "
              f"n_est={p['n_estimators']} spw_exp={p['scale_pos_weight_exponent']:.2f}",
              flush=True)
        model = train_disposition(Xd_tr, yd_tr, Xd_val, yd_val, FROZEN_GROUP2, g,
                                  f"    trial #{trial.number} disp")
        y_prob = model.predict_proba(Xd_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = disposition_metrics(yd_val, y_pred, y_prob)
        baseline = trial.study.user_attrs.get("baseline_under_rate")
        if baseline is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
            baseline = m["under_rate"]
        for k, v in m.items():
            if v is not None:
                trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("constraint", [m["under_rate"] - baseline])
        return m["roc_auc"] if m["roc_auc"] is not None else 0.0

    def constraints_func(trial):
        return trial.user_attrs.get("constraint", [0.0])

    _ensure_warn_shim()
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name or DISP_G1_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True, constraints_func=constraints_func),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP1_DISP)
        print("  Enqueued incumbent Group-1 config as trial 0 (defines baseline).")

    log_path = out_dir / "tuning_log_disposition_g1.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger_g1(log_path, "disposition", out_dir)],
                   gc_after_trial=True)
    best = best_feasible_trial(study)
    if best is None:
        print("  No feasible trial yet (none matched the under-triage baseline).")
        return
    print(f"  Best FEASIBLE disposition Group-1 trial: #{best.number}  "
          f"auc={best.value:.4f}  under={best.user_attrs['under_rate']*100:.2f}%  "
          f"(baseline under={study.user_attrs['baseline_under_rate']*100:.2f}%)")
    for k, v in best.params.items():
        print(f"    {k:24s} = {v}")
    _update_tuned_params(out_dir, "disposition", _g1_disp_block(study, best),
                         fname=G1_TUNED_FNAME)


# Group-1 report (incumbent vs best on outer-test; live calibrated)
# Live operating point for the doctor disposition head (doctor_disposition_tool
# DECISION_THRESHOLD; tuned from 0.50 via benchmarks/sweep_disposition_threshold).
DISP_LIVE_THRESHOLD = 0.40
_OP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def run_report_stage_g1(args, gcfg, optuna, out_dir, dept_cache_dir, disp_cache_dir):
    tuned = _read_tuned_params(out_dir, G1_TUNED_FNAME) or {}
    results = {"generated_at": datetime.now().isoformat(), "mode": "Group-1"}
    if tuned.get("department"):
        results["department"] = _report_department_g1(args, gcfg, tuned, dept_cache_dir)
    else:
        print("  [department] no Group-1 block yet - skipping.")
    if tuned.get("disposition"):
        results["disposition"] = _report_disposition_g1(args, gcfg, tuned, disp_cache_dir)
    else:
        print("  [disposition] no Group-1 block yet - skipping.")
    out = out_dir / "doctor_hpo_g1_results.json"
    _atomic_write_json(out, results)
    print(f"\n  Wrote {out}")
    _print_report_summary_g1(results)


def _report_department_g1(args, gcfg, tuned, cache_dir):
    from proiect_licenta.training.train_nurse_v3 import (
        build_diag_cascade_cols, _attach_diag_cascade,
    )
    live_diag = DOCTOR_V3_DIR / "diagnosis_model.joblib"
    live_dept = DOCTOR_V3_DIR / "department_model.joblib"
    if not live_diag.exists() or not live_dept.exists():
        print(f"  ERROR: live diagnosis/department models not found in "
              f"{DOCTOR_V3_DIR}.", file=sys.stderr)
        return {"error": "live department/diagnosis model missing"}

    X, y_diag, y_dept, diagnosis_labels, department_labels = load_or_build_dept_cache(
        cache_dir, rebuild=False, subsample=None,
    )
    n_classes = len(department_labels)
    X_train, X_test, _, _, ydept_tr, ydept_te = train_test_split(
        X, y_diag, y_dept, test_size=0.2, random_state=42, stratify=y_diag,
    )
    diag_model = joblib.load(live_diag)
    cascade_cols = build_diag_cascade_cols(diagnosis_labels)
    X_train_c = _attach_diag_cascade(X_train, diag_model, cascade_cols)
    X_test_c = _attach_diag_cascade(X_test, diag_model, cascade_cols)
    dept_label_ids = list(range(n_classes))

    print("\n  [department] incumbent (live calibrated model) on outer-test ...")
    inc_model = joblib.load(live_dept)
    inc_m = department_metrics(ydept_te, _predict_labels(inc_model, X_test_c),
                               labels=dept_label_ids)

    print("  [department] best Group-1 config: retrain + calibrate on outer-train ...")
    sel = tuned["department"]["selected"]["params"]
    g = _g1_dept_gcfg(gcfg, sel)
    Xc_fit, Xc_cal, yc_fit, yc_cal = train_test_split(
        X_train_c, ydept_tr, test_size=0.1, random_state=42, stratify=ydept_tr,
    )
    best_raw = train_department(Xc_fit, yc_fit, X_test_c, ydept_te, FROZEN_GROUP2, g,
                                n_classes, "    department (best G1)")
    best_cal = _calibrate(best_raw, Xc_cal, yc_cal)
    best_m = department_metrics(ydept_te, _predict_labels(best_cal, X_test_c),
                                labels=dept_label_ids)
    return {
        "incumbent": inc_m, "best": best_m,
        "delta_macro_f1": best_m["macro_f1"] - inc_m["macro_f1"],
        "delta_accuracy": best_m["accuracy"] - inc_m["accuracy"],
        "test_rows": int(len(X_test_c)), "group1_params": sel,
    }


def _operating_point_table(y_test, inc_prob, best_prob) -> list:
    yt = np.asarray(y_test)
    rows = []
    for thr in _OP_THRESHOLDS:
        im = disposition_metrics(yt, (inc_prob >= thr).astype(int), inc_prob)
        bm = disposition_metrics(yt, (best_prob >= thr).astype(int), best_prob)
        rows.append({
            "threshold": thr,
            "incumbent": {k: im[k] for k in ("accuracy", "under_rate", "over_rate")},
            "best": {k: bm[k] for k in ("accuracy", "under_rate", "over_rate")},
        })
    return rows


def _report_disposition_g1(args, gcfg, tuned, cache_dir):
    live_disp = DOCTOR_V3_DIR / "disposition_model.joblib"
    if not live_disp.exists():
        print(f"  ERROR: live disposition model not found in {DOCTOR_V3_DIR}.",
              file=sys.stderr)
        return {"error": "live disposition model missing"}

    X, y = load_or_build_disp_cache(cache_dir, rebuild=False, subsample=None)
    y = y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    thr = DISP_LIVE_THRESHOLD

    print("\n  [disposition] incumbent (live calibrated model) on outer-test ...")
    inc_model = joblib.load(live_disp)
    inc_prob = inc_model.predict_proba(X_test)[:, 1]
    inc_m = disposition_metrics(y_test, (inc_prob >= thr).astype(int), inc_prob, full=True)

    print("  [disposition] best Group-1 config: retrain + calibrate on outer-train ...")
    sel = tuned["disposition"]["selected"]["params"]
    g = _g1_disp_gcfg(gcfg, sel)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train,
    )
    best_raw = train_disposition(X_fit, y_fit, X_test, y_test, FROZEN_GROUP2, g,
                                 "    disposition (best G1)")
    best_cal = _calibrate(best_raw, X_cal, y_cal)
    best_prob = best_cal.predict_proba(X_test)[:, 1]
    best_m = disposition_metrics(y_test, (best_prob >= thr).astype(int), best_prob, full=True)

    out = {
        "operating_point_threshold": thr,
        "incumbent": inc_m, "best": best_m,
        "operating_point_table": _operating_point_table(y_test, inc_prob, best_prob),
        "test_rows": int(len(X_test)), "group1_params": sel,
    }
    if inc_m.get("roc_auc") and best_m.get("roc_auc"):
        out["delta_roc_auc"] = best_m["roc_auc"] - inc_m["roc_auc"]
    out["delta_under_rate"] = best_m["under_rate"] - inc_m["under_rate"]
    return out


def _print_report_summary_g1(results):
    d = results.get("department")
    if d and "incumbent" in d:
        inc, best = d["incumbent"], d["best"]
        print("\n  ── Department Group-1 (outer-test) ──")
        print(f"    incumbent:  macro_f1={inc['macro_f1']:.4f}  acc={inc['accuracy']*100:.2f}%")
        print(f"    best G1:    macro_f1={best['macro_f1']:.4f}  acc={best['accuracy']*100:.2f}%")
        print(f"    delta:      macro_f1={d['delta_macro_f1']:+.4f}  "
              f"acc={d['delta_accuracy']*100:+.2f}pp")
    p = results.get("disposition")
    if p and "incumbent" in p:
        inc, best = p["incumbent"], p["best"]
        print(f"  ── Disposition Group-1 (outer-test, thr={p['operating_point_threshold']:.2f}) ──")
        print(f"    incumbent:  auc={inc['roc_auc']}  acc={inc['accuracy']*100:.2f}%  "
              f"under={inc['under_rate']*100:.2f}%  ece={inc['ece']:.4f}")
        print(f"    best G1:    auc={best['roc_auc']}  acc={best['accuracy']*100:.2f}%  "
              f"under={best['under_rate']*100:.2f}%  ece={best['ece']:.4f}")
        if "delta_roc_auc" in p:
            print(f"    delta:      auc={p['delta_roc_auc']:+.4f}  "
                  f"under={p['delta_under_rate']*100:+.2f}pp")


# Self-test (synthetic, local, no MIMIC / no GPU) - both heads.
def run_selftest():
    print("  SELFTEST - synthetic data, CPU, throwaway paths (both heads)")
    import optuna
    from optuna.samplers import TPESampler

    # Snapshot live-model mtimes (must be untouched afterwards).
    live_dept = DOCTOR_V3_DIR / "department_model.joblib"
    live_disp = DOCTOR_V3_DIR / "disposition_model.joblib"
    dept_mtime = live_dept.stat().st_mtime if live_dept.exists() else None
    disp_mtime = live_disp.stat().st_mtime if live_disp.exists() else None

    rng = np.random.default_rng(0)
    n, d = 1500, 24
    X = pd.DataFrame(rng.standard_normal((n, d)).astype("float32"),
                     columns=[f"f{i}" for i in range(d)])
    gcfg = {"dept_n_estimators": 10, "dept_learning_rate": 0.3,
            "dept_early_stopping": 5,
            "disp_n_estimators": 10, "disp_learning_rate": 0.3,
            "disp_early_stopping": 5}

    tmp = Path(tempfile.mkdtemp(prefix="doctor_hpo_selftest_"))
    storage = f"sqlite:///{tmp / 'selftest.db'}"

    # department (macro-F1, unconstrained)
    y_dept = pd.Series(rng.integers(0, 4, n))
    Xtr, Xval, ytr, yval = train_test_split(X, y_dept, test_size=0.3, random_state=1)
    dept_log = tmp / "tuning_log_department.json"

    def dept_obj(trial):
        g = suggest_group2(trial)
        model = train_department(Xtr, ytr, Xval, yval, g, gcfg, 4, "selftest-dept")
        m = department_metrics(yval, _predict_labels(model, Xval))
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return m["macro_f1"]

    dept_study = optuna.create_study(direction="maximize", study_name="st_dept",
                                     storage=storage, load_if_exists=True,
                                     sampler=TPESampler(seed=42))
    dept_study.enqueue_trial(FROZEN_GROUP2)
    dept_study.optimize(dept_obj, n_trials=1,
                        callbacks=[make_trial_logger(dept_log, "department", tmp)])
    assert dept_log.exists(), "department JSON log not written"
    dl = json.loads(dept_log.read_text())
    assert dl["n_complete"] >= 1, "no completed department trial"
    assert "macro_f1" in dl["trials"][0], "macro_f1 missing from dept log"

    # disposition (constrained ROC AUC)
    y_disp = pd.Series(rng.integers(0, 2, n))
    Xtr2, Xval2, ytr2, yval2 = train_test_split(X, y_disp, test_size=0.3, random_state=1)
    disp_log = tmp / "tuning_log_disposition.json"

    def disp_obj(trial):
        g = suggest_group2(trial)
        model = train_disposition(Xtr2, ytr2, Xval2, yval2, g, gcfg, "selftest-disp")
        prob = model.predict_proba(Xval2)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = disposition_metrics(yval2, pred, prob)
        base = trial.study.user_attrs.get("baseline_under_rate")
        if base is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
            base = m["under_rate"]
        for k, v in m.items():
            if v is not None:
                trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("constraint", [m["under_rate"] - base])
        return m["roc_auc"] if m["roc_auc"] is not None else 0.0

    sampler = TPESampler(seed=42,
                         constraints_func=lambda t: t.user_attrs.get("constraint", [0.0]))
    disp_study = optuna.create_study(direction="maximize", study_name="st_disp",
                                     storage=storage, load_if_exists=True,
                                     sampler=sampler)
    disp_study.enqueue_trial(FROZEN_GROUP2)
    disp_study.optimize(disp_obj, n_trials=1,
                        callbacks=[make_trial_logger(disp_log, "disposition", tmp)])
    assert disp_log.exists(), "disposition JSON log not written"
    pl = json.loads(disp_log.read_text())
    assert pl["n_complete"] >= 1, "no completed disposition trial"
    assert "feasible" in pl["trials"][0], "constraint/feasible flag missing"

    # resume + reporting-only guarantees
    reopened = optuna.load_study(study_name="st_disp", storage=storage)
    assert len(reopened.trials) >= 1, "study did not persist for resume"
    tp = _read_tuned_params(tmp) or {}
    assert "department" in tp and "disposition" in tp, "tuned_params blocks missing"
    if dept_mtime is not None:
        assert live_dept.stat().st_mtime == dept_mtime, "live department model modified!"
    if disp_mtime is not None:
        assert live_disp.stat().st_mtime == disp_mtime, "live disposition model modified!"

    print(f"  study DB:   {tmp / 'selftest.db'}")
    print(f"  dept log:   {dept_log}  ({dl['n_complete']} trial)")
    print(f"  disp log:   {disp_log}  ({pl['n_complete']} trial, "
          f"baseline_under={disp_study.user_attrs['baseline_under_rate']:.4f})")
    print(f"  tuned_params blocks: {sorted(tp.keys())}")
    print(f"  live models untouched: dept={'n/a' if dept_mtime is None else 'yes'}, "
          f"disp={'n/a' if disp_mtime is None else 'yes'}")
    print("\n  SELFTEST PASSED")


def run_selftest_g1():
    """Synthetic CPU plumbing test for the Group-1 path (both heads, single-
    objective): studies build, trials complete, tuned_params_doctor_g1.json gets
    'selected' blocks, constraint flag present, live models untouched."""
    print("  SELFTEST (Group-1) - synthetic data, CPU, throwaway paths (both heads)")
    import optuna
    from optuna.samplers import TPESampler

    live_dept = DOCTOR_V3_DIR / "department_model.joblib"
    live_disp = DOCTOR_V3_DIR / "disposition_model.joblib"
    dept_mtime = live_dept.stat().st_mtime if live_dept.exists() else None
    disp_mtime = live_disp.stat().st_mtime if live_disp.exists() else None

    rng = np.random.default_rng(0)
    n, d = 1500, 24
    X = pd.DataFrame(rng.standard_normal((n, d)).astype("float32"),
                     columns=[f"f{i}" for i in range(d)])
    gcfg = {"dept_n_estimators": 10, "dept_learning_rate": 0.3, "dept_early_stopping": 5,
            "disp_n_estimators": 10, "disp_learning_rate": 0.3, "disp_early_stopping": 5}
    tmp = Path(tempfile.mkdtemp(prefix="doctor_hpo_g1_selftest_"))
    storage = f"sqlite:///{tmp / 'selftest_g1.db'}"

    # department (macro-F1, Group-1)
    y_dept = pd.Series(rng.integers(0, 4, n))
    Xtr, Xval, ytr, yval = train_test_split(X, y_dept, test_size=0.3, random_state=1)
    dept_log = tmp / "tuning_log_department_g1.json"

    def dept_obj(trial):
        p = suggest_group1_department(trial)
        g = _g1_dept_gcfg(gcfg, p)
        model = train_department(Xtr, ytr, Xval, yval, FROZEN_GROUP2, g, 4, "st-dept-g1")
        m = department_metrics(yval, _predict_labels(model, Xval))
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return m["macro_f1"]

    dept_study = optuna.create_study(direction="maximize", study_name="st_dept_g1",
                                     storage=storage, load_if_exists=True,
                                     sampler=TPESampler(seed=42))
    dept_study.enqueue_trial(FROZEN_GROUP1_DEPT)
    dept_study.optimize(dept_obj, n_trials=2,
                        callbacks=[make_trial_logger_g1(dept_log, "department", tmp)])
    assert dept_log.exists(), "department G1 log not written"
    dl = json.loads(dept_log.read_text())
    assert dl["n_complete"] >= 1 and "macro_f1" in dl["trials"][0], "dept G1 log malformed"

    # disposition (constrained ROC AUC, Group-1)
    y_disp = pd.Series(rng.integers(0, 2, n))
    Xtr2, Xval2, ytr2, yval2 = train_test_split(X, y_disp, test_size=0.3, random_state=1)
    disp_log = tmp / "tuning_log_disposition_g1.json"

    def disp_obj(trial):
        p = suggest_group1_disposition(trial)
        g = _g1_disp_gcfg(gcfg, p)
        model = train_disposition(Xtr2, ytr2, Xval2, yval2, FROZEN_GROUP2, g, "st-disp-g1")
        prob = model.predict_proba(Xval2)[:, 1]
        m = disposition_metrics(yval2, (prob >= 0.5).astype(int), prob)
        base = trial.study.user_attrs.get("baseline_under_rate")
        if base is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
            base = m["under_rate"]
        for k, v in m.items():
            if v is not None:
                trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("constraint", [m["under_rate"] - base])
        return m["roc_auc"] if m["roc_auc"] is not None else 0.0

    sampler = TPESampler(seed=42,
                         constraints_func=lambda t: t.user_attrs.get("constraint", [0.0]))
    disp_study = optuna.create_study(direction="maximize", study_name="st_disp_g1",
                                     storage=storage, load_if_exists=True, sampler=sampler)
    disp_study.enqueue_trial(FROZEN_GROUP1_DISP)
    disp_study.optimize(disp_obj, n_trials=2,
                        callbacks=[make_trial_logger_g1(disp_log, "disposition", tmp)])
    pl = json.loads(disp_log.read_text())
    assert pl["n_complete"] >= 1 and "feasible" in pl["trials"][0], "disp G1 log malformed"

    reopened = optuna.load_study(study_name="st_disp_g1", storage=storage)
    assert len(reopened.trials) >= 1, "study did not persist for resume"
    tp = _read_tuned_params(tmp, G1_TUNED_FNAME) or {}
    assert "department" in tp and "disposition" in tp, "g1 tuned_params blocks missing"
    assert tp["department"].get("selected") and tp["disposition"].get("selected"), \
        "g1 'selected' (chaining) blocks missing"
    if dept_mtime is not None:
        assert live_dept.stat().st_mtime == dept_mtime, "live department model modified!"
    if disp_mtime is not None:
        assert live_disp.stat().st_mtime == disp_mtime, "live disposition model modified!"

    print(f"  study DB:   {tmp / 'selftest_g1.db'}")
    print(f"  dept G1:    best macro_f1 trial recorded; disp G1: baseline_under="
          f"{disp_study.user_attrs['baseline_under_rate']:.4f}")
    print(f"  g1 tuned_params blocks: {sorted(tp.keys())} (both have 'selected')")
    print(f"  live models untouched: dept={'n/a' if dept_mtime is None else 'yes'}, "
          f"disp={'n/a' if disp_mtime is None else 'yes'}")
    print("\n  SELFTEST (Group-1) PASSED")


def _apply_group1_best(gcfg: dict, out_dir: Path) -> dict:
    """Chaining hook: override the frozen Group-1 fields in gcfg with the
    'selected' best from the Group-1 sweep, so the Group-2 re-run searches the
    regularization on top of the confirmed Group-1 config (--use-group1-best)."""
    tuned = _read_tuned_params(out_dir, G1_TUNED_FNAME)
    if not tuned:
        print(f"  ERROR: --use-group1-best set but {out_dir / G1_TUNED_FNAME} not "
              f"found. Run the Group-1 sweep (--group group1) first.", file=sys.stderr)
        sys.exit(2)
    g = dict(gcfg)
    dept = (tuned.get("department") or {}).get("selected")
    disp = (tuned.get("disposition") or {}).get("selected")
    if dept and dept.get("params"):
        p = dept["params"]
        g["dept_learning_rate"] = p["learning_rate"]
        g["dept_n_estimators"] = int(p["n_estimators"])
        g["dept_cw_exponent"] = p["class_weight_exponent"]
        print(f"  [chain] department Group-1 <- g1 trial #{dept.get('trial')}")
    if disp and disp.get("params"):
        p = disp["params"]
        g["disp_learning_rate"] = p["learning_rate"]
        g["disp_n_estimators"] = int(p["n_estimators"])
        g["disp_scale_pos_weight_exponent"] = p["scale_pos_weight_exponent"]
        print(f"  [chain] disposition Group-1 <- g1 trial #{disp.get('trial')}")
    if not dept and not disp:
        print(f"  ERROR: --use-group1-best set but no 'selected' block in "
              f"{G1_TUNED_FNAME}.", file=sys.stderr)
        sys.exit(2)
    return g


# Main
def _make_gcfg(smoke: bool) -> dict:
    """Frozen Group-1 training config. Smoke reduces trees purely for speed
    (throwaway numbers); full runs use the documented frozen constants."""
    if smoke:
        return {
            "dept_n_estimators": 300, "dept_learning_rate": 0.05,
            "dept_early_stopping": 30,
            "disp_n_estimators": 300, "disp_learning_rate": 0.05,
            "disp_early_stopping": 30,
        }
    return {
        "dept_n_estimators": DEPT_N_ESTIMATORS,
        "dept_learning_rate": DEPT_LEARNING_RATE,
        "dept_early_stopping": DEPT_EARLY_STOPPING,
        "disp_n_estimators": DISP_N_ESTIMATORS,
        "disp_learning_rate": DISP_LEARNING_RATE,
        "disp_early_stopping": DISP_EARLY_STOPPING,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["department", "disposition", "report"],
                    default="department")
    ap.add_argument("--group", choices=["group1", "group2"], default="group2",
                    help="group2 (default): the inherited regularization knobs. "
                         "group1: the per-head cost/weighting config (lr, "
                         "n_estimators, class-weight / scale_pos_weight exponent), "
                         "single-objective with the same objectives/constraints.")
    ap.add_argument("--use-group1-best", action="store_true",
                    help="Group-2 only: freeze Group-1 at the 'selected' best from "
                         "a prior --group group1 sweep (chaining) instead of the "
                         "original incumbent.")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Real-data subsample, throwaway paths, fast trees.")
    ap.add_argument("--selftest", action="store_true",
                    help="Synthetic CPU plumbing test (no data/GPU).")
    ap.add_argument("--storage", default=None, help="Optuna storage URL.")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="Override the feature-cache dir for this stage.")
    args = ap.parse_args()

    if args.selftest:
        run_selftest_g1() if args.group == "group1" else run_selftest()
        return

    # Re-read the device at call time (runpy re-runs the module top level, but a
    # cached-module import would not). Mirrors tune_triage_v3.py.
    global XGB_DEVICE, XGB_TREE_METHOD
    XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")
    XGB_TREE_METHOD = os.environ.get("XGB_TREE_METHOD", "hist")

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("optuna is not installed. `pip install optuna`.", file=sys.stderr)
        sys.exit(2)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    g1 = args.group == "group1"
    # Dedicated, non-colliding output paths (smoke -> throwaway subdir).
    out_dir = (DOCTOR_V3_HPO_DIR / "smoke") if args.smoke else DOCTOR_V3_HPO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.storage is None:
        stem = "optuna_doctor_g1" if g1 else "optuna_doctor"
        db = f"{stem}_smoke.db" if args.smoke else f"{stem}.db"
        args.storage = f"sqlite:///{out_dir / db}"

    # Caches: department reuses the existing tune_cache; disposition has its own.
    if args.cache_dir:
        dept_cache = disp_cache = Path(args.cache_dir)
    else:
        dept_cache = (DEPT_CACHE_DIR.parent / "tune_cache_smoke") if args.smoke else DEPT_CACHE_DIR
        disp_cache = (DOCTOR_DISPO_TUNE_CACHE_DIR / "smoke") if args.smoke else DOCTOR_DISPO_TUNE_CACHE_DIR

    gcfg = _make_gcfg(args.smoke)
    # Chaining (Group-2 only): fold the confirmed Group-1 best into the frozen config.
    if args.use_group1_best:
        if g1:
            print("  NOTE: --use-group1-best is ignored for --group group1.")
        else:
            gcfg = _apply_group1_best(gcfg, out_dir)

    print(f"  Doctor v3 HPO - group={args.group} stage={args.stage}"
          f"{'  [SMOKE]' if args.smoke else ''}")
    print(f"  storage:    {args.storage}")
    print(f"  out_dir:    {out_dir}")
    print(f"  dept_cache: {dept_cache}")
    print(f"  disp_cache: {disp_cache}")
    print(f"  device:     {XGB_DEVICE}  tree_method: {XGB_TREE_METHOD}")
    if XGB_DEVICE == "cpu":
        print("  WARNING: device is CPU - training will be very slow. Set "
              "XGB_DEVICE=cuda (and use a GPU runtime) before running.")
    print(f"  Reporting only: live models in {DOCTOR_V3_DIR} will NOT be modified.")

    if g1:
        if args.stage == "department":
            run_department_stage_g1(args, gcfg, optuna, TPESampler, out_dir, dept_cache)
        elif args.stage == "disposition":
            run_disposition_stage_g1(args, gcfg, optuna, TPESampler, out_dir, disp_cache)
        elif args.stage == "report":
            run_report_stage_g1(args, gcfg, optuna, out_dir, dept_cache, disp_cache)
    else:
        if args.stage == "department":
            run_department_stage(args, gcfg, optuna, TPESampler, out_dir, dept_cache)
        elif args.stage == "disposition":
            run_disposition_stage(args, gcfg, optuna, TPESampler, out_dir, disp_cache)
        elif args.stage == "report":
            run_report_stage(args, gcfg, optuna, out_dir, dept_cache, disp_cache)


if __name__ == "__main__":
    main()
