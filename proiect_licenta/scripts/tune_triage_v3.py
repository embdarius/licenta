"""Constrained Optuna hyperparameter sweep for the Triage v3 XGBoost heads.

REPORTING-ONLY. This sweep searches only the inherited "Group 2" XGBoost
regularization knobs (max_depth, subsample, colsample_*, min_child_weight,
gamma, reg_alpha, reg_lambda). The documented "Group 1" clinical-safety
choices are FROZEN and imported verbatim from train_triage_v3:

    learning_rate=0.01, n_estimators=5000, early_stopping_rounds=150,
    ESI_EXTREME_BOOST sample weights, neg_quadratic_kappa early-stopping metric.

It never calls save_models() and never writes the live triage v3 joblibs. All
output goes to a dedicated `artifacts/triage/v3/hpo/` subdir + a triage-only
feature cache, so nothing produced by previous Colab notebooks is overwritten.

Objective design (mirrors the thesis metrics in benchmark_triage_v3.evaluate_acuity):
  * acuity      — maximize quadratic-weighted kappa (QWK) SUBJECT TO a hard
                  constraint: under-triage rate <= the incumbent's under-triage
                  rate. The current (hand-tuned) Group-2 values are enqueued as
                  trial 0 and define that baseline in-sample on the inner-val
                  split. We never optimize headline accuracy.
  * disposition — maximize ROC AUC (binary; no ordinal/safety constraint).

Resume is built in (mirrors scripts/tune_doctor_v3.py): every trial is committed
to a SQLite study on Drive; `load_if_exists=True` means a fresh Colab session
re-running the tune cell continues where it left off. A human-readable JSON log
is rewritten after every completed trial.

Stages:
    --stage acuity        tune the acuity head (constrained QWK)
    --stage disposition   tune the disposition head (ROC AUC); needs acuity done
    --stage report        evaluate incumbent vs best-feasible on the OUTER test
                          split and write triage_hpo_results.json (thesis table)

Pre-flight:
    --selftest            synthetic, CPU-only, no MIMIC/GPU — validates all
                          plumbing (study + JSON log + resume) in seconds
    --smoke               real-data subsample, throwaway paths, 2 trials —
                          run on Colab GPU BEFORE committing to a full sweep

Usage on Colab (after symlinking artifacts/ + data/ to Drive):
    !XGB_DEVICE=cuda XGB_TREE_METHOD=hist python scripts/tune_triage_v3.py \\
        --stage acuity --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Python 3.12 / pydantic / optuna compatibility shim (copied verbatim from
# scripts/tune_doctor_v3.py). Pydantic (via CrewAI) monkey-patches
# warnings.warn with a filter that predates 3.12's skip_file_prefixes kwarg,
# which optuna passes. Wrap it kwargs-tolerantly. Must run before `import optuna`.
# ---------------------------------------------------------------------------
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

    _w.warn = _compatible_warn
    _w._optuna_compat_patched = True


import numpy as np
import pandas as pd
import joblib

# Run with the project on the path even when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, roc_auc_score, recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from proiect_licenta.paths import (
    TRIAGE_V3_DIR, TRIAGE_V3_HPO_DIR, TRIAGE_TUNE_CACHE_DIR,
)
# Frozen Group-1 config + reusable feature/metric machinery — single source of
# truth is the training pipeline; we never re-implement it here.
from proiect_licenta.training.train_triage_v3 import (
    load_and_clean_data, build_features,
    neg_quadratic_kappa, ESI_EXTREME_BOOST,
    ACUITY_N_ESTIMATORS, ACUITY_LEARNING_RATE, ACUITY_EARLY_STOPPING_ROUNDS,
    DISP_N_ESTIMATORS, DISP_LEARNING_RATE, DISP_EARLY_STOPPING_ROUNDS,
)

# Read the GPU device at RUNTIME, not import-time. train_triage_v3 caches its
# own XGB_DEVICE at first import; in a Colab session where an earlier cell
# imported that module before `XGB_DEVICE=cuda` was set (e.g. the self-test
# cell), the cached constant would be stale "cpu" and every later run would
# silently train on CPU. Reading os.environ here — this file's top level re-runs
# on every `runpy.run_path` call, and is re-read in main() too — keeps the
# device correct regardless of cell order.
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")
XGB_TREE_METHOD = os.environ.get("XGB_TREE_METHOD", "hist")

# Early-stopping metric used DURING THE SEARCH only. The custom QWK metric
# (neg_quadratic_kappa) runs in Python and forces a GPU→host round-trip every
# boosting round, which made each acuity trial take hours. The native mlogloss
# is computed on-device and picks essentially the same iteration (the docs note
# QWK early-stopping is "mostly cosmetic"; the sample-weight boost is the lever).
# The trial is still SCORED by QWK + under-triage after fit, and the `report`
# stage re-fits the winner with the real neg_quadratic_kappa at full fidelity.
SEARCH_EVAL_METRIC = "mlogloss"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACUITY_STUDY_NAME = "triage_acuity_qwk"
DISP_STUDY_NAME = "triage_disposition_auc"

# The current hand-tuned Group-2 values (train_triage_v3.py:569-590). Enqueued
# as trial 0 of each study so Optuna always evaluates the incumbent first; for
# acuity its inner-val under-triage rate becomes the feasibility threshold.
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

ACUITY_CLASSES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Group-2 search space (regularization box; ranges from tune_doctor_v3.py,
# minus the frozen learning_rate). subsample is searched here too.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Group-1 search (the clinical-safety levers themselves). MULTI-OBJECTIVE per
# the triage future-work plan: the boost vector + disposition threshold move the
# safety profile, so we report a Pareto frontier rather than collapse it into one
# constrained scalar. Group-2 is frozen at FROZEN_GROUP2 throughout. ESI 3/4 are
# pinned to 1.0 as the reference floor (only the extreme classes are boosted).
# ---------------------------------------------------------------------------
ACUITY_G1_STUDY_NAME = "triage_acuity_g1"
DISP_G1_STUDY_NAME = "triage_disposition_g1"

# Incumbent Group-1 — the documented iter-2 config (train_triage_v3). Enqueued as
# trial 0 of each study so it sits on the frontier for comparison.
FROZEN_GROUP1_ACUITY = {
    "learning_rate": ACUITY_LEARNING_RATE,      # 0.01
    "n_estimators": ACUITY_N_ESTIMATORS,        # 5000
    "boost_esi1": ESI_EXTREME_BOOST[1],         # 1.5
    "boost_esi2": ESI_EXTREME_BOOST[2],         # 1.3
    "boost_esi5": ESI_EXTREME_BOOST[5],         # 2.0
}
FROZEN_GROUP1_DISP = {
    "learning_rate": DISP_LEARNING_RATE,        # 0.01
    "n_estimators": DISP_N_ESTIMATORS,          # 5000
    "scale_pos_weight_exponent": 1.0,           # live = raw neg/pos ratio
    "decision_threshold": 0.5,                  # live triage disposition cutoff
}


def suggest_group1_acuity(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "n_estimators": trial.suggest_categorical("n_estimators", [3000, 5000, 8000]),
        "boost_esi1": trial.suggest_float("boost_esi1", 1.0, 2.5),
        "boost_esi2": trial.suggest_float("boost_esi2", 1.0, 1.8),
        "boost_esi5": trial.suggest_float("boost_esi5", 1.0, 3.0),
    }


def suggest_group1_disposition(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "n_estimators": trial.suggest_categorical("n_estimators", [3000, 5000, 8000]),
        "scale_pos_weight_exponent": trial.suggest_float("scale_pos_weight_exponent", 0.5, 1.3),
        "decision_threshold": trial.suggest_float("decision_threshold", 0.2, 0.6),
    }


def _boost_map(p: dict) -> dict:
    """Reconstruct the full ESI 1-5 boost dict from the three searched knobs."""
    return {1: p["boost_esi1"], 2: p["boost_esi2"], 3: 1.0, 4: 1.0, 5: p["boost_esi5"]}


def _g1_acuity_gcfg(base: dict, p: dict) -> dict:
    """Per-trial Group-1 acuity training config: frozen-base + overrides."""
    g = dict(base)
    g["acuity_learning_rate"] = p["learning_rate"]
    g["acuity_n_estimators"] = int(p["n_estimators"])
    g["acuity_boost_map"] = _boost_map(p)
    return g


def _g1_disp_gcfg(base: dict, p: dict) -> dict:
    """Per-trial Group-1 disposition training config (threshold is applied
    post-fit, not in training, so it is not threaded into gcfg)."""
    g = dict(base)
    g["disp_learning_rate"] = p["learning_rate"]
    g["disp_n_estimators"] = int(p["n_estimators"])
    g["disp_scale_pos_weight_exponent"] = p["scale_pos_weight_exponent"]
    return g


# ---------------------------------------------------------------------------
# Per-trial XGBoost iteration progress bar (tqdm). Optional; silent fallback.
# Mirrors train_triage_v3._make_xgb_progress_callback.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Frozen training (Group-1 fixed; Group-2 passed in)
# ---------------------------------------------------------------------------
def _acuity_sample_weights(y_train: pd.Series, boost_map: dict | None = None) -> pd.Series:
    """sqrt-balanced base weight * ESI boost — identical to
    train_triage_v3.train_acuity_model when boost_map is None (Group-1 frozen).

    The Group-1 sweep passes a candidate boost vector here; Group-2 callers leave
    it None and get the documented ESI_EXTREME_BOOST."""
    boost = boost_map if boost_map is not None else ESI_EXTREME_BOOST
    class_counts = y_train.value_counts()
    total = len(y_train)
    n_classes = len(class_counts)
    base = y_train.map(lambda x: np.sqrt(total / (n_classes * class_counts[x])))
    return base * y_train.map(boost)


def train_acuity(X_tr, y_tr, X_val, y_val, group2, gcfg, desc, eval_metric=None):
    """Train one acuity model (frozen G1 + given Group-2). y_* are ESI 1-5;
    shifted to 0-4 for the multi:softprob objective (matches training).

    eval_metric controls the early-stopping metric only: the search passes the
    native SEARCH_EVAL_METRIC ("mlogloss") for speed; report/full-fidelity calls
    leave it None to use the documented neg_quadratic_kappa. Either way the model
    is scored by QWK after fit."""
    metric = eval_metric if eval_metric is not None else neg_quadratic_kappa
    # Group-1 sweep injects a candidate boost vector via gcfg; absent -> frozen.
    sample_weights = _acuity_sample_weights(y_tr, gcfg.get("acuity_boost_map"))
    cb = _make_trial_progress_callback(gcfg["acuity_n_estimators"], desc)
    model = XGBClassifier(
        n_estimators=gcfg["acuity_n_estimators"],
        learning_rate=gcfg["acuity_learning_rate"],
        early_stopping_rounds=gcfg["acuity_early_stopping_rounds"],
        objective="multi:softprob", num_class=5,
        eval_metric=metric,
        random_state=42, n_jobs=-1, verbosity=0,
        device=XGB_DEVICE, tree_method=XGB_TREE_METHOD,
        callbacks=[cb] if cb is not None else None,
        **group2,
    )
    model.fit(
        X_tr, y_tr - 1,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val - 1)],
        verbose=False,
    )
    try:
        model.set_params(callbacks=None, eval_metric=None)
    except Exception:
        pass
    return model


def train_disposition(X_tr, y_tr, X_val, y_val, group2, gcfg, desc):
    """Train one disposition model (frozen G1 + given Group-2). Binary
    admit/discharge with scale_pos_weight from the data (Group-1).

    Live triage uses the raw ratio (exponent 1.0). The Group-1 sweep injects a
    candidate exponent via gcfg["disp_scale_pos_weight_exponent"] so the operating
    point can be tuned; absent -> raw ratio (frozen)."""
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    ratio = neg / pos if pos else 1.0
    exponent = gcfg.get("disp_scale_pos_weight_exponent")
    scale = ratio ** exponent if exponent is not None else ratio
    cb = _make_trial_progress_callback(gcfg["disp_n_estimators"], desc)
    model = XGBClassifier(
        n_estimators=gcfg["disp_n_estimators"],
        learning_rate=gcfg["disp_learning_rate"],
        early_stopping_rounds=gcfg["disp_early_stopping_rounds"],
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


# ---------------------------------------------------------------------------
# Metrics — replicate benchmark_triage_v3.evaluate_acuity:57-81 exactly.
# ---------------------------------------------------------------------------
def acuity_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    yt = y_true.values
    errors = y_pred - yt
    return {
        "exact": float(accuracy_score(y_true, y_pred)),
        "within_1": float(np.mean(np.abs(errors) <= 1)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "over_rate": float((errors < 0).mean()),     # predicted lower ESI
        "under_rate": float((errors > 0).mean()),     # predicted higher ESI
    }


def disposition_metrics(y_true: pd.Series, y_pred, y_prob) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    return out


def disposition_metrics_at_threshold(y_true: pd.Series, y_prob, threshold: float) -> dict:
    """Group-1 disposition metrics at a candidate decision threshold. under_rate
    is the under-triage rate (a true admit predicted as discharge) as a fraction
    of the whole population — the dangerous error, analogous to the acuity
    under_rate; the Pareto study trades it against overall accuracy. roc_auc is
    threshold-free, logged as a reference."""
    yt = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(yt, y_pred)),
        "under_rate": float(((yt == 1) & (y_pred == 0)).mean()),
        "over_rate": float(((yt == 0) & (y_pred == 1)).mean()),
        "threshold": float(threshold),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(yt, y_prob))
    except Exception:
        out["roc_auc"] = None
    return out


# ---------------------------------------------------------------------------
# Feature cache (float32 parquet). Distinct dir from the doctor sweep.
# ---------------------------------------------------------------------------
def load_or_build_cache(cache_dir: Path, rebuild: bool, subsample: int | None):
    """Return (X_train, X_test, y_train_df, y_test_df). y_*_df has columns
    'acuity' (1-5) and 'admitted' (0/1). Features fit on OUTER-TRAIN only."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fxtr = cache_dir / "X_train.parquet"
    fxte = cache_dir / "X_test.parquet"
    fytr = cache_dir / "y_train.parquet"
    fyte = cache_dir / "y_test.parquet"
    fmeta = cache_dir / "meta.json"

    if not rebuild and all(f.exists() for f in (fxtr, fxte, fytr, fyte, fmeta)):
        print(f"[cache] Loading from {cache_dir}")
        return (
            pd.read_parquet(fxtr), pd.read_parquet(fxte),
            pd.read_parquet(fytr), pd.read_parquet(fyte),
        )

    print("[cache] Building feature matrix from raw data (slow: parses discharge.csv)")
    df, _edstays = load_and_clean_data()
    if subsample and len(df) > subsample:
        df, _ = train_test_split(
            df, train_size=subsample, random_state=7, stratify=df["acuity"],
        )
        print(f"[cache] Subsampled to {len(df):,} rows (smoke).")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["acuity"],
    )
    X_train, tfidf, severity_map, vital_medians = build_features(train_df, fit=True)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        X_test, _, _, _ = build_features(
            test_df, tfidf=tfidf, severity_map=severity_map,
            vital_medians=vital_medians, fit=False,
        )
    y_train = pd.DataFrame({
        "acuity": train_df["acuity"].reset_index(drop=True),
        "admitted": train_df["admitted"].reset_index(drop=True),
    })
    y_test = pd.DataFrame({
        "acuity": test_df["acuity"].reset_index(drop=True),
        "admitted": test_df["admitted"].reset_index(drop=True),
    })

    # float64 -> float32 to halve parquet size; XGBoost is fine with float32.
    for X in (X_train, X_test):
        for c in X.columns:
            if X[c].dtype == "float64":
                X[c] = X[c].astype("float32")

    X_train.to_parquet(fxtr, index=False)
    X_test.to_parquet(fxte, index=False)
    y_train.to_parquet(fytr, index=False)
    y_test.to_parquet(fyte, index=False)
    fmeta.write_text(json.dumps({
        "built_at": datetime.now().isoformat(),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "subsample": subsample,
    }, indent=2))
    print(f"[cache] Wrote {X_train.shape[1]} features, "
          f"{len(X_train):,} train / {len(X_test):,} test rows to {cache_dir}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Incremental JSON log — rewritten after every completed trial (atomic).
# ---------------------------------------------------------------------------
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


def make_trial_logger(log_path: Path, stage: str, out_dir: Path):
    """Optuna callback. After every trial it (1) rewrites the human-readable
    per-trial JSON log and (2) re-writes the best-config block into
    tuned_params_triage.json. Writing the block every trial (not just at the end
    of optimize()) makes the chunked/interruptible workflow safe: a Ctrl-C or
    Colab reclaim mid-chunk still leaves a usable tuned_params for the next
    stage."""
    def _log(study, trial):
        rows = []
        for t in study.trials:
            if t.state.name != "COMPLETE":
                continue
            ua = t.user_attrs
            row = {"number": t.number, "value": t.value, "params": t.params}
            for k in ("qwk", "exact", "within_1", "under_rate", "over_rate",
                      "roc_auc", "accuracy", "best_iteration"):
                if k in ua:
                    row[k] = ua[k]
            if "constraint" in ua:
                row["feasible"] = bool(ua["constraint"][0] <= 1e-9)
            rows.append(row)
        payload = {
            "stage": stage,
            "study_name": study.study_name,
            "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
            "baseline_qwk": study.user_attrs.get("baseline_qwk"),
            "n_complete": len(rows),
            "updated_at": datetime.now().isoformat(),
            "trials": rows,
        }
        _atomic_write_json(log_path, payload)

        # Persist the best-config block every trial (interruption-safe).
        if stage == "acuity":
            best = best_feasible_trial(study)
            if best is not None:
                _update_tuned_params(out_dir, "acuity", _acuity_block(study, best),
                                     quiet=True)
        elif stage == "disposition":
            try:
                best = study.best_trial
            except ValueError:
                best = None
            if best is not None:
                _update_tuned_params(out_dir, "disposition",
                                     _disposition_block(study, best), quiet=True)

        if trial.state.name == "COMPLETE":
            ua = trial.user_attrs
            extra = ""
            if "qwk" in ua:
                feas = "feasible" if ua.get("constraint", [1])[0] <= 1e-9 else "INFEASIBLE"
                extra = (f"qwk={ua['qwk']:.4f} under={ua['under_rate']*100:.2f}% "
                         f"within1={ua['within_1']*100:.2f}% [{feas}]")
            elif "roc_auc" in ua:
                extra = f"auc={ua['roc_auc']:.4f} acc={ua['accuracy']*100:.2f}%"
            print(f"  trial #{trial.number:3d} done | {extra} "
                  f"best_iter={ua.get('best_iteration', '?')}", flush=True)
    return _log


def best_feasible_trial(study):
    feas = [
        t for t in study.trials
        if t.state.name == "COMPLETE" and t.user_attrs.get("constraint", [1.0])[0] <= 1e-9
    ]
    return max(feas, key=lambda t: t.value) if feas else None


def _acuity_block(study, best) -> dict:
    return {
        "study_name": study.study_name,
        "best_trial": int(best.number),
        "objective": "QWK s.t. under_rate <= incumbent",
        "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
        "baseline_qwk": study.user_attrs.get("baseline_qwk"),
        "best_qwk": float(best.value),
        "best_under_rate": float(best.user_attrs["under_rate"]),
        "best_within_1": float(best.user_attrs["within_1"]),
        "best_exact": float(best.user_attrs["exact"]),
        "n_trials": len(study.trials),
        "params": best.params,
        "written_at": datetime.now().isoformat(),
    }


def _disposition_block(study, best) -> dict:
    return {
        "study_name": study.study_name,
        "best_trial": int(best.number),
        "objective": "ROC AUC",
        "best_roc_auc": float(best.value),
        "best_accuracy": float(best.user_attrs["accuracy"]),
        "n_trials": len(study.trials),
        "params": best.params,
        "written_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Stage: acuity (constrained QWK)
# ---------------------------------------------------------------------------
def run_acuity_stage(args, gcfg, optuna, TPESampler, X_train, y_train, out_dir):
    y_acuity = y_train["acuity"].reset_index(drop=True)
    X_in_tr, X_in_val, y_in_tr, y_in_val = train_test_split(
        X_train, y_acuity, test_size=0.2, random_state=1, stratify=y_acuity,
    )
    print(f"  Inner split (tuning): train={len(X_in_tr):,}, val={len(X_in_val):,}")

    def objective(trial):
        group2 = suggest_group2(trial)
        print(f"\n  >>> trial #{trial.number:3d} | "
              f"depth={group2['max_depth']} sub={group2['subsample']:.2f} "
              f"mcw={group2['min_child_weight']} gamma={group2['gamma']:.3f} "
              f"a={group2['reg_alpha']:.3f} l={group2['reg_lambda']:.3f}", flush=True)
        model = train_acuity(X_in_tr, y_in_tr, X_in_val, y_in_val, group2, gcfg,
                             f"    trial #{trial.number} acuity",
                             eval_metric=SEARCH_EVAL_METRIC)
        y_pred = model.predict(X_in_val) + 1
        m = acuity_metrics(y_in_val, y_pred)

        # Trial 0 (the enqueued incumbent) defines the in-sample baseline.
        baseline = trial.study.user_attrs.get("baseline_under_rate")
        if baseline is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
            trial.study.set_user_attr("baseline_qwk", m["qwk"])
            baseline = m["under_rate"]

        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        # <= 0 means feasible (under-triage no worse than the incumbent).
        trial.set_user_attr("constraint", [m["under_rate"] - baseline])
        return m["qwk"]

    def constraints_func(trial):
        return trial.user_attrs.get("constraint", [0.0])

    study = optuna.create_study(
        direction="maximize", study_name=args.study_name or ACUITY_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True, constraints_func=constraints_func),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP2)  # incumbent = trial 0 = baseline
        print("  Enqueued incumbent Group-2 config as trial 0 (defines baseline).")

    log_path = out_dir / "tuning_log_acuity.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger(log_path, "acuity", out_dir)],
                   gc_after_trial=True)

    best = best_feasible_trial(study)
    print("\n" + "=" * 60)
    if best is None:
        print("  No feasible trial yet (none beat the under-triage baseline).")
        return
    print(f"  Best FEASIBLE acuity trial: #{best.number}  "
          f"qwk={best.value:.4f}  under={best.user_attrs['under_rate']*100:.2f}%  "
          f"(baseline under={study.user_attrs['baseline_under_rate']*100:.2f}%)")
    for k, v in best.params.items():
        print(f"    {k:18s} = {v}")
    _update_tuned_params(out_dir, "acuity", _acuity_block(study, best))


# ---------------------------------------------------------------------------
# Stage: disposition (ROC AUC, cascades on predicted_acuity)
# ---------------------------------------------------------------------------
def run_disposition_stage(args, gcfg, optuna, TPESampler, X_train, y_train, out_dir):
    tuned = _read_tuned_params(out_dir)
    acu = (tuned or {}).get("acuity")
    if not acu:
        print("  ERROR: run `--stage acuity` first (tuned_params_triage.json has "
              "no acuity block).", file=sys.stderr)
        sys.exit(2)
    acu_params = acu["params"]
    print(f"  Using best acuity Group-2 (trial #{acu['best_trial']}, "
          f"qwk={acu['best_qwk']:.4f}) to build predicted_acuity.")

    # Train acuity once on full outer-train with best G2, then add the in-sample
    # predicted_acuity feature (mirrors train_triage_v3.py:843-846).
    y_acuity = y_train["acuity"].reset_index(drop=True)
    y_admit = y_train["admitted"].reset_index(drop=True)
    acu_tr, acu_val, ya_tr, ya_val = train_test_split(
        X_train, y_acuity, test_size=0.2, random_state=1, stratify=y_acuity,
    )
    acu_model = train_acuity(acu_tr, ya_tr, acu_val, ya_val, acu_params, gcfg,
                             "    acuity (for cascade)",
                             eval_metric=SEARCH_EVAL_METRIC)
    X_disp = X_train.copy()
    X_disp["predicted_acuity"] = acu_model.predict(X_train) + 1

    Xd_tr, Xd_val, yd_tr, yd_val = train_test_split(
        X_disp, y_admit, test_size=0.2, random_state=1, stratify=y_admit,
    )
    print(f"  Inner split (tuning): train={len(Xd_tr):,}, val={len(Xd_val):,}")

    def objective(trial):
        group2 = suggest_group2(trial)
        print(f"\n  >>> trial #{trial.number:3d} | "
              f"depth={group2['max_depth']} sub={group2['subsample']:.2f} "
              f"mcw={group2['min_child_weight']}", flush=True)
        model = train_disposition(Xd_tr, yd_tr, Xd_val, yd_val, group2, gcfg,
                                  f"    trial #{trial.number} disp")
        y_pred = model.predict(Xd_val)
        y_prob = model.predict_proba(Xd_val)[:, 1]
        m = disposition_metrics(yd_val, y_pred, y_prob)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return m["roc_auc"] if m["roc_auc"] is not None else 0.0

    study = optuna.create_study(
        direction="maximize", study_name=args.study_name or DISP_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP2)
        print("  Enqueued incumbent Group-2 config as trial 0.")

    log_path = out_dir / "tuning_log_disposition.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger(log_path, "disposition", out_dir)],
                   gc_after_trial=True)

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"  Best disposition trial: #{best.number}  auc={best.value:.4f}  "
          f"acc={best.user_attrs['accuracy']*100:.2f}%")
    for k, v in best.params.items():
        print(f"    {k:18s} = {v}")
    _update_tuned_params(out_dir, "disposition", _disposition_block(study, best))


# ---------------------------------------------------------------------------
# Stage: report — incumbent vs best-feasible on the OUTER test split.
# ---------------------------------------------------------------------------
def run_report_stage(gcfg, X_train, y_train, X_test, y_test, out_dir):
    live_acuity = TRIAGE_V3_DIR / "acuity_model.joblib"
    live_disp = TRIAGE_V3_DIR / "disposition_model.joblib"
    if not live_acuity.exists() or not live_disp.exists():
        print(f"  ERROR: live models not found in {TRIAGE_V3_DIR}. Report needs "
              f"the deployed iter-2 joblibs as the incumbent.", file=sys.stderr)
        sys.exit(2)

    tuned = _read_tuned_params(out_dir) or {}
    y_acuity_tr = y_train["acuity"].reset_index(drop=True)
    y_admit_tr = y_train["admitted"].reset_index(drop=True)
    y_acuity_te = y_test["acuity"].reset_index(drop=True)
    y_admit_te = y_test["admitted"].reset_index(drop=True)

    results = {"generated_at": datetime.now().isoformat(),
               "test_rows": int(len(X_test))}

    # ---- Acuity ----
    print("\n  [acuity] incumbent (live iter-2 model) on outer-test ...")
    inc_model = joblib.load(live_acuity)
    inc_pred = inc_model.predict(X_test) + 1
    inc_m = acuity_metrics(y_acuity_te, inc_pred)
    inc_m["per_class_recall"] = _per_class_recall(y_acuity_te, inc_pred)

    acu_block = {"incumbent": inc_m}
    if tuned.get("acuity"):
        print("  [acuity] best-feasible config: retrain on outer-train ...")
        acu_tr, acu_val, ya_tr, ya_val = train_test_split(
            X_train, y_acuity_tr, test_size=0.2, random_state=1, stratify=y_acuity_tr,
        )
        best_model = train_acuity(acu_tr, ya_tr, acu_val, ya_val,
                                  tuned["acuity"]["params"], gcfg, "    acuity (best)")
        best_pred = best_model.predict(X_test) + 1
        best_m = acuity_metrics(y_acuity_te, best_pred)
        best_m["per_class_recall"] = _per_class_recall(y_acuity_te, best_pred)
        acu_block["best_feasible"] = best_m
        acu_block["delta_qwk"] = best_m["qwk"] - inc_m["qwk"]
        acu_block["delta_under_rate"] = best_m["under_rate"] - inc_m["under_rate"]
    else:
        print("  [acuity] no tuned acuity block yet — incumbent only.")
        best_model = inc_model  # for cascade below
    results["acuity"] = acu_block

    # ---- Disposition (cascade on predicted_acuity) ----
    print("\n  [disposition] incumbent (live models) on outer-test ...")
    inc_disp = joblib.load(live_disp)
    X_test_inc = X_test.copy()
    X_test_inc["predicted_acuity"] = inc_model.predict(X_test) + 1
    inc_dm = disposition_metrics(
        y_admit_te, inc_disp.predict(X_test_inc),
        inc_disp.predict_proba(X_test_inc)[:, 1],
    )
    disp_block = {"incumbent": inc_dm}
    if tuned.get("disposition") and tuned.get("acuity"):
        print("  [disposition] best config: retrain acuity+disposition on outer-train ...")
        X_disp_tr = X_train.copy()
        X_disp_tr["predicted_acuity"] = best_model.predict(X_train) + 1
        Xd_tr, Xd_val, yd_tr, yd_val = train_test_split(
            X_disp_tr, y_admit_tr, test_size=0.2, random_state=1, stratify=y_admit_tr,
        )
        best_disp = train_disposition(Xd_tr, yd_tr, Xd_val, yd_val,
                                      tuned["disposition"]["params"], gcfg,
                                      "    disposition (best)")
        X_test_best = X_test.copy()
        X_test_best["predicted_acuity"] = best_model.predict(X_test) + 1
        best_dm = disposition_metrics(
            y_admit_te, best_disp.predict(X_test_best),
            best_disp.predict_proba(X_test_best)[:, 1],
        )
        disp_block["best"] = best_dm
        if inc_dm["roc_auc"] and best_dm["roc_auc"]:
            disp_block["delta_roc_auc"] = best_dm["roc_auc"] - inc_dm["roc_auc"]
    results["disposition"] = disp_block

    out = out_dir / "triage_hpo_results.json"
    _atomic_write_json(out, results)
    print(f"\n  Wrote {out}")
    _print_report_summary(results)


def _per_class_recall(y_true, y_pred) -> dict:
    rec = recall_score(y_true, y_pred, labels=ACUITY_CLASSES, average=None,
                       zero_division=0)
    return {str(c): float(r) for c, r in zip(ACUITY_CLASSES, rec)}


def _print_report_summary(results):
    a = results.get("acuity", {})
    inc, best = a.get("incumbent"), a.get("best_feasible")
    print("\n  ── Acuity (outer-test) ──")
    print(f"    incumbent:      exact={inc['exact']*100:.2f}%  "
          f"qwk={inc['qwk']:.4f}  under={inc['under_rate']*100:.2f}%")
    if best:
        print(f"    best-feasible:  exact={best['exact']*100:.2f}%  "
              f"qwk={best['qwk']:.4f}  under={best['under_rate']*100:.2f}%")
        print(f"    delta:          qwk={a['delta_qwk']:+.4f}  "
              f"under={a['delta_under_rate']*100:+.2f}pp")
    d = results.get("disposition", {})
    if d.get("incumbent"):
        di, db = d["incumbent"], d.get("best")
        print("  ── Disposition (outer-test) ──")
        print(f"    incumbent:      acc={di['accuracy']*100:.2f}%  auc={di['roc_auc']}")
        if db:
            print(f"    best:           acc={db['accuracy']*100:.2f}%  auc={db['roc_auc']}")


# ---------------------------------------------------------------------------
# tuned_params_triage.json read/update (preserve sibling blocks across stages)
# ---------------------------------------------------------------------------
def _tuned_path(out_dir: Path, fname: str = "tuned_params_triage.json") -> Path:
    return out_dir / fname


def _read_tuned_params(out_dir: Path, fname: str = "tuned_params_triage.json") -> dict | None:
    p = _tuned_path(out_dir, fname)
    return json.loads(p.read_text()) if p.exists() else None


def _update_tuned_params(out_dir: Path, key: str, block: dict, quiet: bool = False,
                         fname: str = "tuned_params_triage.json"):
    data = _read_tuned_params(out_dir, fname) or {}
    data[key] = block
    _atomic_write_json(_tuned_path(out_dir, fname), data)
    if not quiet:
        print(f"  Updated {_tuned_path(out_dir, fname)} ['{key}']")


# Group-1 tuned-params live in a sibling file so the two studies never collide.
G1_TUNED_FNAME = "tuned_params_triage_g1.json"


# ===========================================================================
# Group-1 multi-objective sweep (NSGA-II) — the clinical-safety levers.
# Group-2 stays frozen at FROZEN_GROUP2; we report a Pareto frontier instead of
# a single constrained scalar (per the triage Group-1 future-work plan).
# ===========================================================================
def _esi5_recall(y_true, y_pred) -> float:
    return float(recall_score(y_true, y_pred, labels=[5], average="macro",
                              zero_division=0))


def make_trial_logger_g1(log_path: Path, stage: str, out_dir: Path,
                         objective_names: list[str]):
    """Multi-objective Optuna callback: rewrites the per-trial JSON log (with the
    current Pareto front flagged) and persists the Group-1 frontier block to
    tuned_params_triage_g1.json after every trial (interruption-safe)."""
    metric_keys = ("exact", "within_1", "qwk", "under_rate", "over_rate",
                   "esi5_recall", "accuracy", "roc_auc", "threshold",
                   "best_iteration")

    def _log(study, trial):
        try:
            front = [t.number for t in study.best_trials]
        except Exception:
            front = []
        rows = []
        for t in study.trials:
            if t.state.name != "COMPLETE":
                continue
            ua = t.user_attrs
            row = {"number": t.number, "params": t.params,
                   "values": list(t.values) if t.values is not None else None,
                   "on_pareto_front": t.number in front}
            if t.values is not None:
                row["objectives"] = {n: float(t.values[i])
                                     for i, n in enumerate(objective_names)}
            for k in metric_keys:
                if k in ua:
                    row[k] = ua[k]
            rows.append(row)
        payload = {
            "stage": stage, "study_name": study.study_name,
            "mode": "multi-objective",
            "objective_names": objective_names,
            "directions": [d.name for d in study.directions],
            "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
            "pareto_front_trials": front,
            "n_complete": len(rows),
            "updated_at": datetime.now().isoformat(),
            "trials": rows,
        }
        _atomic_write_json(log_path, payload)
        if front:
            _update_tuned_params(out_dir, stage, _g1_block(study, objective_names, stage),
                                 quiet=True, fname=G1_TUNED_FNAME)
        if trial.state.name == "COMPLETE" and trial.values is not None:
            vstr = "  ".join(f"{n}={v:.4f}" for n, v in zip(objective_names, trial.values))
            flag = "[PARETO]" if trial.number in front else ""
            print(f"  trial #{trial.number:3d} done | {vstr} {flag} "
                  f"best_iter={trial.user_attrs.get('best_iteration', '?')}", flush=True)
    return _log


def _g1_select(study, stage: str) -> dict | None:
    """Heuristic knee used as the CHAINING DEFAULT: the best headline metric on
    the Pareto front whose under_rate does not exceed the incumbent (trial 0).
    The user can override the 'selected' block by hand before chaining into a
    Group-2 re-run (--use-group1-best reads it)."""
    front = study.best_trials
    if not front:
        return None
    base = study.user_attrs.get("baseline_under_rate")
    head_key = "exact" if stage == "acuity" else "accuracy"
    feasible = [t for t in front
                if base is None or t.user_attrs.get("under_rate", 1.0) <= base + 1e-9]
    pool = feasible or front
    best = max(pool, key=lambda t: t.user_attrs.get(head_key, 0.0))
    return {
        "trial": int(best.number),
        "rationale": (f"max {head_key} on the Pareto front with under_rate <= incumbent"
                      if feasible else
                      f"no front point matched the incumbent under_rate; max {head_key} overall"),
        "params": best.params,
        "metrics": {k: best.user_attrs[k] for k in
                    ("exact", "qwk", "under_rate", "esi5_recall", "accuracy", "threshold")
                    if k in best.user_attrs},
    }


def _g1_block(study, objective_names: list[str], stage: str) -> dict:
    metric_keys = ("exact", "within_1", "qwk", "under_rate", "over_rate",
                   "esi5_recall", "accuracy", "roc_auc", "threshold")
    front = []
    for t in study.best_trials:
        front.append({
            "trial": int(t.number),
            "params": t.params,
            "objectives": {n: float(t.values[i]) for i, n in enumerate(objective_names)},
            "metrics": {k: t.user_attrs[k] for k in metric_keys if k in t.user_attrs},
        })
    return {
        "study_name": study.study_name,
        "mode": "multi-objective",
        "objective_names": objective_names,
        "directions": [d.name for d in study.directions],
        "incumbent_trial": 0,
        "baseline_under_rate": study.user_attrs.get("baseline_under_rate"),
        "pareto_front": front,
        "selected": _g1_select(study, stage),
        "n_trials": len(study.trials),
        "written_at": datetime.now().isoformat(),
    }


def run_acuity_stage_g1(args, gcfg, optuna, X_train, y_train, out_dir):
    """Multi-objective acuity Group-1: maximize (exact accuracy, ESI-5 recall),
    minimize under-triage rate, over the boost vector + lr + n_estimators.
    Group-2 frozen at FROZEN_GROUP2."""
    from optuna.samplers import NSGAIISampler
    y_acuity = y_train["acuity"].reset_index(drop=True)
    X_in_tr, X_in_val, y_in_tr, y_in_val = train_test_split(
        X_train, y_acuity, test_size=0.2, random_state=1, stratify=y_acuity,
    )
    print(f"  Inner split (tuning): train={len(X_in_tr):,}, val={len(X_in_val):,}")
    obj_names = ["exact", "under_rate", "esi5_recall"]

    def objective(trial):
        p = suggest_group1_acuity(trial)
        g = _g1_acuity_gcfg(gcfg, p)
        print(f"\n  >>> trial #{trial.number:3d} | lr={p['learning_rate']:.4f} "
              f"n_est={p['n_estimators']} boost=({p['boost_esi1']:.2f},"
              f"{p['boost_esi2']:.2f},{p['boost_esi5']:.2f})", flush=True)
        model = train_acuity(X_in_tr, y_in_tr, X_in_val, y_in_val, FROZEN_GROUP2, g,
                             f"    trial #{trial.number} acuity",
                             eval_metric=SEARCH_EVAL_METRIC)
        y_pred = model.predict(X_in_val) + 1
        m = acuity_metrics(y_in_val, y_pred)
        m["esi5_recall"] = _esi5_recall(y_in_val, y_pred)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        if trial.study.user_attrs.get("baseline_under_rate") is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
        return m["exact"], m["under_rate"], m["esi5_recall"]

    study = optuna.create_study(
        directions=["maximize", "minimize", "maximize"],
        study_name=args.study_name or ACUITY_G1_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=NSGAIISampler(seed=42),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP1_ACUITY)
        print("  Enqueued incumbent Group-1 config as trial 0 (sits on the frontier).")

    log_path = out_dir / "tuning_log_acuity_g1.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger_g1(log_path, "acuity", out_dir, obj_names)],
                   gc_after_trial=True)
    _print_pareto(study, obj_names, "acuity")
    _update_tuned_params(out_dir, "acuity", _g1_block(study, obj_names, "acuity"),
                         fname=G1_TUNED_FNAME)


def run_disposition_stage_g1(args, gcfg, optuna, X_train, y_train, out_dir):
    """Multi-objective disposition Group-1: maximize accuracy, minimize
    under-triage, over (scale_pos_weight exponent, lr, n_estimators, decision
    threshold). predicted_acuity is cascaded from an INCUMBENT acuity model
    (frozen G1+G2) so the study isolates the disposition Group-1 levers."""
    from optuna.samplers import NSGAIISampler
    y_acuity = y_train["acuity"].reset_index(drop=True)
    y_admit = y_train["admitted"].reset_index(drop=True)
    acu_tr, acu_val, ya_tr, ya_val = train_test_split(
        X_train, y_acuity, test_size=0.2, random_state=1, stratify=y_acuity,
    )
    print("  Building predicted_acuity cascade from the incumbent acuity model ...")
    acu_model = train_acuity(acu_tr, ya_tr, acu_val, ya_val, FROZEN_GROUP2, gcfg,
                             "    acuity (cascade, incumbent)",
                             eval_metric=SEARCH_EVAL_METRIC)
    X_disp = X_train.copy()
    X_disp["predicted_acuity"] = acu_model.predict(X_train) + 1

    Xd_tr, Xd_val, yd_tr, yd_val = train_test_split(
        X_disp, y_admit, test_size=0.2, random_state=1, stratify=y_admit,
    )
    print(f"  Inner split (tuning): train={len(Xd_tr):,}, val={len(Xd_val):,}")
    obj_names = ["accuracy", "under_rate"]

    def objective(trial):
        p = suggest_group1_disposition(trial)
        g = _g1_disp_gcfg(gcfg, p)
        print(f"\n  >>> trial #{trial.number:3d} | lr={p['learning_rate']:.4f} "
              f"n_est={p['n_estimators']} spw_exp={p['scale_pos_weight_exponent']:.2f} "
              f"thr={p['decision_threshold']:.2f}", flush=True)
        model = train_disposition(Xd_tr, yd_tr, Xd_val, yd_val, FROZEN_GROUP2, g,
                                  f"    trial #{trial.number} disp")
        y_prob = model.predict_proba(Xd_val)[:, 1]
        m = disposition_metrics_at_threshold(yd_val, y_prob, p["decision_threshold"])
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        if trial.study.user_attrs.get("baseline_under_rate") is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
        return m["accuracy"], m["under_rate"]

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=args.study_name or DISP_G1_STUDY_NAME,
        storage=args.storage, load_if_exists=True,
        sampler=NSGAIISampler(seed=42),
    )
    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial(FROZEN_GROUP1_DISP)
        print("  Enqueued incumbent Group-1 config as trial 0 (sits on the frontier).")

    log_path = out_dir / "tuning_log_disposition_g1.json"
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[make_trial_logger_g1(log_path, "disposition", out_dir, obj_names)],
                   gc_after_trial=True)
    _print_pareto(study, obj_names, "disposition")
    _update_tuned_params(out_dir, "disposition", _g1_block(study, obj_names, "disposition"),
                         fname=G1_TUNED_FNAME)


def _print_pareto(study, obj_names, stage):
    print("\n" + "=" * 60)
    front = study.best_trials
    print(f"  Pareto front ({stage}): {len(front)} non-dominated trial(s)")
    for t in sorted(front, key=lambda t: t.number):
        vstr = "  ".join(f"{n}={v:.4f}" for n, v in zip(obj_names, t.values))
        print(f"    trial #{t.number:3d} | {vstr}")
    sel = _g1_select(study, stage)
    if sel:
        print(f"  Chaining default (selected): trial #{sel['trial']} — {sel['rationale']}")


# ---------------------------------------------------------------------------
# Group-1 report — Pareto frontier on the held-out OUTER test split.
# ---------------------------------------------------------------------------
def _acuity_test_metrics(model, X_test, y_te) -> dict:
    pred = model.predict(X_test) + 1
    m = acuity_metrics(y_te, pred)
    m["esi5_recall"] = _esi5_recall(y_te, pred)
    m["per_class_recall"] = _per_class_recall(y_te, pred)
    return m


def run_report_stage_g1(gcfg, X_train, y_train, X_test, y_test, out_dir):
    """For every Pareto-front trial, retrain on the full outer-train with that
    Group-1 config (Group-2 frozen) and evaluate on the held-out outer-test;
    print the frontier with the incumbent flagged; write triage_hpo_g1_results.json."""
    live_acuity = TRIAGE_V3_DIR / "acuity_model.joblib"
    live_disp = TRIAGE_V3_DIR / "disposition_model.joblib"
    if not live_acuity.exists() or not live_disp.exists():
        print(f"  ERROR: live models not found in {TRIAGE_V3_DIR}. The Group-1 "
              f"report needs the deployed joblibs as the incumbent.", file=sys.stderr)
        sys.exit(2)
    tuned = _read_tuned_params(out_dir, G1_TUNED_FNAME) or {}

    y_acuity_tr = y_train["acuity"].reset_index(drop=True)
    y_admit_tr = y_train["admitted"].reset_index(drop=True)
    y_acuity_te = y_test["acuity"].reset_index(drop=True)
    y_admit_te = y_test["admitted"].reset_index(drop=True)
    results = {"generated_at": datetime.now().isoformat(),
               "test_rows": int(len(X_test)), "mode": "multi-objective"}

    # ---- Acuity frontier ----
    print("\n  [acuity] incumbent (live iter-2 model) on outer-test ...")
    inc_model = joblib.load(live_acuity)
    inc_m = _acuity_test_metrics(inc_model, X_test, y_acuity_te)
    acu_block = {"objective_names": ["exact", "under_rate", "esi5_recall"],
                 "incumbent": inc_m, "frontier": []}
    if tuned.get("acuity", {}).get("pareto_front"):
        acu_tr, acu_val, ya_tr, ya_val = train_test_split(
            X_train, y_acuity_tr, test_size=0.2, random_state=1, stratify=y_acuity_tr,
        )
        for entry in tuned["acuity"]["pareto_front"]:
            if int(entry["trial"]) == 0:
                fm = dict(inc_m); fm["trial"] = 0; fm["is_incumbent"] = True
                acu_block["frontier"].append(fm); continue
            print(f"  [acuity] retrain Pareto trial #{entry['trial']} on outer-train ...")
            g = _g1_acuity_gcfg(gcfg, entry["params"])
            mdl = train_acuity(acu_tr, ya_tr, acu_val, ya_val, FROZEN_GROUP2, g,
                               f"    acuity (trial #{entry['trial']})",
                               eval_metric=SEARCH_EVAL_METRIC)
            fm = _acuity_test_metrics(mdl, X_test, y_acuity_te)
            fm["trial"] = int(entry["trial"]); fm["params"] = entry["params"]
            acu_block["frontier"].append(fm)
    acu_block["selected"] = tuned.get("acuity", {}).get("selected")
    results["acuity"] = acu_block

    # ---- Disposition frontier (cascade with the incumbent/live acuity) ----
    print("\n  [disposition] incumbent (live models) on outer-test ...")
    inc_disp = joblib.load(live_disp)
    X_test_casc = X_test.copy()
    X_test_casc["predicted_acuity"] = inc_model.predict(X_test) + 1
    inc_prob = inc_disp.predict_proba(X_test_casc)[:, 1]
    inc_dm = disposition_metrics_at_threshold(y_admit_te, inc_prob,
                                              FROZEN_GROUP1_DISP["decision_threshold"])
    disp_block = {"objective_names": ["accuracy", "under_rate"],
                  "incumbent": inc_dm, "frontier": []}
    if tuned.get("disposition", {}).get("pareto_front"):
        X_disp_tr = X_train.copy()
        X_disp_tr["predicted_acuity"] = inc_model.predict(X_train) + 1
        for entry in tuned["disposition"]["pareto_front"]:
            thr = float(entry["params"]["decision_threshold"])
            if int(entry["trial"]) == 0:
                fm = dict(inc_dm); fm["trial"] = 0; fm["is_incumbent"] = True
                disp_block["frontier"].append(fm); continue
            print(f"  [disposition] retrain Pareto trial #{entry['trial']} on outer-train ...")
            g = _g1_disp_gcfg(gcfg, entry["params"])
            Xd_tr, Xd_val, yd_tr, yd_val = train_test_split(
                X_disp_tr, y_admit_tr, test_size=0.2, random_state=1, stratify=y_admit_tr,
            )
            mdl = train_disposition(Xd_tr, yd_tr, Xd_val, yd_val, FROZEN_GROUP2, g,
                                    f"    disposition (trial #{entry['trial']})")
            prob = mdl.predict_proba(X_test_casc)[:, 1]
            fm = disposition_metrics_at_threshold(y_admit_te, prob, thr)
            fm["trial"] = int(entry["trial"]); fm["params"] = entry["params"]
            disp_block["frontier"].append(fm)
    disp_block["selected"] = tuned.get("disposition", {}).get("selected")
    results["disposition"] = disp_block

    out = out_dir / "triage_hpo_g1_results.json"
    _atomic_write_json(out, results)
    print(f"\n  Wrote {out}")
    _print_report_summary_g1(results)


def _print_report_summary_g1(results):
    a = results.get("acuity", {})
    if a.get("incumbent"):
        inc = a["incumbent"]
        print("\n  ── Acuity Pareto frontier (outer-test) ──")
        print(f"    incumbent (#0): exact={inc['exact']*100:.2f}%  "
              f"under={inc['under_rate']*100:.2f}%  esi5_recall={inc['esi5_recall']*100:.2f}%")
        for fm in a.get("frontier", []):
            if fm.get("is_incumbent"):
                continue
            print(f"    trial #{fm['trial']:3d}: exact={fm['exact']*100:.2f}%  "
                  f"under={fm['under_rate']*100:.2f}%  esi5_recall={fm['esi5_recall']*100:.2f}%")
    d = results.get("disposition", {})
    if d.get("incumbent"):
        inc = d["incumbent"]
        print("  ── Disposition Pareto frontier (outer-test) ──")
        print(f"    incumbent (#0): acc={inc['accuracy']*100:.2f}%  "
              f"under={inc['under_rate']*100:.2f}%  thr={inc['threshold']:.2f}")
        for fm in d.get("frontier", []):
            if fm.get("is_incumbent"):
                continue
            print(f"    trial #{fm['trial']:3d}: acc={fm['accuracy']*100:.2f}%  "
                  f"under={fm['under_rate']*100:.2f}%  thr={fm['threshold']:.2f}")


# ---------------------------------------------------------------------------
# Self-test (synthetic, local, no MIMIC / no GPU)
# ---------------------------------------------------------------------------
def run_selftest():
    print("=" * 60)
    print("  SELFTEST — synthetic data, CPU, throwaway paths")
    print("=" * 60)
    import optuna
    from optuna.samplers import TPESampler

    # Snapshot live-model mtimes (must be untouched afterwards).
    live = TRIAGE_V3_DIR / "acuity_model.joblib"
    live_mtime = live.stat().st_mtime if live.exists() else None

    rng = np.random.default_rng(0)
    n, d = 1500, 24
    X = pd.DataFrame(rng.standard_normal((n, d)).astype("float32"),
                     columns=[f"f{i}" for i in range(d)])
    y = pd.Series(rng.integers(1, 6, n), name="acuity")

    gcfg = {"acuity_n_estimators": 10, "acuity_learning_rate": 0.3,
            "acuity_early_stopping_rounds": 5}
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

    tmp = Path(tempfile.mkdtemp(prefix="triage_hpo_selftest_"))
    storage = f"sqlite:///{tmp / 'selftest.db'}"
    log_path = tmp / "tuning_log_acuity.json"

    def objective(trial):
        g = suggest_group2(trial)
        model = train_acuity(X_tr, y_tr, X_val, y_val, g, gcfg, "selftest")
        m = acuity_metrics(y_val, model.predict(X_val) + 1)
        base = trial.study.user_attrs.get("baseline_under_rate")
        if base is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
            trial.study.set_user_attr("baseline_qwk", m["qwk"])
            base = m["under_rate"]
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("constraint", [m["under_rate"] - base])
        return m["qwk"]

    sampler = TPESampler(seed=42, constraints_func=lambda t: t.user_attrs.get("constraint", [0.0]))
    study = optuna.create_study(direction="maximize", study_name="selftest",
                                storage=storage, load_if_exists=True, sampler=sampler)
    study.enqueue_trial(FROZEN_GROUP2)
    study.optimize(objective, n_trials=1,
                   callbacks=[make_trial_logger(log_path, "acuity", tmp)])

    assert (tmp / "selftest.db").exists(), "study DB not written"
    assert log_path.exists(), "JSON log not written"
    logged = json.loads(log_path.read_text())
    assert logged["n_complete"] >= 1, "no completed trial in log"
    assert "feasible" in logged["trials"][0], "constraint/feasible flag missing"
    # Resume: reopen the persisted study.
    reopened = optuna.load_study(study_name="selftest", storage=storage)
    assert len(reopened.trials) >= 1, "study did not persist for resume"
    # Live model untouched.
    if live_mtime is not None:
        assert live.stat().st_mtime == live_mtime, "live acuity model was modified!"

    print(f"  study DB:   {tmp / 'selftest.db'}  ({len(reopened.trials)} trial)")
    print(f"  JSON log:   {log_path}")
    print(f"  baseline_under_rate stored: {study.user_attrs['baseline_under_rate']:.4f}")
    print(f"  live models untouched: {'n/a' if live_mtime is None else 'yes'}")
    print("\n  SELFTEST PASSED")


def run_selftest_g1():
    """Synthetic CPU plumbing test for the multi-objective Group-1 path:
    NSGA-II study builds, trials complete, Pareto front + selection are
    extracted, tuned_params_triage_g1.json is written, live models untouched."""
    print("=" * 60)
    print("  SELFTEST (Group-1, multi-objective) — synthetic, CPU, throwaway paths")
    print("=" * 60)
    import optuna
    from optuna.samplers import NSGAIISampler

    live = TRIAGE_V3_DIR / "acuity_model.joblib"
    live_mtime = live.stat().st_mtime if live.exists() else None

    rng = np.random.default_rng(0)
    n, d = 1500, 24
    X = pd.DataFrame(rng.standard_normal((n, d)).astype("float32"),
                     columns=[f"f{i}" for i in range(d)])
    y = pd.Series(rng.integers(1, 6, n), name="acuity")
    gcfg = {"acuity_n_estimators": 10, "acuity_learning_rate": 0.3,
            "acuity_early_stopping_rounds": 5}
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

    tmp = Path(tempfile.mkdtemp(prefix="triage_hpo_g1_selftest_"))
    storage = f"sqlite:///{tmp / 'selftest_g1.db'}"
    log_path = tmp / "tuning_log_acuity_g1.json"
    obj_names = ["exact", "under_rate", "esi5_recall"]

    def objective(trial):
        p = suggest_group1_acuity(trial)
        g = _g1_acuity_gcfg(gcfg, p)
        model = train_acuity(X_tr, y_tr, X_val, y_val, FROZEN_GROUP2, g, "selftest",
                             eval_metric=SEARCH_EVAL_METRIC)
        yp = model.predict(X_val) + 1
        m = acuity_metrics(y_val, yp)
        m["esi5_recall"] = _esi5_recall(y_val, yp)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        if trial.study.user_attrs.get("baseline_under_rate") is None:
            trial.study.set_user_attr("baseline_under_rate", m["under_rate"])
        return m["exact"], m["under_rate"], m["esi5_recall"]

    study = optuna.create_study(
        directions=["maximize", "minimize", "maximize"], study_name="selftest_g1",
        storage=storage, load_if_exists=True, sampler=NSGAIISampler(seed=42),
    )
    study.enqueue_trial(FROZEN_GROUP1_ACUITY)
    study.optimize(objective, n_trials=4,
                   callbacks=[make_trial_logger_g1(log_path, "acuity", tmp, obj_names)])

    assert (tmp / "selftest_g1.db").exists(), "study DB not written"
    assert log_path.exists(), "JSON log not written"
    logged = json.loads(log_path.read_text())
    assert logged["mode"] == "multi-objective", "log not flagged multi-objective"
    assert "pareto_front_trials" in logged, "pareto front missing from log"
    assert len(study.best_trials) >= 1, "no Pareto-optimal trial"
    tp = _read_tuned_params(tmp, G1_TUNED_FNAME)
    assert tp and tp.get("acuity", {}).get("pareto_front"), "g1 frontier block not written"
    assert tp["acuity"].get("selected") is not None, "no chaining selection recorded"
    reopened = optuna.load_study(study_name="selftest_g1", storage=storage)
    assert len(reopened.trials) >= 1, "study did not persist for resume"
    if live_mtime is not None:
        assert live.stat().st_mtime == live_mtime, "live acuity model was modified!"

    print(f"  study DB:        {tmp / 'selftest_g1.db'}  ({len(reopened.trials)} trials)")
    print(f"  Pareto front:    {len(study.best_trials)} trial(s)")
    print(f"  selected (chain): trial #{tp['acuity']['selected']['trial']}")
    print(f"  live models untouched: {'n/a' if live_mtime is None else 'yes'}")
    print("\n  SELFTEST (Group-1) PASSED")


def _apply_group1_best(gcfg: dict, out_dir: Path) -> dict:
    """Chaining hook: override the frozen Group-1 fields in gcfg with the
    'selected' Pareto point recorded by the Group-1 sweep. Used by the Group-2
    re-run (--use-group1-best) so the regularization is searched on top of the
    confirmed Group-1 config rather than the original incumbent."""
    tuned = _read_tuned_params(out_dir, G1_TUNED_FNAME)
    if not tuned:
        print(f"  ERROR: --use-group1-best set but {out_dir / G1_TUNED_FNAME} not "
              f"found. Run the Group-1 sweep (--group group1) first.", file=sys.stderr)
        sys.exit(2)
    g = dict(gcfg)
    acu = (tuned.get("acuity") or {}).get("selected")
    disp = (tuned.get("disposition") or {}).get("selected")
    if acu and acu.get("params"):
        p = acu["params"]
        g["acuity_learning_rate"] = p["learning_rate"]
        g["acuity_n_estimators"] = int(p["n_estimators"])
        g["acuity_boost_map"] = _boost_map(p)
        print(f"  [chain] acuity Group-1 <- g1 trial #{acu.get('trial')}")
    if disp and disp.get("params"):
        p = disp["params"]
        g["disp_learning_rate"] = p["learning_rate"]
        g["disp_n_estimators"] = int(p["n_estimators"])
        g["disp_scale_pos_weight_exponent"] = p["scale_pos_weight_exponent"]
        print(f"  [chain] disposition Group-1 <- g1 trial #{disp.get('trial')}")
    if not acu and not disp:
        print(f"  ERROR: --use-group1-best set but no 'selected' block in "
              f"{G1_TUNED_FNAME}. Pick a Pareto point first.", file=sys.stderr)
        sys.exit(2)
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _make_gcfg(smoke: bool) -> dict:
    """Group-1 frozen training config. Smoke reduces trees purely for speed
    (throwaway numbers); full runs use the documented frozen constants."""
    if smoke:
        return {
            "acuity_n_estimators": 300, "acuity_learning_rate": 0.05,
            "acuity_early_stopping_rounds": 30,
            "disp_n_estimators": 300, "disp_learning_rate": 0.05,
            "disp_early_stopping_rounds": 30,
        }
    return {
        "acuity_n_estimators": ACUITY_N_ESTIMATORS,
        "acuity_learning_rate": ACUITY_LEARNING_RATE,
        "acuity_early_stopping_rounds": ACUITY_EARLY_STOPPING_ROUNDS,
        "disp_n_estimators": DISP_N_ESTIMATORS,
        "disp_learning_rate": DISP_LEARNING_RATE,
        "disp_early_stopping_rounds": DISP_EARLY_STOPPING_ROUNDS,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["acuity", "disposition", "report"],
                    default="acuity")
    ap.add_argument("--group", choices=["group1", "group2"], default="group2",
                    help="group2 (default): the inherited regularization knobs "
                         "(constrained single-objective). group1: the "
                         "clinical-safety levers (boost vector, lr/n_estimators, "
                         "disposition threshold/scale_pos_weight) as a "
                         "MULTI-OBJECTIVE Pareto study (NSGA-II).")
    ap.add_argument("--use-group1-best", action="store_true",
                    help="Group-2 only: freeze Group-1 at the 'selected' Pareto "
                         "point from a prior --group group1 sweep (chaining) "
                         "instead of the original incumbent.")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Real-data subsample, throwaway paths, fast trees.")
    ap.add_argument("--selftest", action="store_true",
                    help="Synthetic CPU plumbing test (no data/GPU).")
    ap.add_argument("--search-tree-cap", type=int, default=None,
                    help="Cap n_estimators during the acuity/disposition SEARCH "
                         "(e.g. 2500). QWK plateaus well before 5000 trees, so a "
                         "cap speeds trials without changing the ranking. The "
                         "report stage always re-fits the winner at full trees.")
    ap.add_argument("--storage", default=None, help="Optuna storage URL.")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    if args.selftest:
        run_selftest_g1() if args.group == "group1" else run_selftest()
        return

    # Re-read the device at call time so the importlib-as-module entry path also
    # honours an XGB_DEVICE set just before main() (runpy re-runs the module top
    # level, but a cached-module import would not).
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

    # Dedicated, non-colliding output paths (smoke -> throwaway subdir).
    g1 = args.group == "group1"
    out_dir = (TRIAGE_V3_HPO_DIR / "smoke") if args.smoke else TRIAGE_V3_HPO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.storage is None:
        stem = "optuna_triage_g1" if g1 else "optuna_triage"
        db = f"{stem}_smoke.db" if args.smoke else f"{stem}.db"
        args.storage = f"sqlite:///{out_dir / db}"
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        TRIAGE_TUNE_CACHE_DIR / "smoke" if args.smoke else TRIAGE_TUNE_CACHE_DIR
    )
    gcfg = _make_gcfg(args.smoke)
    # Chaining (Group-2 only): fold the confirmed Group-1 best into the frozen config.
    if args.use_group1_best:
        if g1:
            print("  NOTE: --use-group1-best is ignored for --group group1.")
        else:
            gcfg = _apply_group1_best(gcfg, out_dir)
    # Tree cap applies to the SEARCH stages only (report re-fits at full trees).
    if args.search_tree_cap and args.stage in ("acuity", "disposition"):
        gcfg["acuity_n_estimators"] = min(gcfg["acuity_n_estimators"], args.search_tree_cap)
        gcfg["disp_n_estimators"] = min(gcfg["disp_n_estimators"], args.search_tree_cap)

    print("=" * 60)
    print(f"  Triage v3 HPO — group={args.group} stage={args.stage}"
          f"{'  [SMOKE]' if args.smoke else ''}")
    print("=" * 60)
    print(f"  storage:   {args.storage}")
    print(f"  out_dir:   {out_dir}")
    print(f"  cache_dir: {cache_dir}")
    print(f"  device:    {XGB_DEVICE}  tree_method: {XGB_TREE_METHOD}")
    if args.search_tree_cap and args.stage in ("acuity", "disposition"):
        print(f"  search tree cap: {args.search_tree_cap}")
    if XGB_DEVICE == "cpu":
        print("  WARNING: device is CPU — training will be very slow. Set "
              "XGB_DEVICE=cuda (and use a GPU runtime) before running.")
    print(f"  REPORTING-ONLY: live models in {TRIAGE_V3_DIR} will NOT be modified.")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_or_build_cache(
        cache_dir, rebuild=args.rebuild_features,
        subsample=20000 if args.smoke else None,
    )

    if g1:
        if args.stage == "acuity":
            run_acuity_stage_g1(args, gcfg, optuna, X_train, y_train, out_dir)
        elif args.stage == "disposition":
            run_disposition_stage_g1(args, gcfg, optuna, X_train, y_train, out_dir)
        elif args.stage == "report":
            run_report_stage_g1(gcfg, X_train, y_train, X_test, y_test, out_dir)
    else:
        if args.stage == "acuity":
            run_acuity_stage(args, gcfg, optuna, TPESampler, X_train, y_train, out_dir)
        elif args.stage == "disposition":
            run_disposition_stage(args, gcfg, optuna, TPESampler, X_train, y_train, out_dir)
        elif args.stage == "report":
            run_report_stage(gcfg, X_train, y_train, X_test, y_test, out_dir)


if __name__ == "__main__":
    main()
