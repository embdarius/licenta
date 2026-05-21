"""A1 — Optuna hyperparameter sweep for the v3-nurse diagnosis model.

Objective: macro-F1 across the 13 diagnosis classes (NOT flat accuracy).
This explicitly trades a fraction of Digestive recall (~81%, dominant) for
Infectious / Other recall (~20%, minority) — the thesis story is "useful
across categories", not "high top-1 only".

Pause/resume is built in: Optuna persists every trial to a SQLite study
database in `artifacts/doctor/v3/optuna_study.db` (which lives on Drive
via the symlink). Each invocation runs `--n_trials` more trials (default
10) on top of the existing study. Ctrl-C mid-trial is safe: the failing
trial is marked FAIL and the study moves on.

After tuning, the best parameters are written to
`artifacts/doctor/v3/tuned_params.json`. `train_nurse_v3.train_model` reads
that file on the next training run (when `tuned_params.json` is present)
and uses the tuned config instead of the hand-picked defaults.

Usage on Colab (after symlinking artifacts/ to Drive):
    !XGB_DEVICE=cuda XGB_TREE_METHOD=hist python scripts/tune_doctor_v3.py \\
        --n-trials 10

Run multiple times (10 + 10 + 10 + ...) until best-trial macro-F1 plateaus.
Recommended budget: 30-50 trials total.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Make sure we run with the project on the path even when invoked directly
# (e.g. `python scripts/tune_doctor_v3.py` from the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from proiect_licenta.paths import DOCTOR_V3_DIR, DERIVED_DIR
from proiect_licenta.training.train_nurse_v3 import (
    XGB_DEVICE, XGB_TREE_METHOD,
    load_and_clean_data, build_features,
    CATCH_ALL_LABEL,
)


# ---------------------------------------------------------------------------
# Feature cache — building the full v3-nurse feature matrix is slow (~30 min
# CPU, dominated by the discharge.csv PMH parse). Cache to parquet on Drive
# so re-tuning sessions reuse it. `--rebuild-features` forces regeneration.
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR = DERIVED_DIR / "tune_cache"


def load_or_build_cache(cache_dir: Path, rebuild: bool = False):
    """Return (X, y_diag, y_dept, diagnosis_labels, department_labels)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fx = cache_dir / "features.parquet"
    fy = cache_dir / "targets.parquet"
    fmeta = cache_dir / "meta.json"

    if not rebuild and fx.exists() and fy.exists() and fmeta.exists():
        print(f"[cache] Loading from {cache_dir}")
        X = pd.read_parquet(fx)
        targets = pd.read_parquet(fy)
        meta = json.loads(fmeta.read_text())
        return (
            X, targets["y_diag"], targets["y_dept"],
            meta["diagnosis_labels"], meta["department_labels"],
        )

    print(f"[cache] Building feature matrix from raw data (this is the slow part)")
    df = load_and_clean_data()
    features = build_features(df)

    diagnosis_labels = sorted(df["diagnosis_group"].unique())
    department_labels = sorted(df["service_group"].unique())
    assert CATCH_ALL_LABEL not in diagnosis_labels

    diag_map = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_map = {l: i for i, l in enumerate(department_labels)}
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True).astype("int32")
    y_dept = df["service_group"].map(dept_map).reset_index(drop=True).astype("int32")

    # Reduce float64 -> float32 to halve the parquet file size. XGBoost is
    # fine with float32 and the precision loss is negligible for tuning.
    for c in features.columns:
        if features[c].dtype == "float64":
            features[c] = features[c].astype("float32")

    print(f"[cache] Writing parquet ({len(features):,} rows, {features.shape[1]} cols)")
    features.to_parquet(fx, index=False)
    pd.DataFrame({"y_diag": y_diag, "y_dept": y_dept}).to_parquet(fy, index=False)
    fmeta.write_text(json.dumps({
        "diagnosis_labels": diagnosis_labels,
        "department_labels": department_labels,
        "built_at": datetime.now().isoformat(),
        "n_rows": int(len(features)),
        "n_cols": int(features.shape[1]),
    }, indent=2))
    print(f"[cache] Done. Subsequent runs read from {cache_dir} in seconds.")
    return features, y_diag, y_dept, diagnosis_labels, department_labels


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(
    X_inner_train, y_inner_train,
    X_inner_val, y_inner_val,
    n_classes,
):
    """Return a callable Optuna objective.

    Inner CV strategy: single 75K/20K train/val split (random_state=1 so it
    doesn't overlap with the train_nurse_v3 outer 80/20 random_state=42
    test split). Cheap; macro-F1 is the score returned to Optuna.
    """
    n_train = len(y_inner_train)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
        # Class-weight exponent p in `weight = (N / (K * count[c])) ** p`.
        # p=0.5 reproduces the current sqrt(inverse-freq) default; p=1.0 is
        # full inverse-freq (heavier minority weighting); p=0.0 is uniform.
        cw_exponent = trial.suggest_float("class_weight_exponent", 0.4, 1.0)

        class_counts = y_inner_train.value_counts()
        sample_weights = y_inner_train.map(
            lambda x: (n_train / (n_classes * class_counts[x])) ** cw_exponent
        )

        model = XGBClassifier(
            n_estimators=3000,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", early_stopping_rounds=100,
            random_state=42, n_jobs=-1, verbosity=0,
            subsample=0.8,
            tree_method=XGB_TREE_METHOD, device=XGB_DEVICE,
            **params,
        )
        model.fit(
            X_inner_train, y_inner_train,
            sample_weight=sample_weights,
            eval_set=[(X_inner_val, y_inner_val)],
            verbose=False,
        )

        y_pred = np.argmax(model.predict_proba(X_inner_val), axis=1)
        macro_f1 = f1_score(y_inner_val, y_pred, average="macro")

        # Useful telemetry on each trial: also log top-1 accuracy and best
        # iteration so we can sanity-check Optuna's choices.
        from sklearn.metrics import accuracy_score
        trial.set_user_attr("top1_accuracy", float(accuracy_score(y_inner_val, y_pred)))
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("class_weight_exponent", float(cw_exponent))

        return float(macro_f1)

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-trials", type=int, default=10,
                    help="Trials to run in this invocation (default 10). "
                         "Use multiple invocations to accumulate to 30-50.")
    ap.add_argument("--timeout", type=float, default=None,
                    help="Alternative termination: stop after N seconds. "
                         "Optuna picks the first of n_trials / timeout.")
    ap.add_argument("--rebuild-features", action="store_true",
                    help="Force rebuild of the feature cache (slow ~30 min).")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL. Default: "
                         "sqlite:///<DOCTOR_V3_DIR>/optuna_study.db")
    ap.add_argument("--study-name", default="diag_v3_macro_f1",
                    help="Optuna study name (allows multiple studies in one DB).")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help="Feature cache directory (parquet files).")
    args = ap.parse_args()

    # Lazy import so the rest of the script (cache loading, etc.) runs even
    # in environments without optuna — useful for debugging the cache path.
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("optuna is not installed. Install with `pip install optuna`.",
              file=sys.stderr)
        sys.exit(2)

    # Optuna's default INFO output is too chatty; bring it down to WARNING
    # except for the per-trial summaries we add via callbacks.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    DOCTOR_V3_DIR.mkdir(parents=True, exist_ok=True)
    storage_url = args.storage or f"sqlite:///{DOCTOR_V3_DIR / 'optuna_study.db'}"
    cache_dir = Path(args.cache_dir)

    print("=" * 60)
    print("  A1 — Optuna hyperparameter sweep (macro-F1 objective)")
    print("=" * 60)
    print(f"  storage:     {storage_url}")
    print(f"  study_name:  {args.study_name}")
    print(f"  n_trials:    {args.n_trials}")
    print(f"  timeout:     {args.timeout}")
    print(f"  cache_dir:   {cache_dir}")
    print(f"  device:      {XGB_DEVICE}  tree_method: {XGB_TREE_METHOD}")
    print("=" * 60)

    # 1. Load / build feature cache
    X, y_diag, y_dept, diagnosis_labels, department_labels = load_or_build_cache(
        cache_dir, rebuild=args.rebuild_features,
    )

    # 2. Reproduce the train_nurse_v3 outer 80/20 split (random_state=42)
    #    so the inner CV stays well clear of the held-out test set.
    X_train, X_test, y_diag_train, y_diag_test = train_test_split(
        X, y_diag, test_size=0.2, random_state=42, stratify=y_diag,
    )
    print(f"\n  Outer split (matches train_nurse_v3): "
          f"train={len(X_train):,}, test={len(X_test):,}")

    # 3. Inner train/val split for tuning — different random_state so this
    #    isn't accidentally correlated with the outer test split.
    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
        X_train, y_diag_train, test_size=0.2, random_state=1,
        stratify=y_diag_train,
    )
    print(f"  Inner split (tuning):                  "
          f"train={len(X_inner_train):,}, val={len(X_inner_val):,}")

    n_classes = len(diagnosis_labels)

    # 4. Open / reload study
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True),
    )
    print(f"\n  Existing trials in study: {len(study.trials)}")
    if study.trials:
        try:
            best = study.best_trial
            print(f"  Best so far: macro_f1={best.value:.4f}  "
                  f"(trial #{best.number}, "
                  f"top1={best.user_attrs.get('top1_accuracy')})")
        except ValueError:
            print("  (no completed trials yet)")

    # 5. Run more trials. Optuna persists each one to SQLite as it goes, so
    #    a Colab session reclaim or ctrl-C mid-trial only loses the
    #    currently-running trial (it's marked FAIL).
    objective = make_objective(
        X_inner_train, y_inner_train, X_inner_val, y_inner_val, n_classes,
    )

    def _per_trial_log(study, trial):
        if trial.state.name != "COMPLETE":
            return
        marker = "*" if trial.value == study.best_value else " "
        print(f"  trial #{trial.number:3d} {marker} macro_f1={trial.value:.4f}  "
              f"top1={trial.user_attrs.get('top1_accuracy', 0):.4f}  "
              f"best_iter={trial.user_attrs.get('best_iteration', '?')}  "
              f"cw_exp={trial.user_attrs.get('class_weight_exponent', '?'):.2f}")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[_per_trial_log],
        gc_after_trial=True,
    )

    # 6. Final summary + write tuned_params.json
    print("\n" + "=" * 60)
    print(f"  Tuning complete. Total trials in study: {len(study.trials)}")
    print("=" * 60)
    best = study.best_trial
    print(f"  Best macro_f1: {best.value:.4f}  (trial #{best.number})")
    print(f"  Top-1 acc at best: "
          f"{best.user_attrs.get('top1_accuracy', '?'):.4f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k:25s} = {v}")

    tuned_path = DOCTOR_V3_DIR / "tuned_params.json"
    payload = {
        "study_name": args.study_name,
        "best_trial": int(best.number),
        "best_macro_f1": float(best.value),
        "best_top1_accuracy": float(best.user_attrs.get("top1_accuracy", -1)),
        "best_iteration": int(best.user_attrs.get("best_iteration", -1)),
        "n_trials": int(len(study.trials)),
        "trained_on": "v3-nurse diagnosis (13 classes), inner CV 75/20 of the "
                       "train_nurse_v3 outer-train split (random_state=1).",
        "objective": "macro-F1 across 13 diagnosis classes",
        "params": best.params,
        "written_at": datetime.now().isoformat(),
    }
    tuned_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Wrote {tuned_path}")
    print(f"  train_nurse_v3.train_model will pick these up on the next "
          f"`uv run train_nurse_v3` (when tuned_params.json is present).")


if __name__ == "__main__":
    main()
