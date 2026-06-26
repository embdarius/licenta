"""Regenerate notebooks/tune_doctor_hpo_v3.ipynb.

Edit cell sources here, then run `python scripts/build_optuna_doctor_notebook.py`.

The notebook runs `scripts/tune_doctor_v3_heads.py` in chunks on Colab GPU. It is
a REPORTING-ONLY, constrained Group-2 hyperparameter sweep for the doctor v3
DEPARTMENT + DISPOSITION heads: it never overwrites the live doctor v3 model
joblibs (everything lands in `artifacts/doctor/v3/hpo/`), and every trial is
persisted to a SQLite study on Drive so runs resume across Colab sessions.

Mirrors scripts/build_optuna_triage_notebook.py cell-for-cell.
"""

import json
from pathlib import Path

cells = []


def md(cell_id: str, src: str) -> None:
    cells.append({
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": src,
    })


def code(cell_id: str, src: str) -> None:
    cells.append({
        "cell_type": "code", "id": cell_id, "metadata": {},
        "execution_count": None, "outputs": [],
        "source": src,
    })


md("intro-md", """# Doctor v3 - Constrained Optuna HPO (Colab GPU, resumable, reporting-only)

A **constrained** search over only the inherited **Group-2** XGBoost
regularization knobs (`max_depth`, `subsample`, `colsample_bytree`,
`colsample_bylevel`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`)
for the two **untuned** doctor heads: **department** (11-class) and
**disposition** (binary admit/discharge). Each head's documented **Group-1**
config is **frozen** (department: `lr=0.02`, `n_estimators=3000`,
`early_stopping=100`, sqrt-inverse class weighting; disposition: `lr=0.02`,
`n_estimators=5000`, `early_stopping=150`, `scale_pos_weight=sqrt(N_neg/N_pos)`).
The diagnosis head is already tuned by the older `scripts/tune_doctor_v3.py`.

**This is reporting-only.** Nothing here overwrites your existing Drive
artifacts:

- the live doctor v3 joblibs in `artifacts/doctor/v3/` are only ever **read**
  (`diagnosis_model.joblib`, `department_model.joblib`, `disposition_model.joblib`);
- all sweep outputs land in a new `artifacts/doctor/v3/hpo/` subdir;
- the department stage reuses the existing `data/derived/tune_cache/`; the
  disposition stage builds its own `data/derived/doctor_dispo_tune_cache/`;
- the Optuna study DB is `hpo/optuna_doctor.db` (distinct from the triage one).

## Objective
- **Department:** maximize **macro-F1** across the 11 service classes (matches
  the diagnosis sweep's "useful across categories" thesis). No constraint. The
  diagnosis-softmax cascade is built once per run and reused across trials.
- **Disposition:** maximize **ROC AUC** **subject to** a hard constraint -
  under-triage rate (admit predicted as discharge, threshold 0.5) must not
  exceed the incumbent's. The current hand-tuned config is enqueued as
  **trial 0** and defines that baseline. The dangerous error (missing an
  admission) can only *hold or improve*; it can never be traded away for AUC.

## How resume works
Every trial is committed to `hpo/optuna_doctor.db` on Drive before the next
starts (`load_if_exists=True`). Ctrl-C mid-trial or a Colab session reclaim only
loses the in-flight trial; re-run the tune cell to continue. A human-readable
`hpo/tuning_log_{stage}.json` is rewritten after every trial.

## Order of operations
1. Setup (Cells 1-5): mount, clone, symlink, install, GPU smoke.
2. **Self-test** (Cell 6) - synthetic plumbing check for both heads, seconds.
3. **Smoke** (Cell 7) - real-data subsample, 2 trials, throwaway paths.
4. **Department** (Cell 8) - run in 10-trial chunks.
5. **Disposition** (Cell 9) - run in 10-trial chunks (independent of department).
6. **State** (Cell 10) - read-only study inspection between chunks.
7. **Report** (Cell 11) - incumbent vs best on the outer-test split ->
   `hpo/doctor_hpo_results.json` (the thesis table).

## Prereq: Colab GPU runtime
Runtime menu -> Change runtime type -> **T4 GPU** (free) or L4/A100 (Pro).""")


md("section1-md", """---
## Section 1 - Setup
""")


code("cell-mount", """# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')""")


code("cell-clone", """# Cell 2: Clone (or pull) the repository
GITHUB_USERNAME = '<YOUR_USERNAME>'  # <- EDIT THIS
REPO_URL        = f'https://github.com/{GITHUB_USERNAME}/licenta.git'
CLONE_PATH      = '/content/licenta'
PROJECT_PATH    = f'{CLONE_PATH}/proiect_licenta'
BRANCH          = 'main'

import os, subprocess

if os.path.exists(CLONE_PATH):
    subprocess.run(['git', '-C', CLONE_PATH, 'fetch'], check=False)
    subprocess.run(['git', '-C', CLONE_PATH, 'checkout', BRANCH], check=False)
    subprocess.run(['git', '-C', CLONE_PATH, 'pull'], check=False)
else:
    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, CLONE_PATH], check=True)

assert os.path.exists(PROJECT_PATH)
subprocess.run(['git', '-C', CLONE_PATH, 'log', '-1', '--oneline'])
print(f'\\nProject root: {PROJECT_PATH}')""")


code("cell-symlink", """# Cell 3: Symlink data/ and artifacts/ from Drive (idempotent; never overwrites)
DRIVE_PROJECT = '/content/drive/MyDrive/proiect_licenta'  # <- EDIT IF NEEDED

for folder in ('data', 'artifacts'):
    src  = f'{DRIVE_PROJECT}/{folder}'
    dest = f'{PROJECT_PATH}/{folder}'
    if os.path.islink(dest):
        print(f'Symlink ok: {dest}')
    elif os.path.exists(dest):
        print(f'WARNING: {dest} is a real dir, skipping')
    else:
        if not os.path.exists(src):
            print(f'WARNING: source {src} does not exist on Drive')
            continue
        os.symlink(src, dest)
        print(f'Created: {dest} -> {src}')""")


code("cell-install", """# Cell 4: Install dependencies (package + Optuna)
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], check=True)

pip('-e', PROJECT_PATH)
pip('xgboost>=2.0.0')
pip('thefuzz[speedup]>=0.22.0')
pip('tqdm>=4.66.0', 'ipywidgets>=8.0.0')
pip('scikit-learn>=1.6.0')
pip('optuna>=3.4.0')
pip('pyarrow>=14.0.0')  # parquet feature cache

print('Dependencies installed.')""")


code("cell-gpu-smoke", """# Cell 5: GPU + XGBoost CUDA smoke test (same gate as the training notebook)
import subprocess, numpy as np
r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
print(r.stdout.strip() or '(nvidia-smi -L produced no output)')
assert 'GPU' in r.stdout, 'No GPU detected - set Runtime > Change runtime type > T4 GPU'

import xgboost as xgb, optuna
print(f'\\nXGBoost: {xgb.__version__}')
print(f'Optuna:  {optuna.__version__}')

X = np.random.randn(2_000, 64).astype('float32')
y = np.random.randint(0, 5, 2_000)
m = xgb.XGBClassifier(
    n_estimators=20, tree_method='hist', device='cuda',
    eval_metric='mlogloss', verbosity=0, n_jobs=-1,
)
m.fit(X, y)
print('XGBoost GPU smoke test PASSED')""")


md("section2-md", """---
## Section 2 - Pre-flight (run BEFORE the full sweep)

These two cells prove the pipeline runs and saves correctly without committing
to a multi-hour run. Both write only to throwaway paths.""")


code("cell-selftest", """# Cell 6: Self-test - synthetic data, CPU, seconds. No MIMIC, no GPU needed.
# Validates the whole plumbing chain for BOTH heads: study DB + JSON log +
# constraint logic + resume, and asserts the live models are untouched.
import runpy, sys
_saved = sys.argv
try:
    sys.argv = ['tune_doctor_v3_heads.py', '--selftest']
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_doctor_v3_heads.py', run_name='__main__')
finally:
    sys.argv = _saved""")


code("cell-smoke", """# Cell 7: Smoke run - REAL data subsample (~20K rows), 2 trials, fast trees.
# Writes ONLY to artifacts/doctor/v3/hpo/smoke/ and smoke feature caches.
# The department smoke reuses/derives a subsample of the diagnosis/department
# features; the disposition smoke parses discharge.csv on the subsample. Proves
# cache + GPU training + JSON save + resume end-to-end. NOT used for the thesis.
import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

_saved = sys.argv
for stage in ('department', 'disposition'):
    try:
        sys.argv = ['tune_doctor_v3_heads.py', '--stage', stage, '--smoke', '--n-trials', '2']
        runpy.run_path(f'{PROJECT_PATH}/scripts/tune_doctor_v3_heads.py', run_name='__main__')
    except KeyboardInterrupt:
        print('\\n[stopped by user. Study persists.]')
        break
    finally:
        sys.argv = _saved""")


md("section3-md", """---
## Section 3 - Run the sweep (resumable, full Group-1 config)

Set `N_TRIALS_THIS_SESSION` and run. Each run appends trials to the same study.
**Recommended total budget: 30-50 trials per head.** The department stage's first
run trains a diagnosis model (for the cascade) once per session; the disposition
stage's first run builds the full 425K feature cache (slow, parses the 3.3 GB
discharge.csv), then later runs load it in seconds.

The two heads are **independent** - run them in either order.""")


code("cell-tune-dept", """# Cell 8: Department head - macro-F1. Run in chunks; safe to ctrl-C.
N_TRIALS_THIS_SESSION = 10    # <- adjust per session
TIMEOUT_SECONDS       = None  # <- or e.g. 3600 to bound the session
REBUILD_FEATURES      = False # <- True only if the cache is stale

import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

argv = ['tune_doctor_v3_heads.py', '--stage', 'department', '--n-trials', str(N_TRIALS_THIS_SESSION)]
if TIMEOUT_SECONDS is not None:
    argv += ['--timeout', str(TIMEOUT_SECONDS)]
if REBUILD_FEATURES:
    argv += ['--rebuild-features']

_saved = sys.argv
try:
    sys.argv = argv
    print(f'running: {\" \".join(argv)}')
    print('(ctrl-C to stop early; SQLite study persists on Drive)\\n')
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_doctor_v3_heads.py', run_name='__main__')
except KeyboardInterrupt:
    print('\\n[stopped by user. Current trial marked FAIL; study persists.]')
finally:
    sys.argv = _saved""")


code("cell-tune-disp", """# Cell 9: Disposition head - constrained ROC AUC. Run in chunks; safe to ctrl-C.
N_TRIALS_THIS_SESSION = 10
TIMEOUT_SECONDS       = None
REBUILD_FEATURES      = False

import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

argv = ['tune_doctor_v3_heads.py', '--stage', 'disposition', '--n-trials', str(N_TRIALS_THIS_SESSION)]
if TIMEOUT_SECONDS is not None:
    argv += ['--timeout', str(TIMEOUT_SECONDS)]
if REBUILD_FEATURES:
    argv += ['--rebuild-features']

_saved = sys.argv
try:
    sys.argv = argv
    print(f'running: {\" \".join(argv)}\\n')
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_doctor_v3_heads.py', run_name='__main__')
except KeyboardInterrupt:
    print('\\n[stopped by user. Study persists.]')
finally:
    sys.argv = _saved""")


md("section4-md", """---
## Section 4 - Inspect & report
""")


code("cell-state", """# Cell 10: Inspect study state (read-only; runs no new trials)
import optuna, json
from proiect_licenta.paths import DOCTOR_V3_HPO_DIR

storage = f'sqlite:///{DOCTOR_V3_HPO_DIR / "optuna_doctor.db"}'

def show(study_name, constrained):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f'[{study_name}] no study yet'); return
    complete = [t for t in study.trials if t.state.name == 'COMPLETE']
    print(f'\\n[{study_name}] trials: {len(study.trials)} (complete {len(complete)})')
    if not complete:
        return
    if constrained:
        base = study.user_attrs.get('baseline_under_rate')
        if base is not None:
            print(f'  baseline under-triage: {base*100:.2f}%')
        feas = [t for t in complete if t.user_attrs.get('constraint', [1])[0] <= 1e-9]
        print(f'  feasible trials: {len(feas)}/{len(complete)}')
        pool = feas or complete
        best = max(pool, key=lambda t: t.value)
        tag = 'best FEASIBLE' if feas else 'best (NONE feasible yet)'
        print(f'  {tag}: #{best.number} auc={best.value:.4f} '
              f'under={best.user_attrs.get("under_rate", 0)*100:.2f}%')
    else:
        best = max(complete, key=lambda t: t.value)
        print(f'  best: #{best.number} macro_f1={best.value:.4f} '
              f'acc={best.user_attrs.get("accuracy", 0)*100:.2f}%')
    for k, v in best.params.items():
        print(f'    {k:18s} = {v}')

show('doctor_department_macro_f1', constrained=False)
show('doctor_disposition_auc', constrained=True)

tp = DOCTOR_V3_HPO_DIR / 'tuned_params_doctor.json'
print(f'\\ntuned_params_doctor.json present: {tp.exists()}')""")


code("cell-report", """# Cell 11: Final report - incumbent vs best on the OUTER test split.
# Loads the live calibrated models read-only as the incumbents, retrains +
# calibrates the best config on outer-train, evaluates both on the same held-out
# test rows, and writes hpo/doctor_hpo_results.json (the thesis table).
import os, sys, runpy, json
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

_saved = sys.argv
try:
    sys.argv = ['tune_doctor_v3_heads.py', '--stage', 'report']
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_doctor_v3_heads.py', run_name='__main__')
finally:
    sys.argv = _saved

from proiect_licenta.paths import DOCTOR_V3_HPO_DIR, DOCTOR_V3_DIR
print('\\nResults JSON:', DOCTOR_V3_HPO_DIR / 'doctor_hpo_results.json')
# Confirm the live models were NOT modified by this notebook.
import os
for f in ('diagnosis_model.joblib', 'department_model.joblib', 'disposition_model.joblib'):
    p = DOCTOR_V3_DIR / f
    if p.exists():
        print(f'  live {f}: mtime {os.path.getmtime(p)} (unchanged by HPO)')""")


md("done-md", """---
## Done

`artifacts/doctor/v3/hpo/` now holds:

- `optuna_doctor.db` - the persistent Optuna study (department + disposition).
- `tuning_log_department.json`, `tuning_log_disposition.json` - per-trial running
  logs (every completed trial + feasibility flag for disposition). Use the
  disposition log to plot **ROC AUC vs under-triage** across trials - a clean
  thesis figure showing the trade-off the constraint refuses to make.
- `tuned_params_doctor.json` - best department + best-feasible disposition Group-2.
- `doctor_hpo_results.json` - incumbent vs best on the outer-test split (the
  table to drop into the thesis).

The live doctor v3 models are untouched. Most likely the search confirms the
inherited Group-2 values are near-optimal - itself a reportable, honest
defensibility result, now validated under a clinical-safety constraint for the
disposition head.""")


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {
            "name": "tune_doctor_hpo_v3.ipynb", "provenance": [],
            "accelerator": "GPU",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}


def main():
    out = Path(__file__).resolve().parents[1] / "notebooks" / "tune_doctor_hpo_v3.ipynb"
    out.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
