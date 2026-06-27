"""Regenerate notebooks/tune_triage_v3.ipynb.

Edit cell sources here, then run `python scripts/build_optuna_triage_notebook.py`.

The notebook runs `scripts/tune_triage_v3.py` in chunks on Colab GPU. It is a
REPORTING-ONLY, constrained Group-2 hyperparameter sweep for the triage v3
acuity + disposition heads: it never overwrites the live triage v3 model
joblibs (everything lands in `artifacts/triage/v3/hpo/`), and every trial is
persisted to a SQLite study on Drive so runs resume across Colab sessions.

Mirrors scripts/build_optuna_notebook.py (the doctor sweep) cell-for-cell.
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


md("intro-md", """# Triage v3 - Constrained Optuna HPO (Colab GPU, resumable, reporting-only)

A **constrained** search over only the inherited **Group-2** XGBoost
regularization knobs (`max_depth`, `subsample`, `colsample_bytree`,
`colsample_bylevel`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`).
The documented **Group-1** clinical-safety choices (`lr=0.01`,
`n_estimators=5000`, `early_stopping=150`, `ESI_EXTREME_BOOST`,
`neg_quadratic_kappa`) are **frozen**.

**This is reporting-only.** Nothing here overwrites your existing Drive
artifacts:

- the live triage v3 joblibs in `artifacts/triage/v3/` are only ever **read**;
- all sweep outputs land in a new `artifacts/triage/v3/hpo/` subdir;
- the feature cache is a triage-only `data/derived/triage_tune_cache/`
  (separate from the doctor sweep's `tune_cache/`);
- the Optuna study DB is `hpo/optuna_triage.db` (distinct from the doctor's).

## Objective
- **Acuity:** maximize quadratic-weighted kappa (QWK) **subject to** a hard
  constraint - under-triage rate must not exceed the incumbent's. The current
  hand-tuned config is enqueued as **trial 0** and defines that baseline. So
  the search can only *hold or improve* under-triage; it can never trade it
  away for headline accuracy.
- **Disposition:** maximize ROC AUC (binary; no safety constraint).

## How resume works
Every trial is committed to `hpo/optuna_triage.db` on Drive before the next
starts (`load_if_exists=True`). Ctrl-C mid-trial or a Colab session reclaim
only loses the in-flight trial; re-run the tune cell to continue. A
human-readable `hpo/tuning_log_{stage}.json` is rewritten after every trial.

## Order of operations
1. Setup (Cells 1-5): mount, clone, symlink, install, GPU smoke.
2. **Self-test** (Cell 6) - synthetic plumbing check, seconds.
3. **Smoke** (Cell 7) - real-data subsample, 2 trials, throwaway paths. Proves
   it runs + saves + resumes BEFORE the full sweep.
4. **Acuity** (Cell 8) - run in 10-trial chunks.
5. **Disposition** (Cell 9) - after acuity is finalized.
6. **State** (Cell 10) - read-only study inspection between chunks.
7. **Report** (Cell 11) - incumbent vs best-feasible on the outer-test split ->
   `hpo/triage_hpo_results.json` (the thesis table).

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
# Validates the whole plumbing chain: study DB + JSON log + constraint logic +
# resume, and asserts the live models are untouched.
import runpy, sys
_saved = sys.argv
try:
    sys.argv = ['tune_triage_v3.py', '--selftest']
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
finally:
    sys.argv = _saved""")


code("cell-smoke", """# Cell 7: Smoke run - REAL data subsample (~20K rows), 2 trials, fast trees.
# Writes ONLY to artifacts/triage/v3/hpo/smoke/ and a smoke feature cache.
# First run builds the subsample cache (a few min, parses discharge.csv);
# proves cache + GPU training + JSON save + resume end-to-end. Smoke numbers
# are NOT used for the thesis.
import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

_saved = sys.argv
try:
    sys.argv = ['tune_triage_v3.py', '--stage', 'acuity', '--smoke', '--n-trials', '2']
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
except KeyboardInterrupt:
    print('\\n[stopped by user. Study persists.]')
finally:
    sys.argv = _saved""")


md("section3-md", """---
## Section 3 - Run the sweep (resumable, full Group-1 config)

Set `N_TRIALS_THIS_SESSION` and run. Each run appends trials to the same study.
**Recommended total budget: 40-60 trials per head.** The first real run builds
the full feature cache (slow, parses the 3.3 GB discharge.csv); later runs load
it in seconds.

Run **acuity to completion first** (the disposition stage needs the best acuity
config to build its `predicted_acuity` cascade feature).""")


code("cell-tune-acuity", """# Cell 8: Acuity head - constrained QWK. Run in chunks; safe to ctrl-C.
N_TRIALS_THIS_SESSION = 10    # <- adjust per session
TIMEOUT_SECONDS       = None  # <- or e.g. 3600 to bound the session
REBUILD_FEATURES      = False # <- True only if the cache is stale

import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

argv = ['tune_triage_v3.py', '--stage', 'acuity', '--n-trials', str(N_TRIALS_THIS_SESSION)]
if TIMEOUT_SECONDS is not None:
    argv += ['--timeout', str(TIMEOUT_SECONDS)]
if REBUILD_FEATURES:
    argv += ['--rebuild-features']

_saved = sys.argv
try:
    sys.argv = argv
    print(f'running: {\" \".join(argv)}')
    print('(ctrl-C to stop early; SQLite study persists on Drive)\\n')
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
except KeyboardInterrupt:
    print('\\n[stopped by user. Current trial marked FAIL; study persists.]')
finally:
    sys.argv = _saved""")


code("cell-tune-disp", """# Cell 9: Disposition head - ROC AUC. Run AFTER acuity is finalized
# (it reads the best acuity config from hpo/tuned_params_triage.json).
N_TRIALS_THIS_SESSION = 10
TIMEOUT_SECONDS       = None

import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

argv = ['tune_triage_v3.py', '--stage', 'disposition', '--n-trials', str(N_TRIALS_THIS_SESSION)]
if TIMEOUT_SECONDS is not None:
    argv += ['--timeout', str(TIMEOUT_SECONDS)]

_saved = sys.argv
try:
    sys.argv = argv
    print(f'running: {\" \".join(argv)}\\n')
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
except KeyboardInterrupt:
    print('\\n[stopped by user. Study persists.]')
finally:
    sys.argv = _saved""")


md("section4-md", """---
## Section 4 - Inspect & report
""")


code("cell-state", """# Cell 10: Inspect study state (read-only; runs no new trials)
import optuna, json
from proiect_licenta.paths import TRIAGE_V3_HPO_DIR

storage = f'sqlite:///{TRIAGE_V3_HPO_DIR / "optuna_triage.db"}'

def show(study_name, constrained):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f'[{study_name}] no study yet'); return
    complete = [t for t in study.trials if t.state.name == 'COMPLETE']
    print(f'\\n[{study_name}] trials: {len(study.trials)} (complete {len(complete)})')
    if not complete:
        return
    base = study.user_attrs.get('baseline_under_rate')
    if base is not None:
        print(f'  baseline under-triage: {base*100:.2f}%  '
              f'qwk: {study.user_attrs.get("baseline_qwk", 0):.4f}')
    if constrained:
        feas = [t for t in complete if t.user_attrs.get('constraint', [1])[0] <= 1e-9]
        print(f'  feasible trials: {len(feas)}/{len(complete)}')
        pool = feas or complete
        best = max(pool, key=lambda t: t.value)
        tag = 'best FEASIBLE' if feas else 'best (NONE feasible yet)'
        print(f'  {tag}: #{best.number} qwk={best.value:.4f} '
              f'under={best.user_attrs.get("under_rate", 0)*100:.2f}%')
    else:
        best = max(complete, key=lambda t: t.value)
        print(f'  best: #{best.number} auc={best.value:.4f} '
              f'acc={best.user_attrs.get("accuracy", 0)*100:.2f}%')
    for k, v in best.params.items():
        print(f'    {k:18s} = {v}')

show('triage_acuity_qwk', constrained=True)
show('triage_disposition_auc', constrained=False)

tp = TRIAGE_V3_HPO_DIR / 'tuned_params_triage.json'
print(f'\\ntuned_params_triage.json present: {tp.exists()}')""")


code("cell-report", """# Cell 11: Final report - incumbent vs best-feasible on the OUTER test split.
# Loads the live iter-2 models read-only as the incumbent, retrains the best
# config on outer-train, evaluates both on the same held-out test rows, and
# writes hpo/triage_hpo_results.json (the thesis table).
import os, sys, runpy, json
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

_saved = sys.argv
try:
    sys.argv = ['tune_triage_v3.py', '--stage', 'report']
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
finally:
    sys.argv = _saved

from proiect_licenta.paths import TRIAGE_V3_HPO_DIR, TRIAGE_V3_DIR
print('\\nResults JSON:', TRIAGE_V3_HPO_DIR / 'triage_hpo_results.json')
# Confirm the live models were NOT modified by this notebook.
import os
for f in ('acuity_model.joblib', 'disposition_model.joblib'):
    p = TRIAGE_V3_DIR / f
    if p.exists():
        print(f'  live {f}: mtime {os.path.getmtime(p)} (unchanged by HPO)')""")


md("section5-md", """---
## Section 5 - Group-1 study (the clinical-safety levers; MULTI-OBJECTIVE)

A separate study over the **Group-1** knobs that Sections 3-4 held frozen: the
`ESI_EXTREME_BOOST` vector, `learning_rate`/`n_estimators`, and the disposition
`scale_pos_weight` + decision threshold. Because these *are* the clinical-safety
levers, this is a **multi-objective Pareto** search (NSGA-II), not a constrained
single-objective one:

- **Acuity:** jointly maximize (exact accuracy, ESI-5 recall) and minimize
  under-triage. Reports the **Pareto frontier**; the incumbent is enqueued as
  trial 0 and sits on the curve.
- **Disposition:** jointly maximize accuracy and minimize under-triage over the
  `scale_pos_weight` exponent, `lr`/`n_estimators`, and the decision threshold.

Group-2 stays frozen at the incumbent throughout. Everything lands in **separate**
files (`optuna_triage_g1.db`, `tuning_log_*_g1.json`, `tuned_params_triage_g1.json`,
`triage_hpo_g1_results.json`) so the Group-2 study above is untouched. Still
**reporting-only**.

After you inspect the frontier and confirm a deployed point (the `selected`
block in `tuned_params_triage_g1.json`), you can optionally re-run the **Group-2**
cells above with `--use-group1-best` to search the regularization on top of the
confirmed Group-1 config.""")


code("cell-selftest-g1", """# Cell 12: Group-1 self-test - synthetic, CPU, seconds. Exercises the
# multi-objective NSGA-II path (study builds, Pareto front + selection extracted)
# and asserts the live models are untouched.
import runpy, sys
_saved = sys.argv
try:
    sys.argv = ['tune_triage_v3.py', '--group', 'group1', '--selftest']
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
finally:
    sys.argv = _saved""")


code("cell-tune-acuity-g1", """# Cell 13: Acuity Group-1 (multi-objective Pareto). Run in chunks; safe to ctrl-C.
# Reuses the same feature cache as the Group-2 sweep; writes to *_g1 files only.
N_TRIALS_THIS_SESSION = 10    # <- adjust per session (40-60 total recommended)
TIMEOUT_SECONDS       = None

import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

argv = ['tune_triage_v3.py', '--group', 'group1', '--stage', 'acuity',
        '--n-trials', str(N_TRIALS_THIS_SESSION)]
if TIMEOUT_SECONDS is not None:
    argv += ['--timeout', str(TIMEOUT_SECONDS)]

_saved = sys.argv
try:
    sys.argv = argv
    print('running:', ' '.join(argv), '\\n')
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
except KeyboardInterrupt:
    print('\\n[stopped by user. Study persists.]')
finally:
    sys.argv = _saved""")


code("cell-tune-disp-g1", """# Cell 14: Disposition Group-1 (multi-objective Pareto). Independent of the
# acuity Group-1 run (it cascades predicted_acuity from the INCUMBENT acuity).
N_TRIALS_THIS_SESSION = 10
TIMEOUT_SECONDS       = None

import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

argv = ['tune_triage_v3.py', '--group', 'group1', '--stage', 'disposition',
        '--n-trials', str(N_TRIALS_THIS_SESSION)]
if TIMEOUT_SECONDS is not None:
    argv += ['--timeout', str(TIMEOUT_SECONDS)]

_saved = sys.argv
try:
    sys.argv = argv
    print('running:', ' '.join(argv), '\\n')
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
except KeyboardInterrupt:
    print('\\n[stopped by user. Study persists.]')
finally:
    sys.argv = _saved""")


code("cell-state-g1", """# Cell 15: Inspect the Group-1 Pareto fronts (read-only; runs no new trials)
import optuna, json
from proiect_licenta.paths import TRIAGE_V3_HPO_DIR

storage = f'sqlite:///{TRIAGE_V3_HPO_DIR / "optuna_triage_g1.db"}'

def show_front(study_name, obj_names):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f'[{study_name}] no study yet'); return
    complete = [t for t in study.trials if t.state.name == 'COMPLETE']
    print(f'\\n[{study_name}] trials: {len(study.trials)} (complete {len(complete)})')
    front = study.best_trials
    print(f'  Pareto front: {len(front)} trial(s)')
    for t in sorted(front, key=lambda t: t.number):
        vals = '  '.join(f'{n}={v:.4f}' for n, v in zip(obj_names, t.values))
        print(f'    #{t.number:3d} | {vals}')

show_front('triage_acuity_g1', ['exact', 'under_rate', 'esi5_recall'])
show_front('triage_disposition_g1', ['accuracy', 'under_rate'])

tp = TRIAGE_V3_HPO_DIR / 'tuned_params_triage_g1.json'
print(f'\\ntuned_params_triage_g1.json present: {tp.exists()}')
if tp.exists():
    data = json.loads(tp.read_text())
    for head in ('acuity', 'disposition'):
        sel = data.get(head, {}).get('selected')
        if sel:
            print(f'  [{head}] chaining default -> trial #{sel[\"trial\"]}: {sel[\"rationale\"]}')""")


code("cell-report-g1", """# Cell 16: Group-1 report - Pareto frontier on the OUTER test split.
# Retrains every frontier point on outer-train (Group-2 frozen), evaluates on the
# held-out test rows, and writes hpo/triage_hpo_g1_results.json (the thesis table).
import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

_saved = sys.argv
try:
    sys.argv = ['tune_triage_v3.py', '--group', 'group1', '--stage', 'report']
    runpy.run_path(f'{PROJECT_PATH}/scripts/tune_triage_v3.py', run_name='__main__')
finally:
    sys.argv = _saved

from proiect_licenta.paths import TRIAGE_V3_HPO_DIR
print('\\nResults JSON:', TRIAGE_V3_HPO_DIR / 'triage_hpo_g1_results.json')""")


md("done-md", """---
## Done

`artifacts/triage/v3/hpo/` now holds:

- `optuna_triage.db` - the persistent Optuna study (acuity + disposition).
- `tuning_log_acuity.json`, `tuning_log_disposition.json` - per-trial running
  logs (every completed trial + feasibility flag). Use the acuity log to plot
  **QWK vs under-triage** across trials - a clean thesis figure showing the
  trade-off the constraint refuses to make.
- `tuned_params_triage.json` - best-feasible acuity + best disposition Group-2.
- `triage_hpo_results.json` - incumbent vs best-feasible on the outer-test
  split (the table to drop into the thesis).

The live triage v3 model is untouched. Most likely the search confirms the
inherited Group-2 values are near-optimal - itself a reportable, honest
defensibility result. The same recipe then extends to the doctor + nurse heads.

**Section 5 (Group-1)** adds the sibling multi-objective study; it writes the
parallel `*_g1` files (`optuna_triage_g1.db`, `tuning_log_*_g1.json`,
`tuned_params_triage_g1.json` with a `pareto_front` + `selected` block per head,
and `triage_hpo_g1_results.json`). Use the acuity frontier to plot **exact
accuracy vs under-triage vs ESI-5 recall** - the clinical trade-off the deployed
point must be justified against.""")


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {
            "name": "tune_triage_v3.ipynb", "provenance": [],
            "accelerator": "GPU",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}


def main():
    out = Path(__file__).resolve().parents[1] / "notebooks" / "tune_triage_v3.ipynb"
    out.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
