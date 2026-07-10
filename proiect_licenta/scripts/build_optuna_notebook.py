"""Regenerate notebooks/tune_optuna_v3.ipynb.

Edit cell sources here, then run `python scripts/build_optuna_notebook.py`.

The notebook runs `scripts/tune_doctor_v3.py` in chunks. Optuna persists
each trial to a SQLite study file on Drive (`artifacts/doctor/v3/optuna_study.db`)
so re-running the tune cell appends trials to the same study across Colab
sessions. Pause/resume works because the SQLite file survives Colab session
reclaim.
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


md("intro-md", """# A1 - Optuna Hyperparameter Tuning (v3 nurse diagnosis) on Colab GPU

Macro-F1 sweep over 9 XGBoost hyperparameters + a class-weight exponent, for
the v3-nurse 13-class diagnosis model. Persistent SQLite study on Drive
means you can run trials in chunks and resume across Colab sessions
without losing progress.

## How resume works

`scripts/tune_doctor_v3.py` opens an Optuna study with:

```python
optuna.create_study(
    storage=f"sqlite:///{artifacts/doctor/v3/optuna_study.db}",
    study_name="diag_v3_macro_f1",
    load_if_exists=True,
)
```

Every trial's params + result are committed to that SQLite file before the
next trial starts. The file lives in `artifacts/doctor/v3/` which is
symlinked to your Drive, so:

- Ctrl-C mid-trial: the failing trial is marked FAIL by Optuna; the next
  invocation skips it and continues.
- Colab session reclaimed: the SQLite file survives on Drive. Open a
  fresh session, re-run Cells 1-5 (mount, clone, symlink, install, GPU
  check), then Cell 6 (tune) - Optuna picks up where it left off.
- Inspecting progress: Cell 7 prints the current study state without
  running new trials.

Recommended total budget: 30-50 trials. The plan's expected lift is
+0.5 to +1.5pp diagnosis top-1, more on minority recall (Infectious /
Other / Blood) because the class-weight exponent is in the search space.

## Prereq: Colab GPU runtime

Runtime menu -> Change runtime type -> T4 GPU (free) or L4/A100 (Pro).

## Prereq: Drive + repo

Same layout as the main training notebook
(`notebooks/train_and_benchmark_v3.ipynb`). If you've already set up Drive
for training, you don't need to do anything extra here - the same
`MyDrive/proiect_licenta/{data,artifacts}` symlinks are reused.

## Recommended runtime per session

| Trials | Time on T4 GPU |
|---|---|
| 10 | ~30-50 min |
| 20 | ~60-100 min |
| 30 | ~90-150 min |

Trial time varies with the trial's `max_depth` and `min_child_weight` -
deeper / less-regularized trees take longer. Optuna's TPE sampler doesn't
care about wall-clock cost, so a few outlier trials will run long.""")


md("section1-md", """## Setup
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


code("cell-symlink", """# Cell 3: Symlink data/ and artifacts/ from Drive
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


md("section2-md", """## Run Optuna trials (resumable)

Set `N_TRIALS_THIS_SESSION` to how many trials you want this Colab session
to run, then run Cell 6. Each invocation of Cell 6 appends that many
trials to the existing study.

The first time you run Cell 6, the feature cache is built (~30 min) from
raw data. Subsequent runs reuse the cached parquet files in seconds.

You can also use `TIMEOUT_SECONDS` as an alternative termination - whichever
limit hits first stops the loop. Useful when you want to bound the session
to fit in Colab's reclaim window.""")


code("cell-tune", """# Cell 6: Run N more trials (resumable). Persistent SQLite study lives on
# Drive at artifacts/doctor/v3/optuna_study.db; re-running this cell adds
# to the same study. Safe to ctrl-C mid-trial; the failing trial is marked
# FAIL and the study moves on.
#
# The tune script is called in-process (runpy) rather than as a subprocess
# so stdout streams live to the notebook output AND any error surfaces with
# a full traceback. Tqdm bars + per-trial start lines provide progress.

N_TRIALS_THIS_SESSION = 10    # <- adjust per session (e.g. 5, 10, 20)
TIMEOUT_SECONDS       = None  # <- or e.g. 3600 for 1-hour bound
REBUILD_FEATURES      = False # <- True only if you suspect cache is stale

import os, sys, runpy
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

argv = [
    'tune_doctor_v3.py',
    '--n-trials', str(N_TRIALS_THIS_SESSION),
]
if TIMEOUT_SECONDS is not None:
    argv += ['--timeout', str(TIMEOUT_SECONDS)]
if REBUILD_FEATURES:
    argv += ['--rebuild-features']

# Save sys.argv so subsequent cells (or restarts) aren't polluted.
_saved_argv = sys.argv
try:
    sys.argv = argv
    print(f'running: {\" \".join(argv)}')
    print(f'(ctrl-C to stop early; SQLite study persists on Drive)\\n')
    runpy.run_path(
        f'{PROJECT_PATH}/scripts/tune_doctor_v3.py',
        run_name='__main__',
    )
except KeyboardInterrupt:
    print('\\n[stopped by user. Current trial marked FAIL; study persists.]')
finally:
    sys.argv = _saved_argv""")


md("section3-md", """## Study state

Cell 7 reads the SQLite file directly (no extra trials run) and prints the
current state: how many trials are complete, best macro-F1, top 5 trials,
and the per-param distribution Optuna's TPE sampler is converging on.""")


code("cell-state", """# Cell 7: Inspect study state (read-only)
import optuna, json
from proiect_licenta.paths import DOCTOR_V3_DIR

storage = f'sqlite:///{DOCTOR_V3_DIR / "optuna_study.db"}'
try:
    study = optuna.load_study(study_name='diag_v3_macro_f1', storage=storage)
except KeyError:
    print(f'No study at {storage}. Run Cell 6 first to create one.')
    raise SystemExit

trials = study.trials
complete = [t for t in trials if t.state.name == 'COMPLETE']
fail = [t for t in trials if t.state.name == 'FAIL']
running = [t for t in trials if t.state.name == 'RUNNING']

print(f'Study:        {study.study_name}')
print(f'Storage:      {storage}')
print(f'Total trials: {len(trials)} (complete: {len(complete)}, fail: {len(fail)}, running: {len(running)})')

if complete:
    print(f'\\nBest macro_f1: {study.best_value:.4f} (trial #{study.best_trial.number})')
    print(f'Top-1 acc at best: {study.best_trial.user_attrs.get("top1_accuracy", -1):.4f}')
    print(f'Best params:')
    for k, v in study.best_trial.params.items():
        print(f'  {k:25s} = {v}')

    print(f'\\nTop 5 trials by macro_f1:')
    top5 = sorted(complete, key=lambda t: -t.value)[:5]
    print(f'  {"#":>4s}  {"macro_f1":>9s}  {"top1_acc":>9s}  {"best_iter":>9s}  {"cw_exp":>6s}')
    for t in top5:
        print(f'  {t.number:>4d}  {t.value:>9.4f}  '
              f'{t.user_attrs.get("top1_accuracy", 0):>9.4f}  '
              f'{t.user_attrs.get("best_iteration", 0):>9d}  '
              f'{t.user_attrs.get("class_weight_exponent", 0):>6.2f}')

# Check tuned_params.json was written
import os
tp = DOCTOR_V3_DIR / 'tuned_params.json'
if tp.exists():
    tp_data = json.loads(tp.read_text())
    print(f'\\ntuned_params.json present (best macro_f1 recorded: '
          f'{tp_data.get("best_macro_f1", "?"):.4f})')
else:
    print(f'\\ntuned_params.json not present yet - run Cell 6 to write it')""")


md("section4-md", """## When you're done tuning

When the best macro-F1 has plateaued (typically after 30-50 trials), the
file `artifacts/doctor/v3/tuned_params.json` already contains the
configuration ready for `train_nurse_v3` to consume. Final step:

1. Open `notebooks/train_and_benchmark_v3.ipynb` (the main training
   notebook) in a fresh Colab session.
2. Re-run Cells 1-7 (mount, clone, symlink, install, GPU smoke, file
   check, PMH smoke).
3. Run Cell 8 (training). It will automatically pick up
   `tuned_params.json` from Drive and use the tuned config for the
   diagnosis model. The log will print "A1: tuned_params loaded" on
   diagnosis training.
4. Run Cells 9-12 (verify, benchmark, compare, audit) - same as a
   normal training run.

The training notebook is unchanged - the only difference is the presence
of `tuned_params.json` in `artifacts/doctor/v3/`.""")


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {
            "name": "tune_optuna_v3.ipynb", "provenance": [],
            "accelerator": "GPU",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}


def main():
    out = Path(__file__).resolve().parents[1] / "notebooks" / "tune_optuna_v3.ipynb"
    out.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
