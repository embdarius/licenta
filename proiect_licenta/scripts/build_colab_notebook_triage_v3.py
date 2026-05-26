"""Regenerate notebooks/train_triage_v3.ipynb.

Parallel to scripts/build_colab_notebook.py (which builds the doctor v3-nurse
notebook). Edit cell sources here, then run:

    python scripts/build_colab_notebook_triage_v3.py

The script-based approach avoids fragile JSON quote/backslash escaping.

Trains `artifacts/triage/v3/` from the current HEAD of the repo on Colab GPU:
    Triage v2 features (vital signs, vital flags, interactions, masked walk-ins)
  + 19 PMH features (13 binary group flags + 6 prior-encounter numerics)
on Colab GPU (XGB_DEVICE=cuda). Then runs benchmark_triage_v3.py head-to-head
against the v2 baseline on the same held-out rows.
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


md("intro-md", """# Triage v3 Training on Colab GPU

Trains `artifacts/triage/v3/` from the current HEAD of the repo on Colab GPU. As of this notebook the v3 pipeline produces:

- The triage v2 baseline - acuity (ESI 1-5) and disposition (admit/discharge) on TF-IDF + structured triage features + 6 vital signs + 8 abnormality flags + vital-transport / vital-age interactions. Vitals are masked to NaN for non-ambulance / non-helicopter patients during training to match inference behavior.
- **PMH features (Doctor v3 nurse Change 1 recipe, applied to the full triage dataset)** - 13 binary `pmh_<group>` flags parsed from the "Past Medical History" section of prior MIMIC discharge summaries (OR'd with ICD-derived flags from prior admissions) + 6 prior-encounter numerics: `n_prior_admissions`, `n_prior_ed_visits`, `days_since_last_admission`, `days_since_last_ed`, `same_complaint_as_prior`, `no_history`.

Leakage is zero by construction: both PMH sources are filtered to `prior_*time < current_intime`, so a stay never sees its own discharge note or admission.

All XGBoost training runs on **GPU (CUDA)** via XGBoost 2.x's `device="cuda"` API. The env vars `XGB_DEVICE=cuda` + `XGB_TREE_METHOD=hist` are set in Cell 7 before training kicks off - read by `train_triage_v3.py:XGB_DEVICE`.

---

## Repo layout assumption

`github.com/<you>/licenta` is the git repo. The actual project root (with `src/`, `pyproject.toml`, `notebooks/`, `.claude/`, etc.) lives in a nested `proiect_licenta/` subfolder:

```
licenta/                                    <- git repo root (cloned to /content/licenta)
+-- proiect_licenta/                        <- project root (PROJECT_PATH below)
    +-- src/proiect_licenta/...
    +-- pyproject.toml
    +-- notebooks/train_triage_v3.ipynb     <- this file
    +-- benchmarks/
    +-- data/         <- symlink to Drive (Cell 3)
    +-- artifacts/    <- symlink to Drive (Cell 3)
```

If your fork is flatter (project files at the git root), edit `PROJECT_PATH` in Cell 2.

---

## Prereq: Colab runtime must have GPU

Runtime menu -> Change runtime type -> **T4 GPU** (free tier) or L4/A100 (Colab Pro). Cell 5 (GPU smoke test) refuses to continue if `nvidia-smi` doesn't see a device.

---

## One-Time Drive Setup (do this before running any cells)

MIMIC-IV data and the existing triage v2 artifacts must be in Google Drive once. The notebook mounts Drive and symlinks those folders into `PROJECT_PATH` so `paths.py` resolves everything transparently.

### Required Drive structure

```
MyDrive/proiect_licenta/
+-- data/
|   +-- mimic-iv-ed/
|   |   +-- triage.csv
|   |   +-- edstays.csv
|   |   +-- files_created/
|   |       +-- categorized_diagnosis.csv
|   +-- mimic-iv/
|   |   +-- hosp/
|   |       +-- patients.csv
|   |       +-- diagnoses_icd.csv      (~181 MB)
|   |       +-- admissions.csv         (~94 MB)
|   +-- mimic-iv-notes/
|       +-- mimic-iv-notes/
|           +-- note/
|               +-- discharge.csv/
|                   +-- discharge.csv     (~3.3 GB)
+-- artifacts/
    +-- triage/v2/         <- REQUIRED (head-to-head baseline in benchmark)
    +-- triage/v3/         <- will be OVERWRITTEN by this notebook
```

Use the [Google Drive desktop app](https://www.google.com/drive/download/) for the 3.3 GB discharge.csv - the browser uploader times out.

> Note: triage v3 does NOT need `vitalsign.csv`, `medrecon.csv`, `services.csv`, or `diagnosis.csv` (those are doctor-only). It DOES need `categorized_diagnosis.csv` because the PMH aggregator uses the same ICD->group map the doctor pipeline uses.

---

## Estimated runtimes (Colab T4 GPU)

| Step | Time |
|------|------|
| Setup + install | ~3 min |
| GPU smoke test + file checks | < 1 min |
| `train_triage_v3` total | **~25-40 min** |
| &nbsp;&nbsp;discharge.csv PMH parse (CPU) | ~15-25 min |
| &nbsp;&nbsp;XGBoost acuity (GPU) | ~5-8 min |
| &nbsp;&nbsp;XGBoost disposition (GPU) | ~3-5 min |
| `benchmark_triage_v3` (head-to-head v2 vs v3) | ~2-3 min |

> The CPU-bound PMH parse dominates. XGBoost on GPU is roughly 4-6x faster than CPU on this workload.""")


md("section1-md", """---
## Section 1 - Environment Setup
*Run these cells at the start of every Colab session.*""")


code("cell-mount", """# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')""")


code("cell-clone", """# Cell 2: Clone the repository
# Edit GITHUB_USERNAME (and BRANCH if you work on a non-main branch).
GITHUB_USERNAME = '<YOUR_USERNAME>'  # <- EDIT THIS
REPO_URL        = f'https://github.com/{GITHUB_USERNAME}/licenta.git'
CLONE_PATH      = '/content/licenta'
PROJECT_PATH    = f'{CLONE_PATH}/proiect_licenta'
BRANCH          = 'main'

import os, subprocess

if os.path.exists(CLONE_PATH):
    print(f'Directory {CLONE_PATH} already exists - pulling latest changes.')
    subprocess.run(['git', '-C', CLONE_PATH, 'fetch'], check=False)
    subprocess.run(['git', '-C', CLONE_PATH, 'checkout', BRANCH], check=False)
    subprocess.run(['git', '-C', CLONE_PATH, 'pull'], check=False)
else:
    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, CLONE_PATH], check=True)

assert os.path.exists(PROJECT_PATH), (
    f'Expected {PROJECT_PATH} to exist after clone. '
    f'This notebook assumes the project files live in a nested '
    f'proiect_licenta/ subfolder under the licenta/ git repo root.'
)

subprocess.run(['git', '-C', CLONE_PATH, 'log', '-1', '--oneline'])
print(f'\\nProject root: {PROJECT_PATH}')""")


code("cell-symlink", """# Cell 3: Symlink data/ and artifacts/ from Drive into PROJECT_PATH
# paths.py resolves PROJECT_ROOT via parents[2] of src/proiect_licenta/paths.py,
# so the symlinks must live inside proiect_licenta/, not at the licenta/ git root.
DRIVE_PROJECT = '/content/drive/MyDrive/proiect_licenta'  # <- EDIT IF NEEDED

for folder in ('data', 'artifacts'):
    src  = f'{DRIVE_PROJECT}/{folder}'
    dest = f'{PROJECT_PATH}/{folder}'
    if os.path.islink(dest):
        print(f'Symlink already exists: {dest} -> {os.readlink(dest)}')
    elif os.path.exists(dest):
        print(f'WARNING: {dest} is a real directory, not a symlink. Skipping.')
    else:
        if not os.path.exists(src):
            print(f'WARNING: source {src} does not exist on Drive. Symlink not created.')
            continue
        os.symlink(src, dest)
        print(f'Created symlink: {dest} -> {src}')

print('Symlinks ready.')""")


code("cell-install", """# Cell 4: Install the proiect_licenta package (editable) + runtime dependencies
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], check=True)

pip('-e', PROJECT_PATH)                      # installs the local package
pip('xgboost>=2.0.0')                        # GPU-capable XGBoost
pip('tqdm>=4.66.0', 'ipywidgets>=8.0.0')     # live progress bars during PMH parse
pip('scikit-learn>=1.4.0')

print('All dependencies installed.')""")


md("section-gpu-md", """---
## Section 2 - Verify GPU + Environment

Cell 5 is the **gate** for the rest of the notebook:
1. Confirms `nvidia-smi` sees a CUDA device.
2. Confirms XGBoost >= 2.0 can build a small model on the GPU end-to-end (catches mismatched CUDA / driver setups before we burn 30 min on a CPU fallback).
3. Confirms all required MIMIC-IV files and the triage v2 baseline artifacts are reachable through the symlinks set up in Cell 3.

**If any check fails, stop and fix it before proceeding** - the long training run will either fall back to CPU silently or crash mid-way.""")


code("cell-gpu-smoke", """# Cell 5: GPU + XGBoost CUDA smoke test
import subprocess, numpy as np
r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
print(r.stdout.strip() or '(nvidia-smi -L produced no output)')
assert 'GPU' in r.stdout, (
    'nvidia-smi did not report a GPU. Set Runtime > Change runtime type > T4 GPU '
    '(or L4/A100 on Colab Pro) and re-run this cell.'
)

import xgboost as xgb
print(f'\\nXGBoost version: {xgb.__version__}')
assert tuple(int(x) for x in xgb.__version__.split('.')[:2]) >= (2, 0), (
    f'XGBoost {xgb.__version__} is too old; expected >= 2.0 for the '
    'device="cuda" API. Cell 4 should have upgraded it - re-run it.'
)

X = np.random.randn(2_000, 64).astype('float32')
y = np.random.randint(0, 5, 2_000)
m = xgb.XGBClassifier(
    n_estimators=20, tree_method='hist', device='cuda',
    eval_metric='mlogloss', verbosity=0, n_jobs=-1,
)
m.fit(X, y)
print('XGBoost GPU smoke test PASSED')""")


code("cell-verify", """# Cell 6: Check all required files and artifacts exist
from proiect_licenta.paths import (
    TRIAGE_CSV, EDSTAYS_CSV, PATIENTS_CSV, DIAGNOSIS_CSV,
    DIAGNOSES_ICD_CSV, ADMISSIONS_CSV, DISCHARGE_NOTES_CSV,
    TRIAGE_V2_DIR, TRIAGE_V3_DIR,
)

checks = {
    'triage.csv':                                  TRIAGE_CSV,
    'edstays.csv':                                 EDSTAYS_CSV,
    'patients.csv':                                PATIENTS_CSV,
    'categorized_diagnosis.csv':                   DIAGNOSIS_CSV,
    'diagnoses_icd.csv (~181 MB)':                 DIAGNOSES_ICD_CSV,
    'admissions.csv (~94 MB)':                     ADMISSIONS_CSV,
    'discharge.csv (~3.3 GB)':                     DISCHARGE_NOTES_CSV,
    'artifacts/triage/v2/ (required for benchmark)': TRIAGE_V2_DIR,
}

required = set(checks.keys())  # all triage v3 checks are required

missing_required = []
for name, path in checks.items():
    ok = path.exists()
    status = 'OK  ' if ok else ('MISS' if name in required else 'opt ')
    print(f'{status}  {name:60s}  {path}')
    if not ok and name in required:
        missing_required.append(name)

print()
if missing_required:
    raise RuntimeError(
        'Missing required files/artifacts: ' + ', '.join(missing_required) +
        '. Upload them to Drive (see Drive Setup at top), re-run Cell 3 + this cell.'
    )

print(f'\\nTRIAGE_V3_DIR (will be created/overwritten): {TRIAGE_V3_DIR}')
print('All required checks passed. Ready to train.')""")


md("section3-md", """---
## Section 3 - Train Triage v3 (GPU, v2 features + PMH)

Trains the v3 pipeline on GPU. Heavy steps inside this single cell:

1. Load `triage.csv` + `edstays.csv` + `patients.csv` (~1 min). Vitals get masked to NaN for non-ambulance / non-helicopter patients to match inference behavior.
2. **PMH aggregation:** stream `discharge.csv` (3.3 GB) in chunks, parse the PMH sections against `pmh_vocab.py`, OR with ICD-derived flags from `diagnoses_icd.csv`. ~15-25 min, CPU-bound.
3. Build the full feature matrix: 23 v1 structured + 28 v2 vital + 19 PMH + 2000 TF-IDF = **~2070 features** (the final number depends on TF-IDF vocab size on your data).
4. Split 80/20 (random_state=42, stratified on `acuity`).
5. **Train acuity model on GPU** (~5-8 min). 3000 trees, max_depth=10, lr=0.02, early stopping=100.
6. **Train disposition model on GPU** (~3-5 min). Same hyperparameters, plus the cascading `predicted_acuity` feature.
7. Save artifacts to Drive via the symlink (`artifacts/triage/v3/`).

**Outputs:**
- `acuity_model.joblib` - 5-class XGBoost.
- `disposition_model.joblib` - binary XGBoost with cascading acuity.
- `tfidf_vectorizer.joblib`, `severity_map.joblib`, `vital_medians.joblib`.
- `model_metadata.json` with the new `pmh_feature_cols` block.

*Expected runtime: 25-40 min on T4 GPU.*""")


code("cell-train-triage-v3", """# Cell 7: Train Triage v3 on GPU
#
# The XGB_DEVICE / XGB_TREE_METHOD env vars are picked up by
# train_triage_v3.py:XGB_DEVICE at import time, so setting them here BEFORE
# runpy.run_path is critical. If you re-run this cell after editing the
# env vars, restart the Colab runtime to clear the module cache.
import os
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

import runpy, time

t0 = time.time()
runpy.run_path(
    f'{PROJECT_PATH}/src/proiect_licenta/training/train_triage_v3.py',
    run_name='__main__',
)
print(f'\\nTotal training time: {(time.time() - t0) / 60:.1f} min')""")


code("cell-verify-v3-saved", """# Cell 8: Confirm v3 artifacts saved to Drive
import json

required_artifacts = [
    'acuity_model.joblib',
    'disposition_model.joblib',
    'tfidf_vectorizer.joblib',
    'severity_map.joblib',
    'vital_medians.joblib',
    'model_metadata.json',
]
missing = [f for f in required_artifacts if not (TRIAGE_V3_DIR / f).exists()]
if missing:
    raise RuntimeError(f'v3 training did not produce: {missing}')

print('v3 artifacts saved to Drive:')
for f in required_artifacts:
    p = TRIAGE_V3_DIR / f
    mb = p.stat().st_size / 1e6
    print(f'  {f:35s}  {mb:>8.1f} MB')

meta = json.loads((TRIAGE_V3_DIR / 'model_metadata.json').read_text())
print('\\nKey metadata fields:')
for key in ('version', 'trained_at', 'n_total_features', 'n_tfidf_features',
            'n_severity_words', 'acuity_accuracy_exact',
            'acuity_accuracy_within_1', 'disposition_accuracy'):
    print(f'  {key:30s} = {meta.get(key)}')

pmh_cols = meta.get('pmh_feature_cols')
assert pmh_cols and len(pmh_cols) == 19, (
    f'Expected 19 PMH feature columns in metadata; got {len(pmh_cols) if pmh_cols else 0}. '
    'Pull the latest commits and re-train.'
)
print(f'\\nPMH feature columns recorded: {len(pmh_cols)}')
print(f'  First three: {pmh_cols[:3]}')
print(f'  Numerics:    {pmh_cols[-6:]}')""")


md("section4-md", """---
## Section 4 - Benchmark v3 vs v2 (head-to-head on identical test rows)

Runs `benchmark_triage_v3.py`, which:

1. Re-runs `load_and_clean_data` from `train_triage_v3.py` (so the test set includes PMH columns).
2. Reproduces the same 80/20 stratified split via `random_state=42`.
3. Predicts with v3 on its full feature matrix.
4. Rebuilds the v2 feature matrix on the **same test rows** and predicts with the v2 artifacts loaded from Drive.
5. Reports the Δ on acuity (exact / within-1 / κ / under-triage) and disposition (accuracy / ROC AUC) — plus per-class recall, confusion matrices, and a PMH-feature-importance audit.

**What to look for:**
- Acuity exact accuracy Δ: predicted +1-2pp, with most lift on **ESI 1 / ESI 2 recall** (where prior history matters most clinically).
- Disposition accuracy Δ: predicted +1.5-2.5pp. Prior admission rate is plausibly the single strongest signal we've added.
- PMH feature audit: ≥1 PMH feature should have non-zero importance. None will likely crack the top 50 (matches the Doctor Change 1 pattern — PMH contributes through many small interactions).

*Expected runtime: 2-3 min.*""")


code("cell-bench-triage-v3", """# Cell 9: Benchmark - v2 vs v3 head-to-head
runpy.run_path(
    f'{PROJECT_PATH}/benchmarks/benchmark_triage_v3.py',
    run_name='__main__',
)""")


md("section5-md", """---
## Section 5 - Audit (PMH feature wiring)

Fail-fast verification gate. Failing this should block trusting the new artifacts.

- The metadata must record exactly 19 PMH feature columns.
- The acuity model's `feature_importances_` must show ≥1 PMH-related column with non-zero gain. Zero everywhere = silent merge failure (e.g. PMH aggregation ran but the columns weren't merged into the training feature matrix).""")


code("cell-audit", """# Cell 10: Audit PMH wiring
import joblib, pandas as pd, json

meta = json.loads((TRIAGE_V3_DIR / 'model_metadata.json').read_text())
pmh_cols_meta = meta.get('pmh_feature_cols', [])
print('PMH metadata audit')
print(f'  pmh_feature_cols recorded: {len(pmh_cols_meta)} (expected 19)')
assert len(pmh_cols_meta) == 19

acuity = joblib.load(TRIAGE_V3_DIR / 'acuity_model.joblib')
feature_names = list(acuity.feature_names_in_)
print(f'\\nAcuity model: {len(feature_names)} features')

pmh_prefixes = ('pmh_', 'n_prior_', 'days_since_', 'no_history',
                'same_complaint_as_prior')
pmh_cols_model = [f for f in feature_names if f.startswith(pmh_prefixes)]
importances = pd.Series(acuity.feature_importances_, index=feature_names)
non_zero_pmh = [f for f in pmh_cols_model if importances[f] > 0]
print(f'  PMH-related columns in model: {len(pmh_cols_model)}')
print(f'  With non-zero gain:           {len(non_zero_pmh)}')
assert len(non_zero_pmh) > 0, 'ZERO PMH features have non-zero gain - silent merge failure.'

# Top PMH columns by gain
pmh_imp = importances[pmh_cols_model].sort_values(ascending=False)
print(f'\\nTop 5 PMH features by gain (acuity model):')
for name, gain in pmh_imp.head(5).items():
    overall_rank = list(importances.sort_values(ascending=False).index).index(name) + 1
    print(f'  rank {overall_rank:4d}: {name:55s} gain={gain:.5f}')

# Disposition model
disp = joblib.load(TRIAGE_V3_DIR / 'disposition_model.joblib')
disp_feature_names = list(disp.feature_names_in_)
disp_pmh_cols = [f for f in disp_feature_names if f.startswith(pmh_prefixes)]
disp_importances = pd.Series(disp.feature_importances_, index=disp_feature_names)
disp_non_zero = [f for f in disp_pmh_cols if disp_importances[f] > 0]
print(f'\\nDisposition model: {len(disp_feature_names)} features')
print(f'  PMH columns: {len(disp_pmh_cols)}; non-zero gain: {len(disp_non_zero)}')
assert len(disp_non_zero) > 0, 'Disposition model also has zero PMH gain.'

disp_pmh_imp = disp_importances[disp_pmh_cols].sort_values(ascending=False)
print(f'\\nTop 5 PMH features by gain (disposition model):')
for name, gain in disp_pmh_imp.head(5).items():
    overall_rank = list(disp_importances.sort_values(ascending=False).index).index(name) + 1
    print(f'  rank {overall_rank:4d}: {name:55s} gain={gain:.5f}')

print(f'\\nAll audits PASSED.')""")


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {
            "name": "train_triage_v3.ipynb", "provenance": [],
            "accelerator": "GPU",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}


def main():
    out = Path(__file__).resolve().parents[1] / "notebooks" / "train_triage_v3.ipynb"
    out.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
