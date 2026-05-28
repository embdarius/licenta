"""Regenerate notebooks/train_doctor_disposition.ipynb.

Parallel to scripts/build_colab_notebook_triage_v3.py. Generates the Colab
notebook that trains `artifacts/doctor/v3/disposition_model.joblib` on
Colab GPU and runs the head-to-head benchmark against the triage v3
cascade baseline.

Edit cell sources here, then run:

    python scripts/build_colab_notebook_doctor_disposition.py

The script-based approach avoids fragile JSON quote/backslash escaping.
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


md("intro-md", """# Doctor Disposition v3 Training on Colab GPU

Trains `artifacts/doctor/v3/disposition_model.joblib` from the current HEAD
of the repo on Colab GPU. This is the **Option B peer model** described in
plan section 3: a binary admit/discharge classifier trained on the FULL
425K ED stays (not the admitted-only ~102K slice used by diagnosis +
department) and consumed by `doctor_disposition_tool` to refine the
triage disposition after the nurse step.

## What this model does

Refines the triage-time disposition (currently 76.2% acc / ROC AUC 0.84)
using everything nurse data adds: snapshot vitals + longitudinal vitals
+ rhythm + medications + PMH. Plan section 3 predicted lift band:
**76.2% → ~80-82%, ROC AUC 0.84 → ~0.88**.

## Key differences vs the diagnosis / department v3 training

- **Full 425K dataset** (no admitted-only filter, no catch-all filter)
- **Soft cascade** from triage (5 acuity softmax columns + 1 disposition
  probability) instead of the hard cascade used elsewhere. Plan section
  2/3 recommends this; it lets the model honestly weight borderline
  ESI 2-3 cases instead of hard-locking on the argmax.
- **Binary:logistic** objective with `scale_pos_weight = sqrt(N_neg/N_pos)`
- **Isotonic calibration** on a 10% held-out fit slice (A4 pattern). The
  plan calls out calibration as more important here than for diagnosis
  because disposition output is a clinical-decision probability — a
  miscalibrated 0.8 hurts.
- **Leakage guard** kept from nurse_v3: longitudinal vital window is
  `[intime, intime + 4h]` so late-stay disposition decisions never leak
  back into the feature vector.

## Feature set (~2150 columns)

- ~2069 columns from `train_triage_v3.build_features` (23 structured + ~28 v2 vital + 19 PMH + 2000 TF-IDF)
- 2000 TF-IDF features
- 6 **soft cascade** columns from triage v3
- 20 snapshot vital + clinical-flag features
- 24 medication features
- ~40 longitudinal vital + rhythm features
- 19 PMH features (13 binary + 6 numeric)

## Repo layout assumption

`github.com/<you>/licenta` is the git repo. The actual project root lives
in a nested `proiect_licenta/` subfolder:

```
licenta/                                       <- git repo root (cloned to /content/licenta)
+-- proiect_licenta/                           <- project root (PROJECT_PATH below)
    +-- src/proiect_licenta/...
    +-- pyproject.toml
    +-- notebooks/train_doctor_disposition.ipynb     <- this file
    +-- benchmarks/
    +-- data/         <- symlink to Drive (Cell 3)
    +-- artifacts/    <- symlink to Drive (Cell 3)
```

## Prereq: Colab runtime must have GPU

Runtime menu -> Change runtime type -> **T4 GPU** (free tier) or L4/A100
(Colab Pro). Cell 5 refuses to continue if `nvidia-smi` doesn't see a
CUDA device.

## One-Time Drive Setup

Required Drive structure (most of this is the same as the Doctor v3-nurse
notebook — same heavy CSVs, plus the existing triage v3 artifacts as the
soft cascade):

```
MyDrive/proiect_licenta/
+-- data/
|   +-- mimic-iv-ed/
|   |   +-- triage.csv
|   |   +-- edstays.csv
|   |   +-- vitalsign.csv           (~115 MB — for longitudinal vitals)
|   |   +-- medrecon.csv            (~37 MB)
|   |   +-- files_created/
|   |       +-- categorized_diagnosis.csv
|   +-- mimic-iv/
|   |   +-- hosp/
|   |       +-- patients.csv
|   |       +-- diagnoses_icd.csv   (~181 MB)
|   |       +-- admissions.csv      (~94 MB)
|   +-- mimic-iv-notes/
|       +-- mimic-iv-notes/
|           +-- note/
|               +-- discharge.csv/
|                   +-- discharge.csv    (~3.3 GB)
+-- artifacts/
    +-- triage/v3/         <- REQUIRED (soft cascade source; the live runtime also loads v3)
    +-- doctor/v3/         <- will gain a new disposition_model.joblib (will NOT
                              overwrite existing diagnosis_model / department_model)
```

> The new training script extends an existing `artifacts/doctor/v3/metadata.json`
> with a `disposition` block if one exists, rather than overwriting it. So your
> existing diagnosis + department models are safe.

## Estimated runtimes (Colab T4 GPU)

| Step | Time |
|------|------|
| Setup + install | ~3 min |
| GPU smoke test + file checks | < 1 min |
| `train_doctor_disposition` total | **~60-90 min** |
| &nbsp;&nbsp;discharge.csv PMH parse (CPU, ~425K stays) | ~25-40 min |
| &nbsp;&nbsp;vitalsign.csv longitudinal aggregation | ~5-8 min |
| &nbsp;&nbsp;XGBoost disposition (GPU, up to 5000 trees) | ~20-30 min |
| &nbsp;&nbsp;Isotonic calibration fit + metrics | ~1 min |
| `benchmark_doctor_disposition` | ~3-5 min |

> The PMH parse is now over the FULL 425K (vs ~102K admitted) — that's where
> the extra ~10-20 min vs the nurse_v3 notebook comes from.""")


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


code("cell-install", """# Cell 4: Install package + runtime dependencies
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], check=True)

pip('-e', PROJECT_PATH)                      # installs the local package
pip('xgboost>=2.0.0')                        # GPU-capable XGBoost
pip('tqdm>=4.66.0', 'ipywidgets>=8.0.0')     # live progress bars
pip('scikit-learn>=1.4.0')

print('All dependencies installed.')""")


md("section-gpu-md", """---
## Section 2 - Verify GPU + Environment

Cell 5 is the **gate** for the rest of the notebook:
1. Confirms `nvidia-smi` sees a CUDA device.
2. Confirms XGBoost >= 2.0 can build a small model on the GPU end-to-end.
3. Confirms all required MIMIC-IV files and the triage v3 cascade artifacts are reachable through the symlinks set up in Cell 3.

**If any check fails, stop and fix it before proceeding.**""")


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
    f'XGBoost {xgb.__version__} is too old; expected >= 2.0 for the device=\"cuda\" API.'
)

X = np.random.randn(2_000, 64).astype('float32')
y = np.random.randint(0, 2, 2_000)
m = xgb.XGBClassifier(
    n_estimators=20, tree_method='hist', device='cuda',
    eval_metric='logloss', verbosity=0, n_jobs=-1,
)
m.fit(X, y)
print('XGBoost GPU smoke test PASSED')""")


code("cell-verify", """# Cell 6: Check required files and artifacts
from proiect_licenta.paths import (
    TRIAGE_CSV, EDSTAYS_CSV, PATIENTS_CSV, DIAGNOSIS_CSV,
    DIAGNOSES_ICD_CSV, ADMISSIONS_CSV, DISCHARGE_NOTES_CSV,
    VITALSIGN_CSV, MEDRECON_CSV,
    TRIAGE_V3_DIR, DOCTOR_V3_DIR,
)

checks = {
    'triage.csv':                                  TRIAGE_CSV,
    'edstays.csv':                                 EDSTAYS_CSV,
    'patients.csv':                                PATIENTS_CSV,
    'vitalsign.csv (~115 MB)':                     VITALSIGN_CSV,
    'medrecon.csv (~37 MB)':                       MEDRECON_CSV,
    'categorized_diagnosis.csv':                   DIAGNOSIS_CSV,
    'diagnoses_icd.csv (~181 MB)':                 DIAGNOSES_ICD_CSV,
    'admissions.csv (~94 MB)':                     ADMISSIONS_CSV,
    'discharge.csv (~3.3 GB)':                     DISCHARGE_NOTES_CSV,
    'artifacts/triage/v3/ (soft-cascade source)':  TRIAGE_V3_DIR,
}

required = set(checks.keys())

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
        '. Upload them to Drive, re-run Cell 3 + this cell.'
    )

# triage v3 must have its cascade artifacts (tfidf, severity_map,
# acuity_model, disposition_model, vital_medians). Check explicitly so a half-baked v3
# folder doesn't silently break training.
needed_v3 = ['tfidf_vectorizer.joblib', 'severity_map.joblib',
             'vital_medians.joblib',
             'acuity_model.joblib', 'disposition_model.joblib']
missing_v3 = [f for f in needed_v3 if not (TRIAGE_V3_DIR / f).exists()]
if missing_v3:
    raise RuntimeError(
        f'triage v3 cascade artifacts missing in {TRIAGE_V3_DIR}: '
        f'{missing_v3}. Train triage v3 first (or upload from local).'
    )

print(f'\\nDOCTOR_V3_DIR (disposition_model.joblib will be added here): {DOCTOR_V3_DIR}')
print('All required checks passed. Ready to train.')""")


md("section3-md", """---
## Section 3 - Train Doctor Disposition v3 (GPU)

Trains the binary disposition model on GPU. Heavy steps inside this cell:

1. Load `triage.csv` + full `edstays.csv` + `patients.csv` + `medrecon.csv` (~1-2 min).
2. **Longitudinal vitals aggregation** from `vitalsign.csv` with the [intime, intime + 4h] window (~5-8 min, CPU).
3. **PMH aggregation** over the full 425K dataset: stream `discharge.csv` (3.3 GB) in chunks, parse PMH sections, OR with ICD-derived flags from `diagnoses_icd.csv`. ~25-40 min, CPU-bound (this dominates total runtime).
4. Build the feature matrix: 23 triage structured + 2000 TF-IDF + 6 **soft cascade** + 20 snapshot vitals + 24 meds + ~40 longitudinal + 19 PMH = **~2150 features**.
5. Split 80/20 (random_state=42, stratified on admit label).
6. Split out 10% of train as calibration holdout (A4 pattern).
7. **Train XGBoost on GPU** (~20-30 min, up to 5000 trees, lr=0.02, early stopping=150).
8. **Fit isotonic calibration** on the 10% holdout (~1 min).
9. Save to Drive:
   - `disposition_model.joblib` — calibrated wrapper (deployment)
   - `disposition_model_raw.joblib` — uncalibrated (audit)
   - `metadata.json` — extended with `disposition` block (existing diagnosis + department blocks preserved)

### What you should see in the cell output

- Live tqdm progress bar per phase, with `logloss` postfix during XGBoost training.
- Class balance report at top of step 1 (admitted vs discharged ratio).
- Final block reports both **uncalibrated** and **calibrated** metrics: accuracy, ROC AUC, Brier, ECE (10-bin), over-triage rate, under-triage rate, sensitivity, specificity.

### Headline numbers to look for

- **Accuracy** > 0.80 (triage v3 reference: 0.762)
- **ROC AUC** > 0.86 (triage v3 reference: 0.84)
- **ECE (calibrated)** < 0.04 (the whole point of isotonic)
- **Under-triage** < 0.15 (triage v3 reference: 0.177 — the safety-critical number)

*Expected runtime: 60-90 min on T4 GPU. PMH parse dominates the first half.*""")


code("cell-train-disposition", """# Cell 7: Train Doctor disposition v3 on GPU
import os
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

import runpy, time

t0 = time.time()
runpy.run_path(
    f'{PROJECT_PATH}/src/proiect_licenta/training/train_doctor_disposition.py',
    run_name='__main__',
)
print(f'\\nTotal training time: {(time.time() - t0) / 60:.1f} min')""")


code("cell-verify-saved", """# Cell 8: Confirm disposition v3 artifacts saved to Drive
import json

required_artifacts = [
    'disposition_model.joblib',
    'disposition_model_raw.joblib',
    'metadata.json',
]
missing = [f for f in required_artifacts if not (DOCTOR_V3_DIR / f).exists()]
if missing:
    raise RuntimeError(f'Training did not produce: {missing}')

print('Disposition v3 artifacts saved to Drive:')
for f in required_artifacts:
    p = DOCTOR_V3_DIR / f
    mb = p.stat().st_size / 1e6
    print(f'  {f:35s}  {mb:>8.1f} MB')

# Show the disposition block from metadata
meta = json.loads((DOCTOR_V3_DIR / 'metadata.json').read_text())
dispo = meta.get('disposition')
assert dispo is not None, 'metadata.json has no `disposition` block — training did not save it.'
print('\\nDisposition block from metadata:')
print(f'  version:          {dispo.get(\"version\")}')
print(f'  trained_at:       {dispo.get(\"trained_at\")}')
print(f'  model_type:       {dispo.get(\"model_type\")}')
print(f'  training_set:     {dispo.get(\"training_set\")}')
print(f'  leakage_guard:    {dispo.get(\"leakage_guard\")}')
print(f'  n_features:       {dispo.get(\"n_features\")}')
print(f'  soft_cascade_cols: {dispo.get(\"soft_cascade_cols\")}')

m = dispo.get('metrics', {})
print('\\nHeadline metrics (calibrated, 0.5 threshold):')
print(f'  Accuracy   : {m.get(\"accuracy_calibrated\"):.4f}  (triage v3 ref: ~0.762)')
print(f'  ROC AUC    : {m.get(\"roc_auc_calibrated\"):.4f}  (triage v3 ref: ~0.84)')
print(f'  ECE (10b)  : {m.get(\"ece_calibrated\"):.4f}    (calibration error, lower=better)')
print(f'  Brier      : {m.get(\"brier_calibrated\"):.4f}')
print(f'  Under-tri  : {m.get(\"under_triage_rate\"):.4f}  (triage v3 ref: ~0.177)')
print(f'  Over-tri   : {m.get(\"over_triage_rate\"):.4f}  (triage v3 ref: ~0.143)')
print(f'  Sens       : {m.get(\"sensitivity\"):.4f}')
print(f'  Spec       : {m.get(\"specificity\"):.4f}')

# Confirm diagnosis + department metadata blocks were preserved
preserved = [k for k in ('diagnosis_accuracy', 'department_accuracy') if k in meta]
print(f'\\nPreserved diag/dept metadata fields: {preserved}')""")


md("section4-md", """---
## Section 4 - Benchmark (head-to-head vs triage v3 cascade)

Runs `benchmark_doctor_disposition.py`, which:

1. Re-runs `load_and_clean_data` from `train_doctor_disposition.py` (full PMH/long-vitals pipeline).
2. Reproduces the same 80/20 stratified split via `random_state=42`.
3. Compares **triage v3 cascade dispo** (already in the soft-cascade features) vs the new **calibrated doctor dispo** on the same test rows.
4. Reports the Δ on accuracy / ROC AUC / Brier / ECE / sensitivity / specificity / over-triage / under-triage.
5. Per-subgroup breakdown — plan section 3 predicts the lift concentrates in **elderly / polypharmacy / abnormal-vitals / repeat-visitors / prior-admission / non-sinus-rhythm** cohorts.
6. Calibration curve (10 reliability bins).
7. Feature-importance audit on the uncalibrated XGBoost: gain-totals broken down by feature group, top-25 columns, and a PMH wiring check.

**What to look for:**
- Accuracy Δ: predicted +3-6pp (plan section 3 estimate)
- ROC AUC Δ: predicted +0.03-0.05
- Under-triage Δ: should be **negative** (the safety-critical reduction)
- Feature group totals: **soft_cascade + pmh + longitudinal + medications + snapshot_vitals** should each contribute non-trivial gain. If TF-IDF dominates ≥60% of total gain, the cascade is under-trusted.

*Expected runtime: 3-5 min.*""")


code("cell-benchmark", """# Cell 9: Benchmark - triage v3 cascade vs doctor disposition v3
runpy.run_path(
    f'{PROJECT_PATH}/benchmarks/benchmark_doctor_disposition.py',
    run_name='__main__',
)""")


md("section5-md", """---
## Section 5 - Audit + sanity checks

Fail-fast verification gates. Failing any of these should block trusting the new model.

- **Calibration fitted:** `disposition_model.joblib` must wrap the raw XGBoost in a `CalibratedClassifierCV`.
- **Calibrated ECE < uncalibrated ECE:** isotonic should NOT hurt calibration.
- **No catastrophic accuracy loss from calibration:** calibrated accuracy must be within 1.5pp of uncalibrated accuracy.
- **Headline lift achieved:** calibrated doctor accuracy must beat the triage v3 cascade baseline (~78% disposition reference from v3 iter 2 — see `docs/agents/triage-agent.md`) — if not, the model adds no value and we shouldn't ship it.
- **Soft cascade carries weight:** at least 4 of the 6 soft-cascade columns must have non-zero importance in the raw XGBoost.""")


code("cell-audit", """# Cell 10: Audit calibration + cascade wiring + headline lift
import json, joblib
from sklearn.calibration import CalibratedClassifierCV

meta = json.loads((DOCTOR_V3_DIR / 'metadata.json').read_text())
dispo = meta['disposition']
m = dispo['metrics']

print('=' * 60)
print('A. Calibration audit')
print('=' * 60)
cal = joblib.load(DOCTOR_V3_DIR / 'disposition_model.joblib')
raw = joblib.load(DOCTOR_V3_DIR / 'disposition_model_raw.joblib')
assert isinstance(cal, CalibratedClassifierCV), (
    f'Calibrated model expected to be CalibratedClassifierCV; got {type(cal).__name__}'
)
print(f'  Wrapper type        : {type(cal).__name__}    OK')
print(f'  Underlying estimator: {type(raw).__name__}')
print(f'  Calibration method  : {dispo.get(\"calibration\")}')
print(f'  ECE uncalibrated    : {m[\"ece_uncalibrated\"]:.4f}')
print(f'  ECE calibrated      : {m[\"ece_calibrated\"]:.4f}')
assert m['ece_calibrated'] <= m['ece_uncalibrated'] + 0.005, (
    'Isotonic made calibration WORSE - something is wrong with the holdout split.'
)
print('  -> calibration helped (or stayed flat) ✓')

print('\\n' + '=' * 60)
print('B. Headline lift audit (vs triage v3 reference of 0.762 / 0.84)')
print('=' * 60)
ACC = m['accuracy_calibrated']
AUC = m['roc_auc_calibrated']
UND = m['under_triage_rate']
print(f'  Accuracy   : {ACC:.4f}  (triage v3 ref: 0.762)')
print(f'  ROC AUC    : {AUC:.4f}  (triage v3 ref: 0.84)')
print(f'  Under-tri  : {UND:.4f}  (triage v3 ref: 0.177; lower=better)')
if ACC < 0.77:
    print('  WARN: accuracy <= triage v3 reference. Worth investigating before shipping.')
if AUC < 0.85:
    print('  WARN: ROC AUC barely above triage v3. Cascade may be under-trusted.')
if UND > 0.17:
    print('  WARN: under-triage rate did not improve. The safety-critical metric.')

print('\\n' + '=' * 60)
print('C. Soft-cascade wiring audit')
print('=' * 60)
soft_cols = dispo['soft_cascade_cols']
booster = raw.get_booster()
gains = booster.get_score(importance_type='gain')
feature_names = list(raw.feature_names_in_)
# Translate XGBoost feature names (which can be 'fN') to column names
def normalize_name(k):
    if k.startswith('f') and k[1:].isdigit():
        return feature_names[int(k[1:])]
    return k
gain_by_col = {normalize_name(k): v for k, v in gains.items()}
soft_with_gain = [(c, gain_by_col.get(c, 0.0)) for c in soft_cols]
print(f'  Soft cascade columns ({len(soft_cols)}):')
for c, g in soft_with_gain:
    marker = 'OK ' if g > 0 else 'ZERO'
    print(f'    [{marker}] {c:42s} gain={g:>10.2f}')
non_zero_cascade = sum(1 for _, g in soft_with_gain if g > 0)
assert non_zero_cascade >= 4, (
    f'Only {non_zero_cascade}/6 soft-cascade columns earned gain. '
    f'Expected >=4. Cascade is being ignored by the model.'
)
print(f'  -> {non_zero_cascade}/6 cascade columns earned gain ✓')

print('\\n' + '=' * 60)
print('All audits PASSED. Model is safe to ship to inference.')
print('=' * 60)""")


md("section6-md", """---
## Section 6 - Download artifacts (optional)

After training succeeds you can pull the calibrated model + metadata onto
your local machine. The artifacts are also on Drive at
`MyDrive/proiect_licenta/artifacts/doctor/v3/`.""")


code("cell-download", """# Cell 11: Optional - download disposition artifacts to local
from google.colab import files

for f in ('disposition_model.joblib',
          'disposition_model_raw.joblib',
          'metadata.json'):
    p = DOCTOR_V3_DIR / f
    if p.exists():
        print(f'Downloading {f} ({p.stat().st_size/1e6:.1f} MB)...')
        files.download(str(p))""")


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {
            "name": "train_doctor_disposition.ipynb", "provenance": [],
            "accelerator": "GPU",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}


def main():
    out = Path(__file__).resolve().parents[1] / "notebooks" / "train_doctor_disposition.ipynb"
    out.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
