"""Regenerate notebooks/train_and_benchmark_v3.ipynb.

Edit cell sources here, then run `python scripts/build_colab_notebook.py`.
The script-based approach avoids fragile JSON quote/backslash escaping.

Current notebook trains v3-nurse with:
    Change 1 (PMH features)               -- already in pmh_vocab.py
  + A2 (expanded PMH vocab)               -- pmh_vocab.py + retrain
  + A3 (diagnosis softmax cascade)        -- train_nurse_v3.py + doctor_tool_v3.py
  + A4 (isotonic dept calibration)        -- train_nurse_v3.py
on Colab GPU (XGB_DEVICE=cuda).
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


md("intro-md", """# Doctor v3 Nurse - Tier A Training (A2+A3+A4) on Colab GPU

Trains `artifacts/doctor/v3/` from the current HEAD of the repo, on Colab GPU. As of this notebook the v3-nurse pipeline produces:

- The original v3-nurse models - 13-class diagnosis + 11-class cascading department, on TF-IDF + structured triage + cascading triage predictions + snapshot vitals + medication flags + longitudinal vitals + cardiac rhythm.
- **Change 1 (PMH features)** - 19 feature columns from prior discharge-note PMH sections + ICD-derived fallback + repeat-visit numerics.
- **A2 (expanded PMH vocabulary)** - 597 keywords (up from 397) covering s/p surgical histories, abbreviation variants, CKD staging, and brand-name drugs as condition proxies.
- **A3 (diagnosis softmax cascade)** - the department model now consumes 13 `diag_proba_<category>` columns instead of a single argmax integer, so it can weight ambiguous diagnoses honestly.
- **A4 (isotonic department calibration)** - department model wrapped with `CalibratedClassifierCV(method='isotonic')` using a 10% held-out calibration set, for trustworthy displayed probabilities.

All XGBoost training runs on **GPU (CUDA)** via XGBoost 2.x's `device="cuda"` API. The env vars `XGB_DEVICE=cuda` + `XGB_TREE_METHOD=hist` are set in Cell 8 before training kicks off - read by `train_nurse_v3.py:XGB_DEVICE`.

---

## Repo layout assumption

`github.com/<you>/licenta` is the git repo. The actual project root (with `src/`, `pyproject.toml`, `notebooks/`, `.claude/`, etc.) lives in a nested `proiect_licenta/` subfolder:

```
licenta/                                    <- git repo root (cloned to /content/licenta)
+-- proiect_licenta/                        <- project root (PROJECT_PATH below)
    +-- src/proiect_licenta/...
    +-- pyproject.toml
    +-- notebooks/train_and_benchmark_v3.ipynb   <- this file
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

MIMIC-IV data and existing trained artifacts must be in Google Drive once. The notebook mounts Drive and symlinks those folders into `PROJECT_PATH` so `paths.py` resolves everything transparently.

### Required Drive structure

```
MyDrive/proiect_licenta/
+-- data/
|   +-- mimic-iv-ed/
|   |   +-- triage.csv
|   |   +-- edstays.csv
|   |   +-- vitalsign.csv          (~115 MB)
|   |   +-- medrecon.csv           (~360 MB)
|   |   +-- files_created/
|   |       +-- categorized_diagnosis.csv
|   +-- mimic-iv/
|   |   +-- hosp/
|   |       +-- patients.csv
|   |       +-- services.csv
|   |       +-- diagnoses_icd.csv      (~181 MB)
|   |       +-- admissions.csv         (~94 MB)
|   +-- mimic-iv-notes/
|       +-- mimic-iv-notes/
|           +-- note/
|               +-- discharge.csv/
|                   +-- discharge.csv     (~3.3 GB)
+-- artifacts/
    +-- triage/v1/         <- REQUIRED (cascading source for v3)
    +-- doctor/v3_base/    <- REQUIRED for the benchmark (NOT retrained here)
    +-- doctor/v3/         <- will be OVERWRITTEN by this notebook
```

Use the [Google Drive desktop app](https://www.google.com/drive/download/) for the 3.3 GB discharge.csv - the browser uploader times out.

---

## Estimated runtimes (Colab T4 GPU)

| Step | Time |
|------|------|
| Setup + install | ~3 min |
| GPU smoke test + file checks + PMH vocab smoke | < 1 min |
| `train_nurse_v3` total | **~35-55 min** |
| &nbsp;&nbsp;discharge.csv PMH parse (CPU, A2 vocab makes this slightly slower) | ~15-25 min |
| &nbsp;&nbsp;longitudinal vitals aggregate (CPU) | ~3-5 min |
| &nbsp;&nbsp;XGBoost diagnosis (GPU) | ~5-8 min |
| &nbsp;&nbsp;XGBoost department uncal (GPU, A3 cascade) | ~5-8 min |
| &nbsp;&nbsp;A4 isotonic calibration | < 1 min |
| `benchmark_nurse_v3` | ~2-5 min |
| `compare_all_versions` | < 1 min |
| Audit (A2 + A3 + A4) | < 1 min |

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
pip('thefuzz[speedup]>=0.22.0')              # fuzzy drug-name matching
pip('tqdm>=4.66.0', 'ipywidgets>=8.0.0')     # live progress bars in Cell 8
pip('scikit-learn>=1.6.0')                   # FrozenEstimator for A4 calibration

print('All dependencies installed.')""")


md("section-gpu-md", """---
## Section 2 - Verify GPU + Environment

Cell 5 is the **gate** for the rest of the notebook:
1. Confirms `nvidia-smi` sees a CUDA device.
2. Confirms XGBoost >= 2.0 can build a small model on the GPU end-to-end (catches mismatched CUDA / driver setups before we burn 30 min on a CPU fallback).
3. Confirms all required MIMIC-IV files and pre-trained artifacts are reachable through the symlinks set up in Cell 3.

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

import sklearn
print(f'sklearn version: {sklearn.__version__}')
# A4 wants FrozenEstimator (added in sklearn 1.6). Older versions silently
# fall back to the deprecated cv="prefit" path which errors on sklearn 1.6+.
try:
    from sklearn.frozen import FrozenEstimator
    print('  FrozenEstimator available (A4 will use cv=None path)')
except ImportError:
    print('  WARN: sklearn.frozen.FrozenEstimator not available. '
          'A4 will fall back to cv="prefit" which is deprecated in sklearn 1.6+.')

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
    SERVICES_CSV, VITALSIGN_CSV, MEDRECON_CSV,
    DIAGNOSES_ICD_CSV, ADMISSIONS_CSV, DISCHARGE_NOTES_CSV,
    TRIAGE_V1_DIR, DOCTOR_V1_DIR, DOCTOR_V2_DIR,
    DOCTOR_V3_BASE_DIR, DOCTOR_V3_DIR,
)

checks = {
    'triage.csv':                                  TRIAGE_CSV,
    'edstays.csv':                                 EDSTAYS_CSV,
    'patients.csv':                                PATIENTS_CSV,
    'categorized_diagnosis.csv':                   DIAGNOSIS_CSV,
    'services.csv':                                SERVICES_CSV,
    'vitalsign.csv (~115 MB)':                     VITALSIGN_CSV,
    'medrecon.csv (~360 MB)':                      MEDRECON_CSV,
    'diagnoses_icd.csv (~181 MB)':                 DIAGNOSES_ICD_CSV,
    'admissions.csv (~94 MB)':                     ADMISSIONS_CSV,
    'discharge.csv (~3.3 GB)':                     DISCHARGE_NOTES_CSV,
    'artifacts/triage/v1/ (required)':             TRIAGE_V1_DIR,
    'artifacts/doctor/v1/ (optional)':             DOCTOR_V1_DIR,
    'artifacts/doctor/v2/ (optional)':             DOCTOR_V2_DIR,
    'artifacts/doctor/v3_base/ (required for benchmark)': DOCTOR_V3_BASE_DIR,
}

required = {'triage.csv', 'edstays.csv', 'patients.csv', 'categorized_diagnosis.csv',
            'services.csv', 'vitalsign.csv (~115 MB)', 'medrecon.csv (~360 MB)',
            'diagnoses_icd.csv (~181 MB)', 'admissions.csv (~94 MB)',
            'discharge.csv (~3.3 GB)', 'artifacts/triage/v1/ (required)',
            'artifacts/doctor/v3_base/ (required for benchmark)'}

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
print('All required checks passed. Ready to train.')""")


md("section3-md", """---
## Section 3 - PMH Vocabulary Smoke Test (A2)

Sanity check on `pmh_vocab.py` before training. Verifies positive matching, negation handling, and the discharge-note section extractor. Also confirms the **A2 expansion** is present (~600 keywords, up from ~400) with new s/p / brand-drug entries. A silent failure here corrupts the PMH columns and makes the trained model worse than v3 base.""")


code("cell-pmh-smoke", """# Cell 7: PMH vocab smoke test (Change 1 + A2 expansion)
from proiect_licenta import pmh_vocab as p

print(f'PMH categories: {len(p.PMH_CATEGORIES)} (should be 13)')
print(f'PMH keywords:   {len(p.PMH_KEYWORD_MAP)} (Change 1 baseline ~397, A2 expanded ~597)')
assert len(p.PMH_KEYWORD_MAP) >= 500, (
    f'Expected A2 vocab expansion (>=500 keywords) but found '
    f'{len(p.PMH_KEYWORD_MAP)}. Pull the latest commits and re-run Cell 2.'
)

# Baseline tests (Change 1)
cases = [
    ('Past Medical History: CHF, T2DM\\nSocial History: lives alone\\n', {'CHF', 'T2DM'}, 'Social'),
    ('PMH:\\n1. CHF\\n2. Diabetes\\n3. CKD\\nSocial History: nonsmoker', {'CHF', 'Diabetes', 'CKD'}, 'Social'),
    ('Past Medical History:\\nNo history of CHF. Denies seizures.\\nSocial: ok', set(), None),
]
for note, must_contain, must_not in cases:
    section = p.extract_pmh_section(note)
    flags   = p.flags_from_text(section)
    print(f'  section: {section[:80]!r:80s} flags={sorted(flags)}')
    for kw in must_contain:
        assert kw in section, f'expected {kw!r} in extracted section'
    if must_not is not None:
        assert must_not not in section, f'unexpected {must_not!r} in section'

# A2-specific tests
a2_cases = [
    ('s/p cabg in 2018, ihd, on lisinopril and lipitor',
     {'Circulatory', 'Endocrine, Nutritional, Metabolic'}),
    ('paf on eliquis, chronic afib',                {'Circulatory'}),
    ('ckd stage 4, on hemodialysis 3x weekly',      {'Genitourinary'}),
    ('hiv positive, on art',                        {'Infectious and Parasitic'}),
    ('gerd on omeprazole',                          {'Digestive'}),
    ('depression on prozac, anxiety',               {'Mental Disorders'}),
    ('No history of cabg',                          set()),  # negation
]
for text, expected in a2_cases:
    got = p.flags_from_text(text)
    ok = got == expected
    print(f'  {"OK " if ok else "FAIL"}  {text:55s} got={sorted(got)}')
    assert ok, f'A2 expectation failed: expected {sorted(expected)}, got {sorted(got)}'

print('\\nPMH vocab smoke test PASSED (Change 1 + A2 expansion).')""")


md("section4-md", """---
## Section 4 - Train Doctor v3 Nurse (GPU, Change 1 + A2 + A3 + A4)

Trains the v3-nurse pipeline on GPU. Heavy steps inside this single cell:

1. Load & merge all ED + hospital tables (~1 min).
2. **A2 + Change 1:** stream `discharge.csv` (3.3 GB) in chunks, parse PMH sections against the expanded vocabulary, build the per-stay flag matrix (~15-25 min, CPU-bound).
3. Aggregate longitudinal vitals from `vitalsign.csv` (~3-5 min, CPU).
4. Build the ~2116-feature matrix and split 80/20 (random_state=42, stratified on diagnosis).
5. **Train diagnosis model on GPU** (~5-8 min).
6. **A3:** cascade diagnosis softmax (13 cols) into the department feature matrix.
7. **A4:** hold out 10% of train as a calibration set; fit the department model on the remaining 90%.
8. **Train uncalibrated department model on GPU** (~5-8 min).
9. **A4:** wrap with isotonic calibration on the held-out 10%.
10. Save artifacts to Drive via the symlink (`artifacts/doctor/v3/`).

**Outputs:**
- `diagnosis_model.joblib` - 13-class XGBoost, unchanged by A3/A4.
- `department_model.joblib` - **CalibratedClassifierCV** wrapping the cascading 11-class XGBoost.
- `metadata.json` with `diag_cascade_cols` (A3) and `department_calibration` (A4) blocks.

`doctor_tool_v3.py` auto-detects the A3 cascade from metadata and rebuilds the same column layout at inference. The A4 wrapper is interface-compatible with `XGBClassifier` so no further inference change is needed.

*Expected runtime: 35-55 min on T4 GPU.*""")


code("cell-train-nurse-v3", """# Cell 8: Train Doctor v3 nurse on GPU (Change 1 + A2 + A3 + A4)
#
# The XGB_DEVICE / XGB_TREE_METHOD env vars are picked up by
# train_nurse_v3.py:XGB_DEVICE at import time, so setting them here BEFORE
# runpy.run_path is critical. If you re-run this cell after editing the
# env vars, restart the Colab runtime to clear the module cache.
import os
os.environ['XGB_DEVICE'] = 'cuda'
os.environ['XGB_TREE_METHOD'] = 'hist'

import runpy, time

t0 = time.time()
runpy.run_path(
    f'{PROJECT_PATH}/src/proiect_licenta/training/train_nurse_v3.py',
    run_name='__main__',
)
print(f'\\nTotal training time: {(time.time() - t0) / 60:.1f} min')""")


code("cell-verify-v3-saved", """# Cell 9: Confirm v3 nurse artifacts saved to Drive (incl. A3 cascade + A4 calibration)
import json

required_artifacts = [
    'diagnosis_model.joblib',
    'department_model.joblib',         # now A4 CalibratedClassifierCV wrapper
    'metadata.json',
]
missing = [f for f in required_artifacts if not (DOCTOR_V3_DIR / f).exists()]
if missing:
    raise RuntimeError(f'v3 training did not produce: {missing}')

print('v3 artifacts saved to Drive:')
for f in required_artifacts:
    p = DOCTOR_V3_DIR / f
    mb = p.stat().st_size / 1e6
    print(f'  {f:35s}  {mb:>8.1f} MB')

meta = json.loads((DOCTOR_V3_DIR / 'metadata.json').read_text())
print('\\nKey metadata fields:')
for key in ('version', 'trained_at', 'diagnosis_accuracy', 'department_accuracy',
            'n_diagnosis_classes', 'n_department_classes', 'n_train', 'n_test',
            'catch_all_excluded'):
    print(f'  {key:30s} = {meta.get(key)}')

# A3 check
diag_cascade_cols = meta.get('diag_cascade_cols')
assert diag_cascade_cols and len(diag_cascade_cols) == meta['n_diagnosis_classes'], (
    'A3 expected diag_cascade_cols list of length n_diagnosis_classes in metadata; '
    f'got {len(diag_cascade_cols) if diag_cascade_cols else 0}. '
    'Pull the latest commits and re-train.'
)
print(f'\\nA3 diagnosis cascade columns: {len(diag_cascade_cols)} (e.g. {diag_cascade_cols[:3]})')

# A4 check
cal = meta.get('department_calibration')
assert cal and cal.get('method') == 'isotonic', (
    'A4 expected department_calibration block with method="isotonic" in metadata; '
    f'got {cal!r}. Pull the latest commits and re-train.'
)
print(f'A4 department calibration:')
print(f'  method            = {cal["method"]}')
print(f'  n_cal             = {cal["n_cal"]:,}')
print(f'  n_fit             = {cal["n_fit"]:,}')
print(f'  uncal -> cal top-1 = {cal["uncalibrated_accuracy"]:.4f} -> {cal["calibrated_accuracy"]:.4f}  '
      f'(delta {cal["calibrated_accuracy"] - cal["uncalibrated_accuracy"]:+.4f})')""")


md("section5-md", """---
## Section 5 - Benchmark v3 Base vs v3 Nurse

Compares **v3 base (reused from Drive, no Tier A)** vs **v3 nurse (just trained, with A2+A3+A4)** on the same held-out 20% test split (seed=42).

**What to look for:**
- v3 base -> v3 nurse top-1 delta. Pre-A2/A3/A4 baseline was **+4.07pp** diagnosis / **+7.96pp** department. With A2 we expect another **+0.3-0.5pp** diagnosis; with A3 another **+0.3-0.8pp** department; with A4 another **+0.1-0.3pp** department.
- Per-class recall: A2 should help Infectious / Endocrine / Blood / Genitourinary (where prior history matters most). A3 should help any department where the diagnosis decision was ambiguous (top-1 vs top-3 close).
- Feature importance: at least one `diag_proba_*` column (A3) and the PMH-related columns (A2 / Change 1) should appear in the dept model's top features.

*Expected runtime: 2-5 min*""")


code("cell-bench-nurse-v3", """# Cell 10: Benchmark - v3 base vs v3 nurse
runpy.run_path(
    f'{PROJECT_PATH}/benchmarks/benchmark_nurse_v3.py',
    run_name='__main__',
)""")


md("section6-md", """---
## Section 6 - Compare All Versions

Four-way accuracy table across v1 / v2 / v3 base / v3 nurse. Reads `metadata.json` from each artifact directory.

> v1 / v2 use 14-class diagnosis (catch-all included); v3 base / v3 nurse use 13 (catch-all excluded). Accuracy across the two label spaces is **not directly comparable** and the script flags this.""")


code("cell-compare-all", """# Cell 11: Compare all versions
runpy.run_path(
    f'{PROJECT_PATH}/benchmarks/compare_all_versions.py',
    run_name='__main__',
)""")


md("section7-md", """---
## Section 7 - Audit (A2 + A3 + A4)

Fail-fast verification gates. Failing any of these should block trusting the new artifacts.

**A2 (PMH vocab expansion):**
- PMH keyword count must be > 500 (Change 1 baseline was 397).
- At least one PMH-related column must have non-zero `gain` in the diagnosis model. Zero everywhere = silent merge failure.

**A3 (diagnosis softmax cascade):**
- 13 `diag_proba_*` columns must be present in the department model's feature_names.
- At least one of them must rank in the dept model's top 30 by gain.

**A4 (isotonic department calibration):**
- The saved `department_model.joblib` must be a `CalibratedClassifierCV` wrapper, not a bare `XGBClassifier`.
- Calibrated top-1 must be >= uncalibrated top-1 - 0.005 (allow 0.5pp downside; isotonic is monotone on probabilities, so the argmax should change very little).""")


code("cell-audit", """# Cell 12: Audit A2 + A3 + A4
import joblib, pandas as pd, json
from sklearn.calibration import CalibratedClassifierCV

meta = json.loads((DOCTOR_V3_DIR / 'metadata.json').read_text())

# ---- A2 audit ----
print('=' * 60)
print('A2: PMH vocab expansion audit')
print('=' * 60)
from proiect_licenta import pmh_vocab as p
print(f'  PMH keywords: {len(p.PMH_KEYWORD_MAP)} (Change 1 baseline 397, A2 target >500)')
assert len(p.PMH_KEYWORD_MAP) >= 500

diag_model = joblib.load(DOCTOR_V3_DIR / 'diagnosis_model.joblib')
feature_names = list(diag_model.feature_names_in_)
print(f'  Diagnosis model: {len(feature_names)} features')

pmh_prefixes = ('pmh_', 'n_prior_', 'days_since_', 'no_history',
                'same_complaint_as_prior')
pmh_cols = [f for f in feature_names if f.startswith(pmh_prefixes)]
importances = pd.Series(diag_model.feature_importances_, index=feature_names)
non_zero_pmh = [f for f in pmh_cols if importances[f] > 0]
print(f'  PMH-related columns: {len(pmh_cols)}; with non-zero gain: {len(non_zero_pmh)}')
assert len(non_zero_pmh) > 0, 'ZERO PMH features have non-zero gain - silent merge failure.'

# ---- A3 audit ----
print('\\n' + '=' * 60)
print('A3: diagnosis softmax cascade audit')
print('=' * 60)
diag_cascade_cols = meta.get('diag_cascade_cols')
assert diag_cascade_cols, 'A3 diag_cascade_cols missing from metadata.'
print(f'  Recorded cascade columns: {len(diag_cascade_cols)}')
print(f'  First three: {diag_cascade_cols[:3]}')

dept_model = joblib.load(DOCTOR_V3_DIR / 'department_model.joblib')
# After A4, dept_model is a CalibratedClassifierCV; the underlying estimator
# is the XGBClassifier we care about for feature_names + importances.
underlying = (
    dept_model.calibrated_classifiers_[0].estimator
    if isinstance(dept_model, CalibratedClassifierCV)
    else dept_model
)
dept_feature_names = list(underlying.feature_names_in_)
print(f'  Department model (underlying): {len(dept_feature_names)} features')

cascade_present = [c for c in diag_cascade_cols if c in dept_feature_names]
print(f'  Cascade columns present in dept feature_names: {len(cascade_present)} / {len(diag_cascade_cols)}')
assert len(cascade_present) == len(diag_cascade_cols), (
    'Some cascade columns are missing from the dept model. A3 wiring broken?'
)

dept_importances = pd.Series(
    underlying.feature_importances_, index=dept_feature_names,
).sort_values(ascending=False)
top30 = dept_importances.head(30)
cascade_in_top30 = [c for c in top30.index if c in diag_cascade_cols]
print(f'  Cascade columns in dept top 30 by gain: {len(cascade_in_top30)}')
for c in cascade_in_top30:
    rank = list(top30.index).index(c) + 1
    print(f'    rank {rank:2d}: {c:55s} gain={dept_importances[c]:.5f}')
if not cascade_in_top30:
    print('  WARNING: no cascade column reached the top 30. A3 may be contributing through '
          'many small interactions (similar to PMH); not necessarily a bug.')

# ---- A4 audit ----
print('\\n' + '=' * 60)
print('A4: isotonic department calibration audit')
print('=' * 60)
assert isinstance(dept_model, CalibratedClassifierCV), (
    f'Expected CalibratedClassifierCV wrapper but got {type(dept_model).__name__}. '
    'A4 wiring broken?'
)
print(f'  Saved type:        {type(dept_model).__name__}')
print(f'  Underlying type:   {type(underlying).__name__}')
print(f'  N inner estimators: {len(dept_model.calibrated_classifiers_)}')

cal_meta = meta.get('department_calibration', {})
delta = cal_meta.get('calibrated_accuracy', 0) - cal_meta.get('uncalibrated_accuracy', 0)
print(f'  Uncalibrated top-1: {cal_meta.get("uncalibrated_accuracy")}')
print(f'  Calibrated   top-1: {cal_meta.get("calibrated_accuracy")}')
print(f'  Delta:              {delta:+.4f}')
assert delta >= -0.005, (
    f'Calibrated top-1 dropped by {-delta:.4f} (>0.005). Isotonic should be near-monotone; '
    'a larger drop suggests the calibration set was too small or pathological.'
)
print(f'\\nAll audits PASSED.')""")


md("section-optional-md", """---
## (Optional) Section 8 - Retrain v3 Base

Skip unless you suspect `artifacts/doctor/v3_base/` is stale. v3 base does NOT use Tier A features - retraining it produces equivalent artifacts.

*Note: train_doctor_v3.py does not currently read XGB_DEVICE env vars; it would train on CPU even with cuda set. Deferred since v3 base is rarely retrained.*""")


code("cell-train-v3-base-optional", """# Cell 13 (OPTIONAL): Re-train Doctor v3 base
# Uncomment to run.
# runpy.run_path(
#     f'{PROJECT_PATH}/src/proiect_licenta/training/train_doctor_v3.py',
#     run_name='__main__',
# )""")


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {
            "name": "train_and_benchmark_v3.ipynb", "provenance": [],
            "accelerator": "GPU",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}


def main():
    out = Path(__file__).resolve().parents[1] / "notebooks" / "train_and_benchmark_v3.ipynb"
    out.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
