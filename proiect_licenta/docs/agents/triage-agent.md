# Triage Agent

The Triage Agent predicts ESI acuity (1-5) and admission vs. discharge from the structured output of the NLP Parser. It is the first ML stage in the cascading pipeline.

A **v3 tier** sits alongside v1/v2 (it does not replace them yet at runtime). v3 keeps the v2 vital-sign feature set and adds 19 PMH features parsed from prior MIMIC `discharge.csv` notes + `diagnoses_icd.csv`, using the same recipe Doctor v3 nurse uses (Change 1). See the [v3 section](#triage-v3--pmh-features) below for the v2→v3 deltas and the PMH details. v1 and v2 stay in the repo as baselines for direct thesis comparison.

---

## Role

- **Type:** CrewAI agent with a custom tool (`TriagePredictionTool`) wrapping two XGBoost models.
- **Purpose:** Predict ESI acuity level (1-5) and admission/discharge disposition.
- **Config:** `src/proiect_licenta/config/agents.yaml` (`triage_agent`) and `tasks.yaml` (`triage_assessment_task`).
- **Output contract:** Emits a structured data block downstream tasks can parse.

---

## Models

Stored in `artifacts/triage/{v1,v2,v3}/`. The runtime currently loads v2; v3 is benchmark-only as of 2026-05-26 (inference rewiring is the next staged change).

### Model 1: Acuity Prediction (ESI 1-5)
- **File:** `acuity_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02)
- **Training data:** 334,466 samples (80/20 split from 418K cleaned rows, full dataset)
- **Features (2051 total):**
  - `pain` score (0-10, or -1 if missing)
  - `pain_missing`, `pain_low`, `pain_mid`, `pain_high` (binary flags)
  - `n_complaints` (count of chief complaints)
  - `complaint_length` (character length of normalized text)
  - Severity priors: `min_severity_prior`, `mean_severity_prior`, `max_severity_prior`, `std_severity_prior`
  - `age`, `age_bin` (binned into 6 groups: 0-18, 18-35, 35-50, 50-65, 65-80, 80+)
  - `gender_male` (binary)
  - `arrival_ambulance`, `arrival_helicopter`, `arrival_walk_in` (binary)
  - v1 interaction features: `age_ambulance`, `pain_x_min_severity`, `age_severity`, `high_pain_ambulance`, `elderly`, `elderly_ambulance`
  - **[v2] Raw vitals** (imputed to training median when missing): `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`
  - **[v2] Vital missing flags** (one per vital): `temperature_missing`, `heartrate_missing`, etc.
  - **[v2] Abnormality flags**: `fever` (>100.4°F), `hypothermic` (<96.8°F), `tachycardic` (HR>100), `bradycardic` (HR<60), `tachypneic` (RR>20), `hypoxic` (O2<94%), `hypertensive` (SBP>140), `hypotensive` (SBP<90)
  - **[v2] `abnormal_vital_count`**: sum of all abnormality flags
  - **[v2] Vital-transport interactions**: `tachycardic_ambulance`, `hypoxic_ambulance`, `hypotensive_ambulance`, `fever_ambulance`
  - **[v2] Vital-age interactions**: `tachycardic_elderly`, `hypoxic_elderly`, `hypotensive_elderly`
  - 2000 TF-IDF features (word unigrams, bigrams, trigrams from chief complaint text)
- **Class weighting:** Soft balanced (sqrt of inverse frequency).
- **Note:** Model outputs classes 0-4 internally; add 1 to get ESI 1-5.

### Model 2: Disposition Prediction (Admit vs. Discharge)
- **File:** `disposition_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02)
- **Features:** Same 2051 features **plus** `predicted_acuity` (cascading).

### Supporting Artifacts
- `tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer (2000 features, trigrams).
- `severity_map.joblib` — Dict mapping complaint words to their mean acuity (819 words).
- `vital_medians.joblib` — Dict of per-vital training medians used for imputation at inference.
- `model_metadata.json` — Training date, accuracy metrics, feature count, vital thresholds.

---

## Realistic Training Design

A key design decision: **vitals are masked to missing for non-ambulance/helicopter patients during training**, matching inference behavior.

At inference, only ambulance/helicopter patients (~36%) have EMS-provided vitals at triage. Walk-in patients (~60%) have not yet had vitals measured. The training pipeline enforces this split: after loading `triage.csv`, vitals for walk-in/other/unknown patients are set to `NaN` before feature engineering. This ensures the model learns what "missing vitals" means for each transport type, rather than treating all-missing as a danger signal.

| Arrival transport | Patients | Vitals at triage |
|---|---|---|
| Ambulance | 36.0% | ✓ Real EMS vitals |
| Walk-in | 60.1% | ✗ Masked → imputed to median |
| Helicopter | 0.1% | ✓ Real EMS vitals |
| Other/Unknown | 3.8% | ✗ Masked → imputed to median |

Vital physiological clip ranges (values outside → treated as data entry errors, set to NaN):
`temperature` 90–110°F, `heartrate` 20–250 bpm, `resprate` 4–60, `o2sat` 50–100%, `sbp` 50–300 mmHg, `dbp` 20–200 mmHg.

---

## Benchmark Results

Test set: **83,617 samples** (20% held-out, same 80/20 stratified split as training).

### Overall

| Metric | Value |
|--------|-------|
| **Acuity exact accuracy** | **68.0%** |
| **Acuity within-1-level** | **98.0%** |
| Acuity within-2-level | 99.9% |
| Acuity MAE | 0.34 ESI levels |
| Acuity quadratic kappa | 0.64 |
| **Disposition accuracy** | **76.2%** |
| Disposition ROC AUC | 0.84 |
| Over-triage rate | 14.3% |
| Under-triage rate | 17.7% |

**Per-class acuity recall:** ESI 1: 58.8%, ESI 2: 65.4%, ESI 3: 71.8%, ESI 4: 60.7%, ESI 5: 21.8%

### By Arrival Transport

| Group | n | Exact accuracy | Within-1 | MAE |
|---|---|---|---|---|
| Ambulance/Helicopter (real vitals) | 30,187 | **70.4%** | 99.0% | 0.31 |
| Walk-in (vitals masked) | 50,327 | 66.8% | 97.6% | 0.36 |
| Other/Unknown (vitals masked) | 3,103 | 64.4% | 95.4% | 0.41 |

Ambulance/helicopter patients benefit directly from their real vitals (+3.6pp vs walk-ins).

---

## Training Pipeline Evolution

> The early v1-v3b versions are internal training iterations; the directory layout under `artifacts/triage/` only carves out three tiers (v1 = pre-vital, v2 = with vitals, v3 = v2 + PMH). The table below uses the iteration numbers used during development; the directory-level mapping is in the footnote.

| Iteration | Changes | Acuity | Disposition |
|---------|---------|--------|-------------|
| v1 | RandomForest + multi-hot top-100 complaints + pain only | 32.1% | 67.2% |
| v2 | XGBoost + TF-IDF 500 + severity priors + abbreviation expansion | 53.4% | 72.4% |
| v3 | + age, gender, arrival_transport | 58.4% | 75.6% |
| v3b | + TF-IDF 2000, 3000 trees, soft class weights, interaction features | 66.7% | 75.9% |
| v4 | + vital signs (ambulance/helicopter only), full 334K training set | 68.0% | 76.2% |
| **v5** | **+ 19 PMH features (Doctor v3 Change 1 recipe), GPU training** | **68.6%** | **78.0%** |

Directory mapping: `artifacts/triage/v1/` = the v3b iteration. `artifacts/triage/v2/` = the v4 iteration. `artifacts/triage/v3/` = the v5 iteration documented in the [v3 section](#triage-v3--pmh-features) below.

Training scripts (canonical paths via `src/proiect_licenta/paths.py`):
- `src/proiect_licenta/training/train_triage_v1.py` — v3b pipeline. Run with `uv run train_models`.
- `src/proiect_licenta/training/train_triage_v2.py` — v4 pipeline. Run with `uv run train_triage_v2` (~90 min CPU on 334K training rows).
- `src/proiect_licenta/training/train_triage_v3.py` — v5 / PMH pipeline. Run with `uv run train_triage_v3` (~30-45 min on Colab T4 GPU; ~120 min CPU because of the discharge.csv PMH parse).

---

## Tool

Implemented in `src/proiect_licenta/tools/triage_tool.py`. The tool lazy-loads both XGBoost models plus the TF-IDF vectorizer, severity map, and vital medians from `artifacts/triage/v2/`. It rebuilds the 2051-feature vector from the Parser's JSON and returns a structured result that the Doctor Agent consumes. v3 is benchmark-only — the inference tool has not yet been updated to load v3 artifacts or build the PMH columns. That's the next staged change; see the v3 section's "Inference status" note below.

**Vital sign handling at inference:**
- Vital inputs default to `-1` (unknown).
- If a vital value is `-1` or outside its physiological clip range, it is treated as missing: the raw value is set to the training median, the `*_missing` flag is set to 1, and all derived abnormality/interaction flags are set to 0.
- For ambulance/helicopter patients the NLP Parser collects EMS vitals and passes them through; for walk-in patients vitals remain at their defaults.

---

## Open Improvements

See [`../future-work.md`](../future-work.md) for the full roadmap. Triage-specific items:

- ~~Add more trees — best_iteration was 2999/3000, model had not yet converged.~~ Still open after v3 — best_iteration is still 2999/3000 on the acuity model. Lowering lr to 0.01 with n_estimators=5000 is the cheapest remaining lift.
- **Ordinal acuity objective.** ESI is an ordered scale 1-5; the current `multi:softprob` treats "predict 1 when truth is 5" the same as "predict 1 when truth is 2." Custom asymmetric sample weights or a QWK-objective sweep should improve κ and ESI 1 / ESI 5 recall (currently 58% / 14%).
- **Inference rewiring for v3.** Add a PMH prompt to the NLP Parser so the runtime gets the same prior-history signal the v3 benchmark relies on.
- Try LightGBM as an alternative to XGBoost.
- Ensemble methods (stacking multiple models).
- Neural network with learned embeddings for complaint text.

---

## Triage v3 — PMH features

Added on **2026-05-26**. Mirrors the Doctor v3 nurse Change 1 recipe (PMH features parsed from prior MIMIC encounters), applied to the full triage dataset (admitted + discharged, not just admitted).

### What it adds (19 new feature columns)

For each training stay, the pipeline looks at the same `subject_id`'s **strictly prior** hospital admissions and ED visits and computes the same 19-column block Doctor v3 uses:

- **13 binary `pmh_<diagnosis_group>` flags**, OR'd between two sources:
  - **Prior discharge-summary PMH sections** parsed from `data/mimic-iv-notes/.../discharge.csv` (~3.3 GB) via `extract_pmh_section()` + the 397-keyword vocabulary in [`pmh_vocab.py`](../../src/proiect_licenta/pmh_vocab.py). Negation-aware ("no history of CHF" doesn't flag).
  - **Prior ICD-derived flags** as fallback from `hosp/diagnoses_icd.csv`, mapped through the same ICD→category lookup used to build doctor supervision labels.
- **4 repeat-visit numerics**: `n_prior_admissions`, `n_prior_ed_visits`, `days_since_last_admission`, `days_since_last_ed`. Capped at `PMH_NO_PRIOR_DAYS = 9999` for first-time patients.
- **`same_complaint_as_prior`**: token-set Jaccard between the current normalized chief complaint and the patient's most recent prior ED chief complaint.
- **`no_history`**: 1 if the patient has no prior MIMIC encounters at all.

Total features now: **2070** (was 2051 in v2 → 19 PMH columns added).

The aggregator lives in [`src/proiect_licenta/pmh_features.py`](../../src/proiect_licenta/pmh_features.py) — extracted from `train_nurse_v3.py` as a shared helper so the doctor and triage pipelines call the same code. The triage pipeline imports `aggregate_pmh` and `fill_missing_pmh_columns` from there.

**Leakage:** zero by construction (same guarantee as Doctor Change 1). All PMH sources filter on `prior_*time < current_intime`.

### Benchmark results — v2 vs v3 (head-to-head on the same test rows)

Same held-out 83,617-row test split (random_state=42, stratified on acuity). Both models predict on the same rows; v2 uses 2051 features, v3 uses 2070.

| Metric | v2 | v3 | Δ |
|---|---|---|---|
| **Acuity exact accuracy** | 68.01% | **68.61%** | **+0.60pp** |
| Acuity within-1-level | 98.04% | **98.24%** | +0.20pp |
| Acuity within-2-level | 99.89% | 99.92% | +0.03pp |
| Acuity MAE | 0.3406 | 0.3323 | −0.0083 |
| Cohen's κ (linear) | 0.5452 | **0.5546** | +0.0094 |
| Cohen's κ (quadratic) | 0.6381 | **0.6477** | +0.0096 |
| Over-triage rate | 14.32% | 14.15% | −0.17pp |
| Under-triage rate | 17.67% | **17.24%** | **−0.43pp** |
| **Disposition accuracy** | 76.17% | **77.96%** | **+1.80pp** |
| **Disposition ROC AUC** | 0.8440 | **0.8643** | +0.0203 |

**The disposition lift (+1.80pp) is the headline result** — right in the middle of the predicted +1.5-2.5pp band. The acuity lift came in below the predicted band (+0.60pp vs predicted +1-2pp), but landed in clinically meaningful places (see below).

**Per-class acuity recall — v3:** ESI 1 = 58.28%, ESI 2 = 66.40%, ESI 3 = 72.13%, ESI 4 = 62.46%, ESI 5 = 14.09%.

PMH coverage on the test set: 50.9% of stays have ≥1 PMH flag set, 61.3% have ≥1 prior MIMIC encounter, 38.6% are first-time patients (`no_history=1`). Mean prior admissions = 3.36, mean prior ED visits = 3.43.

### Walk-in asymmetry — PMH does what we wanted

The single most interesting finding: **PMH lifts walk-ins more than ambulance patients**, validating the design hypothesis (PMH fills the signal gap where vitals are masked).

| Arrival transport | n | v2 exact | v3 exact | Δ |
|---|---|---|---|---|
| Ambulance/Helicopter (real vitals) | 30,187 | 70.40% | 70.74% | +0.34pp |
| Walk-in (vitals masked) | 50,327 | 66.80% | **67.51%** | **+0.72pp** |
| Other/Unknown (vitals masked) | 3,103 | 64.42% | **65.68%** | **+1.26pp** |

Walk-ins and "other" gain ~2-4× more than ambulance patients. For ambulance patients, real vital signs already carry most of the signal; for walk-ins, vitals are masked → PMH is the model's only way to know that this 60-year-old presenting with chest pain has prior CAD admissions.

### Feature-importance audit (verification gate)

| Check | Result |
|---|---|
| PMH features with non-zero gain (acuity model) | **19 / 19** (no silent merge failure) |
| PMH features in top 50 by gain (acuity model) | 0 (matches Doctor Change 1 pattern; top 50 is TF-IDF + vitals dominated) |
| PMH features with non-zero gain (disposition model) | **19 / 19** |
| **`pmh_Blood and Blood-Forming Organs` rank in disposition model** | **rank 5** |
| PMH features in disposition top 50 by gain | **4** (`pmh_Blood`, `pmh_Digestive`, `pmh_Circulatory`, `pmh_Endocrine`) |

The asymmetry between the two heads is real and clinically interpretable: chronic-condition flags are stronger predictors of *admission* than of *acuity*. ESI is largely about the acute presentation; the admit/discharge decision is much more influenced by prior cardiac / blood / endocrine history.

### Inference status

**v3 is benchmark-only as of 2026-05-26.** The runtime `triage_tool.py` still loads `artifacts/triage/v2/` and builds the 2051-feature vector without PMH columns. Wiring the v3 model into the live crew requires:

1. Adding a PMH prompt to the NLP Parser (or a step before triage) so the patient self-reports chronic conditions. The Doctor v3 path already has this prompt in the Nurse Agent — moving it earlier in the pipeline or adding a peer prompt to the Parser would do the job.
2. Updating `triage_tool.py` to load from `artifacts/triage/v3/` and build the 19-column PMH block from the Parser's output (or zero-fill with `no_history=1` if the patient skips).

Both changes are scoped but not yet implemented. Until then, triage v3 is a thesis benchmark, not a runtime artifact.

### Implementation

- **Training pipeline:** [`src/proiect_licenta/training/train_triage_v3.py`](../../src/proiect_licenta/training/train_triage_v3.py). Reuses the same XGBoost hyperparameters as v2 (3000 trees, max_depth=10, lr=0.02, early stopping=100). Honors `XGB_DEVICE` and `XGB_TREE_METHOD` env vars for Colab GPU training.
- **Shared PMH aggregator:** [`src/proiect_licenta/pmh_features.py`](../../src/proiect_licenta/pmh_features.py). Exports `PMH_FEATURE_COLS`, `PMH_NO_PRIOR_DAYS`, `aggregate_pmh()`, `fill_missing_pmh_columns()`. Both `train_triage_v3.py` and `train_nurse_v3.py` import from here.
- **Benchmark:** [`benchmarks/benchmark_triage_v3.py`](../../benchmarks/benchmark_triage_v3.py). Head-to-head v2 vs v3 on identical test rows, plus a PMH-feature audit.
- **Colab notebook:** [`notebooks/train_triage_v3.ipynb`](../../notebooks/train_triage_v3.ipynb), generated by [`scripts/build_colab_notebook_triage_v3.py`](../../scripts/build_colab_notebook_triage_v3.py). T4 GPU, ~30-45 min end-to-end (PMH parse dominates at ~15-25 min).
- **Paths:** `TRIAGE_V3_DIR` added to [`paths.py`](../../src/proiect_licenta/paths.py). Same on-disk layout as v1/v2.

### Result vs prediction, and why

The recommendation in `plans/read-the-documentation-of-tidy-scone.md` predicted +1-2pp on acuity and +1.5-2.5pp on disposition. Actual: **+0.60pp acuity, +1.80pp disposition**.

Why acuity came in below the band:

1. **PMH is more about chronic disease than acute presentation.** ESI is heavily driven by what the patient walked in with *today* — pain, vital signs, red-flag language. Prior CHF tells you the patient has cardiac history; it doesn't reliably tell you if today's visit is ESI 2 or ESI 3.
2. **~39% of test rows are first-time patients (`no_history=1`).** For those rows all 13 PMH flags are zero and v3 contributes nothing on acuity beyond noise.
3. **The acuity model's `best_iteration` is still 2999/3000.** The model hasn't converged — section 1.1 of the original recommendation (more trees / lower lr) remains the cheapest untapped lift.

Why disposition matched the band cleanly:

1. **Prior admission rate is a near-deterministic predictor of admission.** A patient with 8 prior MIMIC admissions in the last 2 years is overwhelmingly likely to be admitted again. The `n_prior_admissions` and `days_since_last_admission` numerics are doing real work.
2. **Specific PMH categories matter for admission decisions.** `pmh_Blood and Blood-Forming Organs` at rank 5 in the disposition model is the smoking gun — sickle cell, anemia, leukemia priors flip the admit threshold.

Both observations are clinically sensible and align with how human triage nurses work: vitals + chief complaint set the ESI; medical history sets the admit threshold.
