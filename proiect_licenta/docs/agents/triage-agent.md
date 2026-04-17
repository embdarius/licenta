# Triage Agent

The Triage Agent predicts ESI acuity (1-5) and admission vs. discharge from the structured output of the NLP Parser. It is the first ML stage in the cascading pipeline.

---

## Role

- **Type:** CrewAI agent with a custom tool (`TriagePredictionTool`) wrapping two XGBoost models.
- **Purpose:** Predict ESI acuity level (1-5) and admission/discharge disposition.
- **Config:** `src/proiect_licenta/config/agents.yaml` (`triage_agent`) and `tasks.yaml` (`triage_assessment_task`).
- **Output contract:** Emits a structured data block downstream tasks can parse.

---

## Models

Stored in `src/proiect_licenta/models_v2/`.

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

| Version | Changes | Acuity | Disposition |
|---------|---------|--------|-------------|
| v1 | RandomForest + multi-hot top-100 complaints + pain only | 32.1% | 67.2% |
| v2 | XGBoost + TF-IDF 500 + severity priors + abbreviation expansion | 53.4% | 72.4% |
| v3 | + age, gender, arrival_transport | 58.4% | 75.6% |
| v3b | + TF-IDF 2000, 3000 trees, soft class weights, interaction features | 66.7% | 75.9% |
| **v4** | **+ vital signs (ambulance/helicopter only), full 334K training set** | **68.0%** | **76.2%** |

Training scripts:
- `src/proiect_licenta/triage_pipeline_v1.py` — v3b pipeline. Run with `uv run train_models`.
- `src/proiect_licenta/triage_pipeline_v2.py` — v4 pipeline (current). Run with `uv run train_triage_v2` (~90 minutes on 334K training rows).

---

## Tool

Implemented in `src/proiect_licenta/tools/triage_tool.py`. The tool lazy-loads both XGBoost models plus the TF-IDF vectorizer, severity map, and vital medians from `models_v2/`. It rebuilds the 2051-feature vector from the Parser's JSON and returns a structured result that the Doctor Agent consumes.

**Vital sign handling at inference:**
- Vital inputs default to `-1` (unknown).
- If a vital value is `-1` or outside its physiological clip range, it is treated as missing: the raw value is set to the training median, the `*_missing` flag is set to 1, and all derived abnormality/interaction flags are set to 0.
- For ambulance/helicopter patients the NLP Parser collects EMS vitals and passes them through; for walk-in patients vitals remain at their defaults.

---

## Open Improvements

See [`../future-work.md`](../future-work.md) for the full roadmap. Triage-specific items:

- Add more trees — best_iteration was 2999/3000, model had not yet converged.
- Try LightGBM as an alternative to XGBoost.
- Ensemble methods (stacking multiple models).
- Neural network with learned embeddings for complaint text.
