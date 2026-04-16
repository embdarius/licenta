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

Stored in `src/proiect_licenta/models/`.

### Model 1: Acuity Prediction (ESI 1-5)
- **File:** `acuity_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02)
- **Training data:** 334K samples (80/20 split from 418K cleaned rows)
- **Features (2023 total):**
  - `pain` score (0-10, or -1 if missing)
  - `pain_missing`, `pain_low`, `pain_mid`, `pain_high` (binary flags)
  - `n_complaints` (count of chief complaints)
  - `complaint_length` (character length of normalized text)
  - Severity priors: `min_severity_prior`, `mean_severity_prior`, `max_severity_prior`, `std_severity_prior` (mean acuity per complaint word from training data)
  - `age`, `age_bin` (binned into 6 groups: 0-18, 18-35, 35-50, 50-65, 65-80, 80+)
  - `gender_male` (binary)
  - `arrival_ambulance`, `arrival_helicopter`, `arrival_walk_in` (binary)
  - Interaction features: `age_ambulance`, `pain_x_min_severity`, `age_severity`, `high_pain_ambulance`, `elderly`, `elderly_ambulance`
  - 2000 TF-IDF features (word unigrams, bigrams, trigrams from chief complaint text)
- **Class weighting:** Soft balanced (sqrt of inverse frequency).
- **Note:** Model outputs classes 0-4 internally; add 1 to get ESI 1-5.

### Model 2: Disposition Prediction (Admit vs. Discharge)
- **File:** `disposition_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02)
- **Features:** Same as the acuity model **plus** `predicted_acuity` (cascading: acuity is predicted first, then used as input for disposition).

### Supporting Artifacts
- `tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer (2000 features, trigrams).
- `severity_map.joblib` — Dict mapping complaint words to their mean acuity (819 words).
- `model_metadata.json` — Training date, accuracy metrics, feature list, model params.

---

## Benchmark Results

Test set: **83,617 samples**.

| Metric | Value |
|--------|-------|
| **Acuity exact accuracy** | **66.7%** |
| **Acuity within-1-level** | **97.7%** |
| Acuity within-2-level | 99.9% |
| Acuity MAE | 0.36 ESI levels |
| Acuity quadratic kappa | 0.62 |
| **Disposition accuracy** | **75.9%** |
| Disposition ROC AUC | 0.84 |
| Over-triage rate | 15.1% |
| Under-triage rate | 18.2% |

**Per-class acuity recall:** ESI 1: 49.8%, ESI 2: 63.5%, ESI 3: 71.5%, ESI 4: 60.4%, ESI 5: 22.7%.

### Key Findings
- The model is strongest on ESI 3 (majority class, 54%) and weakest on ESI 1/5 (minority classes).
- 97.7% within-1 accuracy means catastrophic misclassifications are very rare.
- Slight lean toward under-triage (55% of errors) vs over-triage (45%).
- Top features: severity priors, `pain_missing`, complaint-specific TF-IDF terms.

---

## Training Pipeline Evolution

Tracked so future work can decide which levers still matter.

| Version | Changes | Acuity | Disposition |
|---------|---------|--------|-------------|
| v1 | RandomForest + multi-hot top-100 complaints + pain only | 32.1% | 67.2% |
| v2 | XGBoost + TF-IDF 500 + severity priors + abbreviation expansion | 53.4% | 72.4% |
| v3 | + age, gender, arrival_transport from edstays/patients tables | 58.4% | 75.6% |
| v3b | + TF-IDF 2000, 3000 trees, soft class weights, interaction features | **66.7%** | **75.9%** |

Training script: `src/proiect_licenta/data_pipeline.py`. Run with `uv run train_models` (~90 minutes on the sampled 334K training set).

---

## Tool

Implemented in `src/proiect_licenta/tools/triage_tool.py`. The tool lazy-loads both XGBoost models plus the TF-IDF vectorizer and severity map, rebuilds the 2023-feature vector from the Parser's JSON, and returns a structured result that the Doctor Agent (both v1 and v2) consumes.

---

## Open Improvements

See [`../future-work.md`](../future-work.md) for the full roadmap. Triage-specific headline items:
- Add more trees (best_iteration was ~2990 — the model hadn't converged).
- Include triage vital signs from `triage.csv` (currently unused; they are taken at triage time and would materially boost accuracy).
- Try LightGBM as an alternative to XGBoost.
- Ensemble methods (stacking multiple models).
- Neural network with learned embeddings for complaint text.
