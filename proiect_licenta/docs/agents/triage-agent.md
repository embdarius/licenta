# Triage Agent

The Triage Agent predicts ESI acuity (1-5) and admission vs. discharge from the structured output of the NLP Parser. It is the first ML stage in the cascading pipeline.

A **v3 tier** sits alongside v1/v2 (it does not replace them yet at runtime). v3 keeps the v2 vital-sign feature set and adds 19 PMH features parsed from prior MIMIC `discharge.csv` notes + `diagnoses_icd.csv`, using the same recipe Doctor v3 nurse uses (Change 1). v3 has been retrained twice:

- **Iteration 1 (2026-05-26)** — initial PMH integration. +0.60pp acuity, +1.80pp disposition vs v2.
- **Iteration 2 (2026-05-27, kept)** — adds section 1.1 (longer training, lr 0.02→0.01, n_estimators 3000→5000) + section 1.2 (ordinal-aware acuity weighting + QWK early stopping). −0.46pp acuity vs v2 but **−2.11pp under-triage** and ESI 5 recall recovered from a regression. Kept for clinical-safety reasons over iter 1's higher headline accuracy.

See the [v3 section](#triage-v3) below for the iteration-by-iteration breakdown. v1 and v2 stay in the repo as baselines for direct thesis comparison.

---

## Role

- **Type:** CrewAI agent with two tools — `TriagePredictionTool` (wrapping two XGBoost models) and `PatientHistoryLookupTool` (EHR-simulation PMH lookup for returning patients, keyed on MRN / `subject_id`).
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
| v5 (v3 iter 1) | + 19 PMH features (Doctor v3 Change 1 recipe), GPU training | 68.6% | 78.0% |
| **v5b (v3 iter 2, kept)** | **+ longer training (lr 0.01, n_est 5000) + ordinal-aware acuity (extreme-class boost + QWK early stopping)** | **67.55%** | **77.98%** |

Directory mapping: `artifacts/triage/v1/` = the v3b iteration. `artifacts/triage/v2/` = the v4 iteration. `artifacts/triage/v3/` = the v5b iteration kept on disk (iter 1's artifacts were overwritten by iter 2). See the [v3 section](#triage-v3) below for the per-iteration deltas and the rationale for keeping iter 2 despite its lower headline accuracy.

Training scripts (canonical paths via `src/proiect_licenta/paths.py`):
- `src/proiect_licenta/training/train_triage_v1.py` — v3b pipeline. Run with `uv run train_models`.
- `src/proiect_licenta/training/train_triage_v2.py` — v4 pipeline. Run with `uv run train_triage_v2` (~90 min CPU on 334K training rows).
- `src/proiect_licenta/training/train_triage_v3.py` — v5 / PMH pipeline. Run with `uv run train_triage_v3` (~30-45 min on Colab T4 GPU; ~120 min CPU because of the discharge.csv PMH parse).

---

## Tool

Implemented in `src/proiect_licenta/tools/triage_tool.py`. The tool lazy-loads both XGBoost models plus the TF-IDF vectorizer, severity map, and vital medians from **`artifacts/triage/v3/`** (the iter-2 kept stack). It rebuilds the **2070-feature vector** (23 v1 structured + 28 v2 vital + 19 PMH + 2000 TF-IDF) from the Parser's JSON and returns a structured result that the Doctor Agent consumes.

**Vital sign handling at inference:**
- Vital inputs default to `-1` (unknown).
- If a vital value is `-1` or outside its physiological clip range, it is treated as missing: the raw value is set to the training median, the `*_missing` flag is set to 1, and all derived abnormality/interaction flags are set to 0.
- For ambulance/helicopter patients the NLP Parser collects EMS vitals and passes them through; for walk-in patients vitals remain at their defaults.

**PMH handling at inference (v3, added 2026-05-29):**
- The NLP Parser asks every patient (walk-in and ambulance/helicopter) about chronic conditions and approximate prior admission count as part of intake.
- `prior_history` (free-text string) is parsed at inference via the same `pmh_vocab` the training pipeline uses — so "CHF, diabetes, prior stroke" fires the `pmh_Circulatory`, `pmh_Endocrine, Nutritional, Metabolic`, and `pmh_Nervous System and Sense Organs` flags without any additional patient questions.
- `n_prior_admissions` (integer) is used directly. On the **self-report path** `days_since_last_admission` / `days_since_last_ed` / `same_complaint_as_prior` / `n_prior_ed_visits` are zero-filled (not collectible at the bedside) — the model saw the same pattern on ~39% of training rows (first-time MIMIC patients), so the runtime is consistent with training-time distribution.
- If the patient skips both PMH prompts (`prior_history=""`, `n_prior_admissions=-1`), the tool zero-fills with `no_history=1` — the exact "first-time patient" pattern the v3 model learned to handle gracefully.
- **MRN / EHR lookup path (added 2026-06-03).** When a *returning* patient provides an MRN, the triage agent first calls `PatientHistoryLookupTool` and passes the resulting real prior-encounter `pmh_block` to the triage tool as `pmh_lookup_json`. When present it **overrides** the self-report and supplies the four numerics a patient can't recall (`days_since_last_admission`, `days_since_last_ed`, `n_prior_ed_visits`, `same_complaint_as_prior`) with their true values — the exact columns the feature-vector benchmark always had but runtime triage previously zero-filled. See [the v3 MRN-lookup subsection](#mrn-based-ehr-lookup-at-triage--wired-2026-06-03). First-time / unknown patients (`subject_id = -1`, or `known_patient: false`) keep the self-report path unchanged.
- The result JSON includes a `prior_history_used` block summarizing which PMH category flags fired, whether the patient was treated as first-time, and — when the EHR lookup was used — `used_history_lookup: true`, the real `history_lookup_numerics`, and a read-only `self_report_not_in_record` reconciliation list (conditions the patient reported that the record didn't contain; the record still drives the prediction). The downstream Doctor Agent can surface that reasoning in its clinical note.

---

## Open Improvements

See [`../future-work.md`](../future-work.md) for the full roadmap. Triage-specific items:

- ~~Add more trees — best_iteration was 2999/3000, model had not yet converged.~~ **DONE in v3 iter 2** (lr 0.02→0.01, n_estimators 3000→5000). Acuity converged at best_iteration=2405/5000; disposition still hits the ceiling at 4999/5000 but accuracy is flat past iteration 2000, so further training has ≤0.2pp expected upside.
- ~~**Ordinal acuity objective.**~~ **DONE in v3 iter 2** via `ESI_EXTREME_BOOST` + `neg_quadratic_kappa` early stopping. ESI 5 recall recovered from 14% to 26.8%; ESI 1, ESI 2 both gained; under-triage rate dropped 2.11pp vs v2.
- ~~**Hand-curated red-flag keyword features (section 1.5).**~~ **TRIED AND REVERTED in v3 iter 3 (2026-05-28).** Headline metrics within noise floor (±0.04pp); under-triage +0.02pp regressed. See "Tried and reverted" subsection below.
- ~~**Inference rewiring for v3.**~~ **DONE (2026-05-29).** Runtime now loads `artifacts/triage/v3/` and collects `prior_history` + `n_prior_admissions` from the patient at parser intake. See [Inference status](#inference-status--wired-2026-05-29) below for details.
- Try LightGBM as an alternative to XGBoost.
- Ensemble methods (stacking multiple models).
- Neural network with learned embeddings for complaint text.

---

## Triage v3

v3 sits alongside v1 and v2. It keeps the v2 feature set (vital signs, abnormality flags, walk-in vitals masking) and adds **19 PMH (Past Medical History) feature columns** derived from prior MIMIC encounters — the same recipe Doctor v3 nurse uses (Change 1).

v3 has been retrained twice. The current production artifacts on disk (`artifacts/triage/v3/`) are **iteration 2**, kept for clinical-safety reasons over iteration 1's higher headline accuracy. The full iteration history is preserved in this section.

### The 19 PMH features (used by both iterations)

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

PMH coverage on the test set (83,617 stays): 50.9% have ≥1 PMH flag set, 61.3% have ≥1 prior MIMIC encounter, 38.6% are first-time patients (`no_history=1`). Mean prior admissions = 3.36, mean prior ED visits = 3.43.

---

### Iteration 1 — PMH features only (2026-05-26)

Initial v3 integration. Same XGBoost hyperparameters as v2 (3000 trees, max_depth=10, lr=0.02, early stopping=100). The only difference vs v2 was the 19 PMH columns added to the feature matrix.

**Benchmark vs v2** (same 83,617-row test split, random_state=42):

| Metric | v2 | v3 iter 1 | Δ |
|---|---|---|---|
| Acuity exact accuracy | 68.01% | **68.61%** | **+0.60pp** |
| Acuity within-1-level | 98.04% | **98.24%** | +0.20pp |
| Acuity MAE | 0.3406 | **0.3323** | −0.0083 |
| Cohen's κ (quadratic) | 0.6381 | **0.6477** | +0.0096 |
| Over-triage rate | 14.32% | 14.15% | −0.17pp |
| Under-triage rate | 17.67% | 17.24% | −0.43pp |
| Disposition accuracy | 76.17% | **77.96%** | **+1.80pp** |
| Disposition ROC AUC | 0.8440 | **0.8643** | +0.0203 |

**Per-class acuity recall** (iter 1): ESI 1 = 58.28%, ESI 2 = 66.40%, ESI 3 = 72.13%, ESI 4 = 62.46%, **ESI 5 = 14.09%**.

**Walk-in asymmetry — PMH does what we wanted.** PMH lifted walk-in patients (no vitals available) more than ambulance patients (vitals available), validating the design hypothesis:

| Arrival transport | n | v2 exact | iter 1 exact | Δ |
|---|---|---|---|---|
| Ambulance/Helicopter | 30,187 | 70.40% | 70.74% | +0.34pp |
| **Walk-in** | 50,327 | 66.80% | **67.51%** | **+0.72pp** |
| **Other/Unknown** | 3,103 | 64.42% | **65.68%** | **+1.26pp** |

**Feature-importance audit** (iter 1):

| Check | Result |
|---|---|
| PMH features with non-zero gain (acuity) | 19 / 19 |
| PMH features in top 50 by gain (acuity) | 0 |
| PMH features with non-zero gain (disposition) | 19 / 19 |
| `pmh_Blood and Blood-Forming Organs` rank (disposition) | **rank 5** |
| PMH features in disposition top 50 | 4 |

Chronic-condition flags are stronger predictors of *admission* than of *acuity* — `pmh_Blood and Blood-Forming Organs` at rank 5 in the disposition model is the smoking gun (sickle cell, anemia, leukemia priors flip the admit threshold).

**The hidden regression that motivated iteration 2.** Iter 1 looks clean at the headline level, but **ESI 5 recall dropped from 21.8% (v2) to 14.09%** — PMH features for the rare, mostly-first-time-patient ESI 5 cohort are noisy and the model deprioritized them. ESI 5 is non-urgent so the clinical risk is low, but the regression is real and motivated rebalancing the per-class weighting.

### Iteration 2 — Section 1.1 (longer training) + Section 1.2 (ordinal-aware acuity), 2026-05-27 (kept)

Two stacked changes on top of iter 1, retrained in a single Colab T4 GPU pass (~44 min total):

**Section 1.1 — longer training** (both heads):
- `learning_rate`: 0.02 → 0.01
- `n_estimators`: 3000 → 5000
- `early_stopping_rounds`: 100 → 150

Iter 1's `best_iteration` was 2999/3000 (acuity) and 2983/3000 (disposition) — neither head had converged. Lower lr + more trees gives the booster room to actually find a minimum. **Result:** acuity converged at `best_iteration = 2405/5000` (well clear of the ceiling). Disposition still hit the ceiling at 4999/5000 — logloss kept dropping monotonically all the way through, but accuracy was already flat by ~iteration 2000, so additional convergence yielded no meaningful lift.

**Section 1.2 — ordinal-aware acuity** (acuity head only — disposition is binary, no ordinal structure):

- **Extreme-class sample-weight boost** — multiplies the existing sqrt-balanced weights by a per-class factor:

  | Class | Boost factor | Rationale |
  |---|---|---|
  | ESI 1 (resuscitation) | 1.5× | Critical; under-triage = death risk |
  | ESI 2 (emergent) | 1.3× | Most clinically dangerous to under-triage |
  | ESI 3 (urgent, dominant) | 1.0× | Baseline |
  | ESI 4 (less urgent) | 1.0× | Baseline |
  | ESI 5 (non-urgent) | 2.0× | Iter 1 recall was a catastrophic 14% |

  With the underlying sqrt-balance, the *effective* training-time weight on the rarest class (ESI 5) ends up ~26× the weight on the dominant class (ESI 3), pushing gradient updates aggressively toward the critical extremes.

- **QWK eval metric for early stopping** — replaces `eval_metric="mlogloss"` with a custom `neg_quadratic_kappa` callable. The booster still trains via the standard `multi:softprob` loss (proper, smooth) but early stopping now picks the iteration with the highest quadratic-weighted κ. In practice the picked iteration is similar to what mlogloss would have chosen (sample-weighting is the dominant lever; QWK is mostly for honest reporting).

**Benchmark vs v2 and iter 1** (same 83,617-row test split):

| Metric | v2 | iter 1 | **iter 2 (kept)** | iter 2 vs v2 | iter 2 vs iter 1 |
|---|---|---|---|---|---|
| Acuity exact | 68.01% | 68.61% | 67.55% | −0.46pp | −1.06pp |
| Acuity within-1 | 98.04% | 98.24% | 98.18% | +0.14pp | −0.06pp |
| Acuity MAE | 0.3406 | 0.3323 | 0.3437 | +0.0031 | +0.0114 |
| Cohen's κ (quadratic) | 0.6381 | 0.6477 | 0.6449 | +0.0068 | −0.0028 |
| Over-triage rate | 14.32% | 14.15% | 16.90% | **+2.58pp** | +2.75pp |
| **Under-triage rate** | 17.67% | 17.24% | **15.56%** | **−2.11pp** | **−1.68pp** |
| Disposition accuracy | 76.17% | 77.96% | **77.98%** | +1.81pp | +0.02pp |
| Disposition ROC AUC | 0.8440 | 0.8643 | **0.8644** | +0.0204 | +0.0001 |

**Per-class acuity recall** — this is where iter 2's design intent is visible:

| Class | v2 | iter 1 | iter 2 | iter 2 vs v2 | iter 2 vs iter 1 |
|---|---|---|---|---|---|
| ESI 1 (resuscitation) | 58.8% | 58.28% | **60.30%** | +1.5pp | +2.0pp |
| ESI 2 (emergent) | 65.4% | 66.40% | **70.83%** | **+5.4pp** | +4.4pp |
| ESI 3 (urgent, dominant) | 71.8% | 72.13% | 67.23% | −4.6pp | −4.9pp |
| ESI 4 (less urgent) | 60.7% | 62.46% | 61.67% | +1.0pp | −0.8pp |
| **ESI 5 (non-urgent)** | **21.8%** | 14.09% | **26.82%** | **+5.0pp** | **+12.7pp** |

ESI 1, ESI 2, and ESI 5 — the clinically critical classes — all improved vs both v2 and iter 1. ESI 3 (the dominant class, 54% of the test set) lost ground, and that loss dominates the headline accuracy. The errors don't disappear; they shift from ESI 2→ESI 3 (under-triage of an emergent patient — clinically dangerous) to ESI 3→ESI 2 (over-triage of an urgent patient — clinically benign, patient gets seen sooner).

**By arrival transport** (iter 2):

| Group | n | v2 | iter 2 | Δ vs v2 |
|---|---|---|---|---|
| Ambulance/Helicopter | 30,187 | 70.40% | 69.34% | −1.06pp |
| Walk-in | 50,327 | 66.80% | 66.63% | −0.16pp |
| Other/Unknown | 3,103 | 64.42% | 64.94% | +0.52pp |

The walk-in lift that iter 1 surfaced got partially eaten by the ESI 3 reshuffle in iter 2 — but the under-triage drop applies to all transport groups, so the safety case still holds.

### Why we kept iteration 2

Three reasons, in order of weight:

1. **No per-class regression vs v2.** Iter 1 had a hidden regression on ESI 5 (21.8% → 14.09%, −7.7pp) that a thesis examiner would surface on inspection. Iter 2 improves every clinically critical class (ESI 1, 2, 5) vs v2. The headline accuracy is slightly lower, but the per-class story is monotonically better on the classes that matter.
2. **Under-triage is the clinically meaningful metric.** ED triage literature is clear: under-triage delays care for sick patients (mortality risk), while over-triage costs only nurse-time. Iter 2 reduces under-triage by 2.11pp on 83,617 patients ≈ ~1,765 fewer dangerous misses. That's a more meaningful clinical number than the 0.46pp drop in headline accuracy.
3. **Disposition is identical between iter 1 and iter 2** (77.96% → 77.98%). Choosing between iterations has zero impact on the disposition story — both deliver the +1.81pp v2 lift.

The cost — a 0.46pp headline accuracy drop, a 2.58pp over-triage increase, and a +0.0114 MAE rise — is consistent with the design intent and is documented honestly here for thesis-defense scrutiny.

### Inference status — wired (2026-05-29)

**v3 (iter 2) is now the runtime model.** `triage_tool.py` loads `artifacts/triage/v3/` and builds the 2070-feature vector including the 19-column PMH block on every call. Two coordinated changes landed together:

1. **NLP Parser collects PMH at intake.** `config/tasks.yaml`'s `parse_symptoms_task` now instructs the parser to ask every patient (walk-in and ambulance/helicopter alike) about (a) chronic conditions / prior medical history as free text and (b) approximate prior admission count. Both can be skipped; skip-equivalents (`""`, "no", "skip", "none", "I don't know", "no significant pmh") all map to the first-time-patient pattern.
2. **`triage_tool.py` parses and applies PMH inline.** Same `pmh_flags_from_text` + `PMH_CATEGORIES` + `PMH_NO_PRIOR_DAYS` machinery the doctor v3 tool uses, so the two heads share inference-side PMH parsing. On the self-report path `days_since_last_admission` / `days_since_last_ed` / `same_complaint_as_prior` are zero-filled (uncollectable at bedside, model trained on the same pattern via the 39% first-time-patient rows). The [MRN-lookup path](#mrn-based-ehr-lookup-at-triage--wired-2026-06-03) below recovers those numerics for returning patients.

The result JSON's new `prior_history_used` block surfaces which PMH categories fired so the Doctor Agent can include "Patient's prior CHF and diabetes informed the prediction" in its clinical reasoning.

### MRN-based EHR lookup at triage — wired (2026-06-03)

Triage v3 consumes a 19-feature PMH block, but until this change runtime triage only ever saw **self-reported** PMH and zero-filled the four numerics a patient can't recall (`days_since_last_admission`, `days_since_last_ed`, `n_prior_ed_visits`, `same_complaint_as_prior`) — even for a returning patient who gave an MRN — because triage runs *before* what used to be the only EHR-lookup call (the doctor disposition task). The triage feature-vector benchmark, by contrast, always fed the real block. This wired the same MRN→record lookup the doctor disposition/reassessment tasks already use into triage, closing that gap for returning patients.

**What changed:**
- **`PatientHistoryLookupTool` registered on the triage agent** (`crew.py`). The agent calls it directly — the lookup is deterministic given `subject_id` + `"now"` and its index is process-cached, so each stage that needs PMH calls it independently rather than threading a 19-key JSON blob verbatim through the `STRUCTURED_DATA` text hops (the chosen design over a "fire-once-and-propagate" variant, for robustness).
- **`triage_tool.py` gained an optional `pmh_lookup_json` arg** (mirroring the doctor tools). When a valid block is present it **overrides** the text-derived PMH `pmh_data` for all 19 columns; when empty/invalid it falls back to self-report. The parse + override use the shared `parse_pmh_lookup` helper.
- **`tasks.yaml`'s `triage_assessment_task` gained step "1b"**: if `subject_id` is a positive integer, call `patient_history_lookup_tool` with `current_intime="now"` first, then pass the returned `pmh_block` as `pmh_lookup_json` to the triage tool. `subject_id = -1` / `known_patient: false` → empty string → unchanged self-report path.
- **Read-only reconciliation.** The record always wins the feature vector (matching training). A new `self_report_not_in_record` field surfaces conditions the patient mentioned that the record lacks (possible outside-system / new diagnosis) without altering the prediction.

**Shared-helper relocation.** `parse_pmh_lookup` (previously duplicated in `doctor_disposition_tool.py` and `doctor_tool_v3.py`) and the new `pmh_self_report_discrepancy` now live in [`pmh_features.py`](../../src/proiect_licenta/pmh_features.py) as the single source of truth; triage and both doctor tools import from there. All three tools now emit `self_report_not_in_record`.

**Note on the doctor side:** the initial doctor (`doctor_tool_v3_base`) has **no** PMH features, so it is *not* a PMH consumer and was intentionally left untouched. The disposition + reassessment tools already called the lookup and are unchanged.

**Operator note: iter-2 v3 weights required on disk.** The runtime expects `artifacts/triage/v3/` to contain the **iter-2** kept artifacts (2070 features). If your Drive currently has iter-3 weights (2114 features, includes the reverted red-flag columns), re-train via the Colab notebook against the post-revert HEAD to restore iter-2 weights. The smoke tests in `tools/triage_tool.py` exercise feature-vector shape (asserts 2070 cols) so a weight mismatch will fail loudly at first call rather than silently corrupting predictions.

### Implementation

- **Training pipeline:** [`src/proiect_licenta/training/train_triage_v3.py`](../../src/proiect_licenta/training/train_triage_v3.py). Module-level constants (`ACUITY_N_ESTIMATORS`, `ACUITY_LEARNING_RATE`, `ESI_EXTREME_BOOST`, etc.) make the iter-2 config one-line tunable. Defines the custom `neg_quadratic_kappa` eval metric (stripped from the model via `set_params(eval_metric=None)` before `joblib.dump` so the saved artifact is pickle-clean). Honors `XGB_DEVICE` and `XGB_TREE_METHOD` env vars for Colab GPU training.
- **Shared PMH aggregator:** [`src/proiect_licenta/pmh_features.py`](../../src/proiect_licenta/pmh_features.py). Exports `PMH_FEATURE_COLS`, `PMH_NO_PRIOR_DAYS`, `aggregate_pmh()`, `fill_missing_pmh_columns()`. Both `train_triage_v3.py` and `train_nurse_v3.py` import from here.
- **Benchmark:** [`benchmarks/benchmark_triage_v3.py`](../../benchmarks/benchmark_triage_v3.py). Head-to-head v2 vs v3 on identical test rows, plus a PMH-feature audit.
- **Colab notebook:** [`notebooks/train_triage_v3.ipynb`](../../notebooks/train_triage_v3.ipynb), generated by [`scripts/build_colab_notebook_triage_v3.py`](../../scripts/build_colab_notebook_triage_v3.py). T4 GPU, ~40-60 min end-to-end for iter 2 (PMH parse ~15-25 min, longer XGBoost training ~20-35 min).
- **Paths:** `TRIAGE_V3_DIR` added to [`paths.py`](../../src/proiect_licenta/paths.py). Same on-disk layout as v1/v2.
- **Iteration 2 metadata:** the saved `model_metadata.json` records `iteration: "1.1+1.2 (longer training + ordinal-aware acuity)"` and a `training_config` block with all hyperparameters and the `esi_extreme_boost` dict for reproducibility.

### Open knobs for follow-up tuning

- **Disposition hit the 5000-tree ceiling.** Iter 2's disposition was flat vs iter 1 despite hitting `best_iteration = 4999/5000`. Dropping `DISP_LEARNING_RATE` to 0.005 or pushing `DISP_N_ESTIMATORS` to 10000 would let it converge, but the iter-1-to-iter-2 disposition delta was +0.02pp, so the upside is probably ≤0.2pp. Not currently scheduled.
- **ESI 5 boost is tunable.** If a future re-train wants to recover some headline accuracy without giving up the under-triage win, dialing the ESI 5 boost from 2.0× to 1.5× (or 1.2×) is the obvious knob. Estimated trade: ~+0.3pp headline accuracy for ~-3pp of the ESI 5 lift. Not recommended for iter 2's documented results, but worth knowing.
- **QWK eval is mostly cosmetic.** Empirically the sample-weight boost is the dominant lever; the QWK early-stopping picked an iteration very close to what mlogloss would have chosen. The metric is kept because it's a more honest *report* of model quality, not because it changes which model gets saved.

### Hyperparameter search — constrained Optuna over Group-2 (2026-06-26, reporting-only)

The iter-2 hyperparameters split into two groups. **Group 1** (`lr`,
`n_estimators`, `early_stopping`, `ESI_EXTREME_BOOST`, `neg_quadratic_kappa`) is
the documented clinical-safety config above. **Group 2** — the eight XGBoost
regularization knobs (`max_depth`, `subsample`, `colsample_bytree`,
`colsample_bylevel`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`) —
was inherited unchanged from v1/v2 and had **never been systematically searched**.
This sweep closes that gap. It is **reporting-only**: it never overwrites the
live model (everything lands in `artifacts/triage/v3/hpo/`); the goal is a
defensible answer to "were those regularization values justified?", not a new
deployment.

- **Engine:** [`scripts/tune_triage_v3.py`](../../scripts/tune_triage_v3.py); **notebook:** [`notebooks/tune_triage_v3.ipynb`](../../notebooks/tune_triage_v3.ipynb); **logs:** [`docs/results/triage_hpo/`](../results/triage_hpo/).
- **Method:** Optuna TPE, single inner train/val split (`random_state=1`) of the outer-train, native `mlogloss` early-stopping during the search for speed (the trial is still **scored** by QWK), resumable SQLite study on Drive. Mirrors the doctor sweep (`scripts/tune_doctor_v3.py`).
- **Acuity objective:** maximize QWK **subject to a hard constraint** — under-triage ≤ the in-sample incumbent baseline. The search therefore **cannot trade away the iter-2 under-triage win**; infeasible trials are excluded from selection.
- **Disposition objective:** maximize ROC AUC (binary head, no ordinal/safety constraint; the operating point is set separately by threshold, as in the doctor disposition story).

**Results (inner-validation, 66,894 rows; full per-trial logs in `docs/results/triage_hpo/`):**

| Head | Trials | Best searched config | Read |
|---|---|---|---|
| Acuity | 9 | trial #5 — QWK **0.6448**, under-triage 15.38%, within-1 98.07% | within the noise floor of the incumbent (QWK ≈ 0.6425–0.6449); **no config beats it**. QWK range across all trials 0.6357–0.6448. The constraint correctly marked the one trial that raised under-triage above baseline (#6, under 15.60%) as **infeasible**. |
| Disposition | 10 | **trial #0 — the incumbent config** — ROC AUC **0.8641** | the deployed Group-2 values are the **outright best**; all searched configs scored ROC AUC 0.8565–0.8641. |

**Conclusion:** the inherited Group-2 regularization is **near-optimal** — exactly
the most-likely-and-still-reportable outcome. Disposition's hand-tuned config won
outright; acuity's sits in a flat basin where every feasible config is within
~0.002 QWK. The previously-unjustified regularization box is now a *defended*
choice rather than an inherited one. The headline `best_iteration ≈ 4999` on
every trial reflects `mlogloss` (a soft-probability metric) still falling at the
tree ceiling — **not** a sign QWK needs more trees; QWK itself plateaus by
~2,400 trees (see iter-2 `best_iteration = 2405` under the real metric), so this
is *not* a reason to raise `n_estimators` (a Group-1 knob, out of scope here).

**Honest caveats for the thesis:** (1) these are inner-validation search metrics
— the definitive incumbent-vs-best comparison on the held-out test split comes
from `--stage report` (`triage_hpo_results.json`); (2) the acuity study's
constraint baseline is trial #2 (the enqueued incumbent trial-0 didn't complete
during the initial CPU run), which doesn't change the conclusion given the tight
clustering; disposition's trial-0 incumbent completed normally and is the winner.

### Hyperparameter search — Group-1 (multi-objective Pareto, reporting-only)

The Group-1 sweep (the clinical-safety levers themselves) is now **implemented**
as a sibling study in the same engine + notebook. Because `ESI_EXTREME_BOOST`,
the disposition decision threshold, and `scale_pos_weight` *are* the safety
levers, this is a **multi-objective Pareto** search (NSGA-II), not a constrained
single-objective one — exactly the framing the future-work note below prescribed.
Group-2 is frozen at the original incumbent (`FROZEN_GROUP2`) so the study
isolates Group-1; everything lands in parallel `*_g1` files and the live model is
never touched.

- **Engine:** [`scripts/tune_triage_v3.py`](../../scripts/tune_triage_v3.py) `--group group1`; **notebook:** [`notebooks/tune_triage_v3.ipynb`](../../notebooks/tune_triage_v3.ipynb) Section 5; **results:** [`docs/results/triage_hpo_group1/`](../results/triage_hpo_group1/).
- **Acuity:** `directions = [maximize exact-accuracy, minimize under-triage, maximize ESI-5 recall]` over the boost vector (`boost_esi1/2/5`, ESI 3/4 pinned at 1.0), `learning_rate`, `n_estimators`. Reports the Pareto frontier; the incumbent is enqueued as trial 0 and sits on the curve.
- **Disposition:** `directions = [maximize accuracy, minimize under-triage]` over `scale_pos_weight_exponent`, `learning_rate`, `n_estimators`, and the `decision_threshold` (a Pareto dimension here since triage has no prior threshold sweep). Cascades `predicted_acuity` from the incumbent acuity model so it isolates the disposition levers.
- **Chaining:** after a deployed Pareto point is confirmed (the `selected` block in `tuned_params_triage_g1.json`), the Group-2 sweep can be re-run on top of it with `--group group2 --use-group1-best`.
- **Status:** RUN PENDING (Colab); results table filled in `docs/results/triage_hpo_group1/`.

### Tried and reverted

- **Iteration 3 — section 1.5, hand-curated red-flag keyword features (2026-05-28, reverted).** A 44-column block of `rf_<name>` binary flags across 9 ED red-flag categories (cardiac, neuro, respiratory, trauma, sepsis, hemorrhage, OB/GYN, anaphylaxis, overdose/self-harm) was added on top of iter 2. Result on the same 83,617-row test split: every headline metric within ±0.04pp noise floor; **under-triage rate +0.02pp** (slightly worse — the metric we kept iter 2 for); ESI 1-5 per-class recall all flat within ≤+0.9pp on 220 samples (noise). The features did engage — **31/44 acuity red flags + 26/44 disposition red flags had non-zero gain, with `rf_palpitations` at rank 13 on disposition** (higher than 3 of 5 top PMH features) — but the signal was already captured by TF-IDF n-grams + iter 2's `ESI_EXTREME_BOOST` weighting, so the gain didn't translate to headline movement. Same structural diagnosis as the doctor v3 Bio_ClinicalBERT experiment: short ED chief-complaint text doesn't carry enough information for hand-curated keyword overlays to beat what TF-IDF on 1-3 grams already extracts. Reverted in commit `cc348e6`; full audit + per-feature ranks in [`docs/future-work.md` entry 6 of "Empirical findings — experiments tried and reverted"](../future-work.md#6-triage-v3-section-15--hand-curated-red-flag-keyword-features--reverted-lift-within-noise-floor).
