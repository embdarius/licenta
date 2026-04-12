# Multi-Agent Medical Decision Support System — Project Context

## Overview

This is a **Bachelor's Thesis** project implementing a **Multi-Agent Architecture for Medical Decision Support** using **CrewAI** and **supervised machine learning** trained on the **MIMIC-IV Emergency Department** dataset (~425K real patient encounters).

The system simulates a clinical emergency department workflow: a patient describes their symptoms in natural language, and a pipeline of specialized AI agents processes the input through triage, diagnosis, and disposition.

**Tech stack:** Python 3.13, CrewAI 1.9.3, Gemini 2.5 Flash (LLM), XGBoost, scikit-learn, pandas, thefuzz.

---

## System Architecture

```
Patient (free text) -> NLP Parser Agent -> Triage Agent -> Doctor Agent -> [Nurse Agent]
                         (LLM)             (ML model)      (ML model)      (future)
```

### Full Agent Pipeline (currently 3 operational, 2 planned)

```
+---------------------------------------------------------------------+
|                  MULTI-AGENT MEDICAL SYSTEM                          |
+---------------------------------------------------------------------+
|                                                                      |
|  1. NLP Parser Agent (LLM - Gemini)               [OPERATIONAL]     |
|     Input:  Free-text patient description                            |
|     Output: JSON { chief_complaints[], pain_score, age,              |
|                    gender, arrival_transport }                        |
|     Tool:   ask_patient (interactive follow-up questions)            |
|                                                                      |
|  2. Triage Agent (ML - XGBoost)                    [OPERATIONAL]     |
|     Input:  Structured complaints + pain + demographics              |
|     Output: ESI acuity level (1-5), admission/discharge prediction   |
|     Tool:   triage_prediction_tool (wraps 2 XGBoost models)         |
|                                                                      |
|  3. Doctor Agent (ML - XGBoost)                    [OPERATIONAL]     |
|     Input:  Triage results + patient data                            |
|     Output: Diagnosis category (14 classes),                         |
|             hospital department (11 classes, admitted only)           |
|     Tool:   doctor_prediction_tool (wraps 2 XGBoost models)         |
|                                                                      |
|  4. Nurse/Assistant Agent (FUTURE - not yet implemented)             |
|     Input:  Requests from Doctor Agent                               |
|     Output: Vital signs, patient history, clinical parameters        |
|     Data:   vitalsign.csv, medrecon.csv from MIMIC-IV               |
|                                                                      |
|  5. Text Generation Agent (FUTURE - not yet implemented)             |
|     Purpose: Generate synthetic patient utterances from structured    |
|              data for testing and ML model validation                 |
|                                                                      |
+---------------------------------------------------------------------+
```

---

## Phase 1: Triage System (Complete)

### NLP Parser Agent
- **Type:** LLM-based (Gemini via CrewAI)
- **What it does:** Takes free-text patient input like *"I'm a 65 year old man, I have severe chest pain and difficulty breathing, I came by ambulance"* and outputs structured JSON:
  ```json
  {
    "chief_complaints": ["chest pain", "dyspnea"],
    "pain_score": 8,
    "age": 65,
    "gender": "male",
    "arrival_transport": "ambulance"
  }
  ```
- **Follow-up:** Uses `AskPatientTool` to interactively ask for missing information (pain score, age, gender, arrival method)
- **Config:** `src/proiect_licenta/config/agents.yaml` (nlp_parser) and `tasks.yaml` (parse_symptoms_task)

### Triage Agent + ML Models
- **Type:** CrewAI agent with a custom tool (`TriagePredictionTool`) wrapping two XGBoost models
- **What it does:** Predicts ESI acuity (1-5) and admission/discharge
- **Config:** `agents.yaml` (triage_agent) and `tasks.yaml` (triage_assessment_task)

### Triage ML Models (stored in `src/proiect_licenta/models/`)

#### Model 1: Acuity Prediction (ESI 1-5)
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
- **Class weighting:** Soft balanced (sqrt of inverse frequency)
- **Note:** Model outputs classes 0-4 internally; add 1 to get ESI 1-5

#### Model 2: Disposition Prediction (Admit vs. Discharge)
- **File:** `disposition_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02)
- **Features:** Same as acuity model PLUS `predicted_acuity` (cascading: acuity is predicted first, then used as input for disposition)

#### Supporting Artifacts
- `tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer (2000 features, trigrams)
- `severity_map.joblib` — Dict mapping complaint words to their mean acuity (819 words)
- `model_metadata.json` — Training date, accuracy metrics, feature list, model params

### Triage Benchmark Results (test set: 83,617 samples)

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

**Per-class acuity recall:** ESI 1: 49.8%, ESI 2: 63.5%, ESI 3: 71.5%, ESI 4: 60.4%, ESI 5: 22.7%

**Key findings:**
- Model is strongest on ESI 3 (majority class, 54%) and weakest on ESI 1/5 (minority classes)
- 97.7% within-1 accuracy means catastrophic misclassifications are very rare
- Slight lean toward under-triage (55% of errors) vs over-triage (45%)
- Top features: severity priors, pain_missing, complaint-specific TF-IDF terms

### Triage Training Pipeline Evolution

| Version | Changes | Acuity | Disposition |
|---------|---------|--------|-------------|
| v1 | RandomForest + multi-hot top-100 complaints + pain only | 32.1% | 67.2% |
| v2 | XGBoost + TF-IDF 500 + severity priors + abbreviation expansion | 53.4% | 72.4% |
| v3 | + age, gender, arrival_transport from edstays/patients tables | 58.4% | 75.6% |
| v3b | + TF-IDF 2000, 3000 trees, soft class weights, interaction features | **66.7%** | **75.9%** |

---

## Phase 2: Doctor Agent (Complete)

### Doctor Agent
- **Type:** CrewAI agent with `DoctorPredictionTool` wrapping two XGBoost models
- **What it does:** For admitted patients, predicts diagnosis category (14 classes) and hospital department (11 classes). For discharged patients, provides discharge summary.
- **Config:** `agents.yaml` (doctor_agent) and `tasks.yaml` (doctor_assessment_task)
- **Data flow:** Receives triage results via structured data block from triage agent

### Doctor ML Models (stored in `src/proiect_licenta/models/doctor/`)

#### Model 3: Diagnosis Category Prediction (14 classes)
- **File:** `diagnosis_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02, early stopping at best_iteration=1413)
- **Training data:** 80K samples (80/20 split from 100K sampled from 157K admitted patients)
- **Features (2025 total):** Same 2023 triage features PLUS `predicted_acuity` and `predicted_disposition` from the triage model (cascading)
- **Class weighting:** Soft balanced (sqrt of inverse frequency)
- **Target:** Primary diagnosis category (seq_num=1 from ICD codes, grouped from 22 raw categories into 14)

**Diagnosis Category Grouping (22 raw -> 14 grouped):**

| # | Grouped Category | Raw Categories Merged | Admitted Count | % |
|---|------------------|-----------------------|----------------|---|
| 1 | Symptoms, Signs, Ill-Defined | Symptoms, Signs, Ill-Defined Conditions | 50,705 | 33.2% |
| 2 | Circulatory | Circulatory System | 18,655 | 12.2% |
| 3 | Digestive | Digestive System | 18,060 | 11.8% |
| 4 | Injury and Poisoning | Injury and Poisoning | 14,321 | 9.4% |
| 5 | Respiratory | Respiratory System | 11,056 | 7.2% |
| 6 | Genitourinary | Genitourinary System | 8,770 | 5.7% |
| 7 | Musculoskeletal | Musculoskeletal System | 5,974 | 3.9% |
| 8 | Endocrine, Nutritional, Metabolic | Endocrine, Nutritional, Metabolic | 5,274 | 3.5% |
| 9 | Mental Disorders | Mental Disorders | 4,494 | 2.9% |
| 10 | Skin and Subcutaneous Tissue | Skin and Subcutaneous Tissue | 4,248 | 2.8% |
| 11 | Nervous System and Sense Organs | Nervous System + Nervous System and Sense Organs + Diseases of the Eye + Diseases of the Ear (ICD-9/10 artifact) | 3,975 | 2.6% |
| 12 | Blood and Blood-Forming Organs | Blood and Blood-Forming Organs | 3,163 | 2.1% |
| 13 | Infectious and Parasitic | Infectious and Parasitic Diseases | 2,559 | 1.7% |
| 14 | Other | Pregnancy/Childbirth (789) + Neoplasms (694) + Supplemental (117) + Congenital (26) + Invalid (3) + Perinatal (2) | 1,551 | 1.0% |

#### Model 4: Department Prediction (11 classes)
- **File:** `department_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02, early stopping at best_iteration=1927)
- **Features:** Same 2025 features PLUS `predicted_diagnosis` from the diagnosis model (cascading: triage -> diagnosis -> department)
- **Target:** First hospital service on admission (from `services.csv`)

**Department Grouping (19 raw -> 11 grouped):**

| # | Code | Full Name | Raw Services Merged | Count | % |
|---|------|-----------|---------------------|-------|---|
| 1 | MED | General Medicine | MED | 91,049 | 59.6% |
| 2 | CMED | Cardiac Medicine | CMED | 13,588 | 8.9% |
| 3 | SURG | General Surgery | SURG | 10,808 | 7.1% |
| 4 | NMED | Neuro Medicine | NMED | 10,598 | 6.9% |
| 5 | OMED | Oncology Medicine | OMED | 8,909 | 5.8% |
| 6 | ORTHO | Orthopedics | ORTHO | 5,300 | 3.5% |
| 7 | NSURG | Neurosurgery | NSURG | 3,804 | 2.5% |
| 8 | OTHER_SURG | Other Surgery | VSURG (1,770) + TSURG (568) + CSURG (537) + PSURG (443) | 3,257 | 2.1% |
| 9 | TRAUM | Trauma | TRAUM | 2,844 | 1.9% |
| 10 | OB_GYN | Obstetrics / Gynecology | GYN (1,078) + OBS (438) | 1,504 | 1.0% |
| 11 | OTHER | Other Specialty | GU (840) + PSYCH (210) + ENT (92) + EYE (6) + DENT (4) | 1,144 | 0.7% |

#### Supporting Artifacts
- `doctor_metadata.json` — Training date, accuracy metrics, label lists, department names

### Doctor Benchmark Results (test set: 20,000 samples)

#### Diagnosis Category Model

| Metric | Value |
|--------|-------|
| **Top-1 accuracy (exact)** | **50.2%** |
| **Top-3 accuracy** | **83.6%** |
| **Top-5 accuracy** | **91.2%** |
| Cohen's kappa | 0.41 (moderate agreement) |
| Accuracy excl. catch-all | 52.6% |
| Majority baseline | 33.2% |
| **Lift over random** | **3.05x** |
| Mean confidence (correct) | 0.56 |
| Mean confidence (incorrect) | 0.45 |

**Per-class diagnosis recall (3 tiers):**
- **Strong (>65%):** Mental Disorders (75.5%), Injury/Poisoning (69.2%), Digestive (66.5%)
- **Moderate (45-58%):** Respiratory (57.8%), Skin (55.6%), Musculoskeletal (52.3%), Nervous System (47.5%), Circulatory (46.0%), Symptoms/Ill-Defined (45.4%)
- **Weak (<36%):** Blood (36.2%), Endocrine (31.9%), Genitourinary (29.7%), Other (20.7%), Infectious (11.0%)

**Key findings:**
- The "Symptoms, Signs, Ill-Defined" category (33% of data) is a catch-all where ICD codes essentially restate the chief complaint as a diagnosis (e.g., "chest pain NOS"). It absorbs 20-28% of every other category's misclassifications.
- Strong categories have **distinctive complaint language** (e.g., "deliberate self harm" -> Mental, "collision/fracture" -> Injury)
- Weak categories have **non-specific symptoms** (e.g., "weakness/fever" could be Infectious, Endocrine, Blood, or anything)
- Top features: "deliberate self" (Mental), "collision" (Injury), "pregnant" (Other/OB), "sickle cell" (Blood)

**Top misclassification patterns:**
1. Symptoms -> Digestive (16% of Symptoms cases)
2. Circulatory -> Symptoms (23% of Circulatory cases)
3. Genitourinary -> Symptoms (28% of Genitourinary cases)
4. Respiratory -> Symptoms (20% of Respiratory cases)

#### Department Model

| Metric | Value |
|--------|-------|
| **Top-1 accuracy (exact)** | **59.1%** |
| **Top-3 accuracy** | **92.5%** |
| Cohen's kappa | 0.39 |
| Accuracy excl. MED | 50.7% |
| Majority baseline | 59.9% (always predict MED) |
| **Lift over random** | **1.54x** |

**Per-class department recall:**
- **Strong (>65%):** NMED/Neuro (67.1%), CMED/Cardiac (65.8%), ORTHO (65.8%)
- **Moderate (40-55%):** TRAUM (54.9%), SURG (51.6%), OB_GYN (42.3%)
- **Weak (<36%):** NSURG (35.7%), OTHER_SURG (28.1%), OTHER (18.5%), OMED/Oncology (16.0%)

**Key finding:** Overall accuracy (59.1%) is slightly below the majority baseline (59.9%) because MED dominates at 60%. However, the model's value is in correctly routing patients to specialized departments — it successfully identifies cardiac, neuro, ortho, and trauma patients instead of sending everyone to General Medicine.

**Department accuracy by diagnosis category:**
- Best: Mental Disorders -> 86.9% dept accuracy (almost all go to MED)
- Strong: Endocrine (69.6%), Respiratory (67.7%), Infectious (67.2%)
- Weakest: Other (44.3%), Musculoskeletal (49.6%), Injury/Poisoning (51.9%)

---

## Text Preprocessing Pipeline (shared across all models)

Both the training pipelines and inference tools apply identical preprocessing:
1. **Lowercase + strip** the chief complaint text
2. **Replace separators** (`,`, `;`, `/`, `-`, `(`, `)`, `.`) with spaces
3. **Expand abbreviations:** 45+ medical abbreviations mapped (e.g., `abd` -> `abdominal`, `sob` -> `shortness of breath`, `cp` -> `chest pain`, `cva` -> `cerebrovascular accident stroke`, etc.)
4. **TF-IDF vectorization** with the saved vectorizer (2000 features, unigram/bigram/trigram)
5. **Severity priors** computed by looking up each word's mean acuity from training data

---

## Cascading Prediction Architecture

The system uses a cascading design where each model's output feeds into the next:

```
Chief complaints + demographics
         |
         v
  [Acuity Model] -> predicted_acuity (ESI 1-5)
         |
         v
  [Disposition Model] -> predicted_disposition (admit/discharge)
         |                 (uses predicted_acuity as feature)
         v
  [Diagnosis Model] -> predicted_diagnosis (14 categories)
         |                 (uses predicted_acuity + predicted_disposition as features)
         v
  [Department Model] -> predicted_department (11 services)
                           (uses predicted_acuity + predicted_disposition
                            + predicted_diagnosis as features)
```

This mirrors clinical flow: severity determines admission, admission status influences diagnosis coding, and diagnosis determines department routing.

---

## Project Structure

```
proiect_licenta/
|-- .env                          # GEMINI_API_KEY, MODEL config
|-- pyproject.toml                # Dependencies + script entry points
|-- benchmark.py                  # Triage model benchmark script
|-- benchmark_doctor.py           # Doctor model benchmark script
|-- src/proiect_licenta/
|   |-- main.py                   # Entry point -- interactive patient input -> crew kickoff
|   |-- crew.py                   # CrewAI crew definition -- 3 agents, 3 tasks, sequential
|   |-- data_pipeline.py          # Triage ML training pipeline (acuity + disposition)
|   |-- doctor_data_pipeline.py   # Doctor ML training pipeline (diagnosis + department)
|   |-- config/
|   |   |-- agents.yaml           # Agent definitions (nlp_parser, triage_agent, doctor_agent)
|   |   +-- tasks.yaml            # Task definitions (parse, triage, doctor assessment)
|   |-- tools/
|   |   |-- __init__.py           # Exports all tools
|   |   |-- triage_tool.py        # CrewAI BaseTool wrapping triage ML models
|   |   |-- doctor_tool.py        # CrewAI BaseTool wrapping doctor ML models
|   |   |-- ask_patient_tool.py   # Interactive follow-up question tool
|   |   +-- custom_tool.py        # (unused template file)
|   |-- models/                   # Triage model artifacts (.joblib)
|   |   |-- acuity_model.joblib
|   |   |-- disposition_model.joblib
|   |   |-- tfidf_vectorizer.joblib
|   |   |-- severity_map.joblib
|   |   |-- model_metadata.json
|   |   +-- doctor/               # Doctor model artifacts
|   |       |-- diagnosis_model.joblib
|   |       |-- department_model.joblib
|   |       +-- doctor_metadata.json
|   +-- datasets/datasets_mimic-iv/
|       |-- mimic-iv-ed/          # Emergency Department data (PRIMARY)
|       |   |-- triage.csv        # 425K rows: complaints, pain, vitals, acuity
|       |   |-- edstays.csv       # 425K rows: stay tracking, gender, arrival, disposition
|       |   |-- diagnosis.csv     # 900K rows: ICD-9/10 diagnosis codes per stay
|       |   |-- vitalsign.csv     # 1.4M rows: longitudinal vital signs during stay
|       |   |-- medrecon.csv      # Medication reconciliation (pre-admission meds)
|       |   |-- pyxis.csv         # Medications dispensed during stay
|       |   +-- files_created/
|       |       +-- categorized_diagnosis.csv  # ICD codes grouped into categories
|       |-- mimic-iv/hosp/        # Hospital data
|       |   |-- patients.csv      # Demographics (age, gender, death date)
|       |   |-- admissions.csv    # Hospital admissions
|       |   |-- services.csv      # Department assignments (MED, SURG, NEURO, etc.)
|       |   |-- diagnoses_icd.csv # Hospital-wide ICD diagnosis codes
|       |   +-- d_icd_diagnoses.csv # ICD code dictionary
|       +-- mimic-iv/icu/         # ICU data (NOT USED)
```

---

## Key MIMIC-IV Dataset Details

### triage.csv (PRIMARY -- used for triage training)
Columns: `subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint`
- `chiefcomplaint`: Free-text, comma-separated (e.g., "Abd pain, Abdominal distention")
- `pain`: 0-10 patient-reported pain score
- `acuity`: ESI 1-5 (1=most severe, 5=least severe)
- Vital signs (temperature, heartrate, etc.): Available but NOT used in current models by design

### edstays.csv
Columns: `subject_id, hadm_id, stay_id, intime, outtime, gender, race, arrival_transport, disposition`
- `disposition`: HOME, ADMITTED, LEFT WITHOUT BEING SEEN, etc.
- `arrival_transport`: AMBULANCE (36%), WALK IN (60%), HELICOPTER (0.1%), OTHER, UNKNOWN
- `hadm_id`: Links to services.csv for department assignment

### patients.csv
Columns: `subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod`
- Age computed as: `anchor_age + (visit_year - anchor_year)`

### diagnosis.csv + categorized_diagnosis.csv
- ICD-9/10 codes per ED stay with categories
- `seq_num=1` is the primary diagnosis (used for training)
- Grouping via `python/group_icd.py` maps ICD codes to broad medical categories
- 22 raw categories grouped into 14 for the doctor model

### services.csv
- Maps `hadm_id` to hospital service/department
- 19 unique services grouped into 11 for the doctor model
- First service per admission used (initial department assignment)

---

## Design Decisions

1. **No vital signs at triage/doctor stage** -- By design, the current models use only complaints + pain + demographics. Vital signs are deferred to the Nurse/Assistant Agent. This allows benchmarking "before Nurse" vs "after Nurse" to measure the impact of additional clinical data.

2. **LLM for NLP parsing, ML for prediction** -- The NLP Parser uses Gemini (LLM) for natural language understanding. All prediction models use supervised ML (XGBoost) trained on 400K+ real encounters for reliable, auditable predictions.

3. **Cascading prediction** -- Each model's output feeds into the next: acuity -> disposition -> diagnosis -> department. This mirrors clinical flow and improves downstream predictions.

4. **Soft class weighting** -- Uses sqrt(inverse_frequency) to balance minority class recall without destroying majority class accuracy.

5. **Category grouping** -- Small diagnosis/department categories are merged to ensure sufficient training samples per class. The "Nervous System" ICD-9/10 split is an artifact that was corrected by merging.

6. **100K training cap for doctor models** -- To keep training times reasonable while benchmarking. Can be increased to full 157K for production.

7. **Admitted-only for doctor models** -- Diagnosis and department prediction only applies to admitted patients. Discharged patients get a discharge summary instead.

8. **No ICU data** -- Project focuses entirely on the Emergency Department pathway.

---

## How to Run

```bash
# Install dependencies
uv sync

# Train/retrain triage ML models (~90 minutes)
uv run train_models

# Train/retrain doctor ML models (~30 minutes)
uv run train_doctor

# Run the full 3-agent system interactively
uv run crewai run

# Or equivalently
uv run run_crew

# Run benchmarks
uv run python benchmark.py          # Triage models
uv run python benchmark_doctor.py   # Doctor models
```

### Environment Variables (`.env`)
```
MODEL=gemini/gemini-2.5-flash
GEMINI_API_KEY=<key>
```

---

## Future Goals

### Phase 3: Nurse/Assistant Agent
- Provides supplementary patient information on request from the Doctor Agent
- Has access to: vital signs (`vitalsign.csv`), medication history (`medrecon.csv`), clinical parameters
- Simulates a real nurse taking vitals and reporting to the doctor
- **Key benchmark:** Compare diagnosis/department accuracy before and after adding vital signs
- Expected improvement areas:
  - Circulatory vs Respiratory differentiation (heart rate, BP, O2 sat)
  - Infectious disease detection (fever/temperature is a strong signal currently missing)
  - Endocrine detection (glucose levels, vital abnormalities)
  - OMED/Oncology detection (likely needs medication history from medrecon.csv)

### Phase 4: Text Generation Agent
- Generates synthetic natural language patient descriptions from structured data
- E.g., structured `"abdominal pain, pain=7"` -> *"I've been having this terrible stomach ache since this morning, it's really bad"*
- Used for creating test cases, validating the system, and training ML models

### Phase 5: Hospital Infrastructure
- Synthetic real-time database of hospital rooms and available beds
- Admission routing based on department prediction and bed availability
- Possibly predict length of stay

### Model Improvement Opportunities

**Triage models:**
- Add more trees (models still weren't converging at 3000 -- best_iteration near 2990)
- Include triage vital signs (from triage.csv) -- these ARE taken at triage time and would significantly boost accuracy
- Try LightGBM as an alternative to XGBoost
- Ensemble methods (stacking multiple models)
- Neural network with learned embeddings for complaint text

**Doctor models:**
- Train on full 157K admitted rows instead of 100K
- Add second/third diagnoses (seq_num 2, 3) as multi-label training signal
- Hierarchical approach: first classify "Symptoms/Ill-Defined vs real diagnosis", then predict specific category
- Add vital signs from Nurse Agent (expected to significantly improve Infectious, Endocrine, Circulatory detection)
- Add medication history from Nurse Agent (expected to help OMED/Oncology detection)
- The "Symptoms, Signs, Ill-Defined" catch-all (33%) is a labeling issue, not a model issue -- patients coded with symptom-level ICD codes vs disease-level codes may have identical presentations

---

## Known Issues

1. **Windows console Unicode:** CrewAI's internal logging uses emojis that cause `charmap` encoding errors on Windows cp1252 console. These are cosmetic warnings from CrewAI's event bus, not from our code. They don't affect functionality.

2. **Windows Application Control:** Pandas DLL loading can occasionally be blocked by Windows Application Control policies. Restarting the terminal or IDE usually resolves this.

3. **Triage training time:** With 3000 trees and 2023 features on 334K samples, training takes ~90 minutes.

4. **Doctor training time:** With 3000 trees and 2025 features on 80K samples, training takes ~30 minutes (two models). Early stopping typically triggers around iteration 1400-1900.

5. **Doctor model accuracy ceiling:** The 50% diagnosis accuracy and 59% department accuracy are fundamentally limited by the available features (chief complaints + demographics only). Adding vital signs, labs, and medication history from the Nurse Agent is expected to significantly improve these numbers.
