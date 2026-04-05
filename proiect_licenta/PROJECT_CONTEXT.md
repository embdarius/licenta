# Multi-Agent Medical Decision Support System — Project Context

## Overview

This is a **Bachelor's Thesis** project implementing a **Multi-Agent Architecture for Medical Decision Support** using **CrewAI** and **supervised machine learning** trained on the **MIMIC-IV Emergency Department** dataset (~425K real patient encounters).

The system simulates a clinical emergency department workflow: a patient describes their symptoms in natural language, and a pipeline of specialized AI agents processes the input through triage, diagnosis, and disposition.

**Tech stack:** Python 3.13, CrewAI 1.9.3, Gemini 2.5 Flash (LLM), XGBoost, scikit-learn, pandas, thefuzz.

---

## System Architecture

```
Patient (free text) → NLP Parser Agent → Triage Agent → [Doctor Agent] → [Nurse Agent]
                         (LLM)            (ML model)      (future)        (future)
```

### Full Agent Pipeline (4 agents total)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT MEDICAL SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. NLP Parser Agent (LLM - Gemini)                                │
│     Input:  Free-text patient description                          │
│     Output: JSON { chief_complaints[], pain_score, age,            │
│                    gender, arrival_transport }                      │
│                                                                     │
│  2. Triage Agent (ML - XGBoost)                                    │
│     Input:  Structured complaints + pain + demographics            │
│     Output: ESI acuity level (1-5), admission/discharge prediction │
│     Tool:   triage_prediction_tool (wraps trained ML models)       │
│                                                                     │
│  3. Doctor Agent (FUTURE - not yet implemented)                    │
│     Input:  Triage results + patient data                          │
│     Output: Diagnosis, department prediction (for admitted)        │
│     Can request vitals from Nurse Agent                            │
│                                                                     │
│  4. Nurse/Assistant Agent (FUTURE - not yet implemented)           │
│     Input:  Requests from Doctor Agent                             │
│     Output: Vital signs, patient history, clinical parameters      │
│     Data:   vitalsign.csv, medrecon.csv from MIMIC-IV              │
│                                                                     │
│  5. Text Generation Agent (FUTURE - not yet implemented)           │
│     Purpose: Generate synthetic patient utterances from structured  │
│              data for testing and ML model validation               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What's Been Accomplished (Phase 1)

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
- **Config:** `src/proiect_licenta/config/agents.yaml` (nlp_parser) and `tasks.yaml` (parse_symptoms_task)

### Triage Agent + ML Models
- **Type:** CrewAI agent with a custom tool (`TriagePredictionTool`) wrapping two XGBoost models
- **What it does:** Predicts ESI acuity (1-5) and admission/discharge
- **Config:** `agents.yaml` (triage_agent) and `tasks.yaml` (triage_assessment_task)

### Trained ML Models (stored in `src/proiect_licenta/models/`)

#### Model 1: Acuity Prediction (ESI 1-5)
- **File:** `acuity_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02)
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
- **Class weighting:** Soft balanced (sqrt of inverse frequency) — balances minority class recall without destroying majority class accuracy
- **Performance:**
  - Exact accuracy: **66.7%**
  - Within-1-level accuracy: **97.7%** (prediction off by at most 1 ESI level)
- **Note:** Model outputs classes 0-4 internally; add 1 to get ESI 1-5

#### Model 2: Disposition Prediction (Admit vs. Discharge)
- **File:** `disposition_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02)
- **Features:** Same as acuity model PLUS `predicted_acuity` (cascading: acuity is predicted first, then used as input for disposition)
- **Performance:** **75.9% accuracy**

#### Supporting Artifacts
- `tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer (2000 features, trigrams)
- `severity_map.joblib` — Dict mapping complaint words to their mean acuity (819 words)
- `model_metadata.json` — Training date, accuracy metrics, feature list, model params

### Text Preprocessing Pipeline
Both the training pipeline and the inference tool apply identical preprocessing:
1. **Lowercase + strip** the chief complaint text
2. **Replace separators** (`,`, `;`, `/`, `-`, `(`, `)`, `.`) with spaces
3. **Expand abbreviations:** 45+ medical abbreviations mapped (e.g., `abd` → `abdominal`, `sob` → `shortness of breath`, `cp` → `chest pain`, `cva` → `cerebrovascular accident stroke`, etc.)
4. **TF-IDF vectorization** with the saved vectorizer
5. **Severity priors** computed by looking up each word's mean acuity from training data

### Training Pipeline Evolution (accuracy progression)

| Version | Changes | Acuity | Disposition |
|---------|---------|--------|-------------|
| v1 | RandomForest + multi-hot top-100 complaints + pain only | 32.1% | 67.2% |
| v2 | XGBoost + TF-IDF 500 + severity priors + abbreviation expansion | 53.4% | 72.4% |
| v3 | + age, gender, arrival_transport from edstays/patients tables | 58.4% | 75.6% |
| v3b | + TF-IDF 2000, 3000 trees, soft class weights, interaction features | **66.7%** | **75.9%** |

---

## Project Structure

```
proiect_licenta/
├── .env                          # GEMINI_API_KEY, MODEL config
├── pyproject.toml                # Dependencies: crewai, pandas, scikit-learn, xgboost, thefuzz, joblib
├── src/proiect_licenta/
│   ├── main.py                   # Entry point — interactive patient input → crew kickoff
│   ├── crew.py                   # CrewAI crew definition — 2 agents, 2 tasks, sequential process
│   ├── data_pipeline.py          # ML training pipeline — load MIMIC-IV, feature engineer, train XGBoost
│   ├── config/
│   │   ├── agents.yaml           # Agent definitions (nlp_parser, triage_agent)
│   │   └── tasks.yaml            # Task definitions (parse_symptoms_task, triage_assessment_task)
│   ├── tools/
│   │   ├── __init__.py           # Exports TriagePredictionTool
│   │   ├── triage_tool.py        # CrewAI BaseTool wrapping both ML models
│   │   └── custom_tool.py        # (unused template file)
│   ├── models/                   # Trained model artifacts (.joblib)
│   │   ├── acuity_model.joblib
│   │   ├── disposition_model.joblib
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── severity_map.joblib
│   │   └── model_metadata.json
│   └── datasets/datasets_mimic-iv/
│       ├── mimic-iv-ed/          # Emergency Department data (PRIMARY)
│       │   ├── triage.csv        # 425K rows: complaints, pain, vitals, acuity (MAIN TRAINING DATA)
│       │   ├── edstays.csv       # 425K rows: stay tracking, gender, arrival, disposition
│       │   ├── diagnosis.csv     # 900K rows: ICD-9/10 diagnosis codes per stay
│       │   ├── vitalsign.csv     # 1.4M rows: longitudinal vital signs during stay
│       │   ├── medrecon.csv      # Medication reconciliation (pre-admission meds)
│       │   ├── pyxis.csv         # Medications dispensed during stay
│       │   └── files_created/
│       │       └── categorized_diagnosis.csv  # ICD codes grouped into medical categories
│       ├── mimic-iv/hosp/        # Hospital data
│       │   ├── patients.csv      # Demographics (age, gender, death date)
│       │   ├── admissions.csv    # Hospital admissions
│       │   ├── services.csv      # Department assignments (MED, SURG, NEURO, etc.)
│       │   ├── diagnoses_icd.csv # Hospital-wide ICD diagnosis codes
│       │   └── d_icd_diagnoses.csv # ICD code dictionary
│       └── mimic-iv/icu/         # ICU data (NOT USED — project focuses on ED)
```

---

## Key MIMIC-IV Dataset Details

### triage.csv (PRIMARY — used for training)
Columns: `subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint`
- `chiefcomplaint`: Free-text, comma-separated (e.g., "Abd pain, Abdominal distention")
- `pain`: 0-10 patient-reported pain score
- `acuity`: ESI 1-5 (1=most severe, 5=least severe) — **our primary prediction target**
- Vital signs (temperature, heartrate, etc.): Available but NOT used in current model by design

### edstays.csv
Columns: `subject_id, hadm_id, stay_id, intime, outtime, gender, race, arrival_transport, disposition`
- `disposition`: HOME, ADMITTED, LEFT WITHOUT BEING SEEN, etc. — **used for admit/discharge prediction**
- `arrival_transport`: AMBULANCE (36%), WALK IN (60%), HELICOPTER (0.1%), OTHER, UNKNOWN
- `gender`: M/F — used as feature

### patients.csv
Columns: `subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod`
- Age computed as: `anchor_age + (visit_year - anchor_year)`

### diagnosis.csv + categorized_diagnosis.csv
- ICD-9/10 codes per ED stay with categories (Circulatory, Respiratory, etc.)
- Already has a grouping script (`python/group_icd.py`) that maps ICD codes to broad categories
- **Will be used by the Doctor Agent for department prediction**

### services.csv
- Maps `hadm_id` to hospital service/department (MED, SURG, NEURO, ORTHO, etc.)
- 21 unique services
- **Will be used for department prediction when patient is admitted**

---

## Design Decisions

1. **No vital signs at triage stage** — By design, the Triage Agent uses only complaints + pain + demographics. Vital signs are deferred to the Nurse/Assistant Agent, which the Doctor Agent can request. This simulates the real-world flow where detailed vitals come after initial triage.

2. **LLM for NLP parsing, ML for prediction** — The NLP Parser uses Gemini (LLM) because it needs to understand arbitrary natural language. The Triage Agent uses trained ML models because supervised learning on 400K+ real encounters gives more reliable, auditable predictions than LLM guessing.

3. **Fuzzy matching** — The triage tool uses `thefuzz` for fuzzy matching of patient complaint terms against MIMIC-IV vocabulary (though currently TF-IDF handles this implicitly through word-level encoding).

4. **No ICU data** — Project focuses entirely on the Emergency Department pathway.

5. **Cascading prediction** — Acuity is predicted first, then used as an input feature for disposition prediction (mimics clinical flow where severity determines admission likelihood).

6. **Soft class weighting** — Uses sqrt(inverse_frequency) instead of full inverse to balance minority class recall (ESI 1, 5) without destroying accuracy on majority classes (ESI 2, 3).

---

## How to Run

```bash
# Install dependencies
uv sync

# Train/retrain ML models (takes ~90 minutes with 3000 trees)
uv run python src/proiect_licenta/data_pipeline.py

# Run the triage system interactively
uv run crewai run

# Or equivalently
uv run run_crew
```

### Environment Variables (`.env`)
```
MODEL=gemini/gemini-2.5-flash
GEMINI_API_KEY=<key>
```

---

## Future Goals (Not Yet Implemented)

### Phase 2: Doctor Agent
- Takes triage results (acuity, admit/discharge) as input
- Establishes diagnosis using available patient data
- **Predicts hospital department** for admitted patients (using `services.csv` — MED, SURG, NEURO, etc.)
- Can iteratively request additional information from the Nurse Agent
- Supervised learning on `diagnosis.csv` + `services.csv`

### Phase 3: Nurse/Assistant Agent
- Provides supplementary patient information on request from the Doctor Agent
- Has access to: vital signs (`vitalsign.csv`), medication history (`medrecon.csv`), clinical parameters
- Simulates a real nurse taking vitals and reporting to the doctor
- When vitals are integrated, triage accuracy is expected to improve significantly (literature: +5-15%)

### Phase 4: Text Generation Agent
- Generates synthetic natural language patient descriptions from structured data
- E.g., structured `"abdominal pain, pain=7"` → *"I've been having this terrible stomach ache since this morning, it's really bad"*
- Used for creating test cases, validating the system, and training ML models

### Phase 5: Hospital Infrastructure (Later)
- Synthetic real-time database of hospital rooms and available beds
- Admission routing based on department prediction and bed availability
- Possibly predict length of stay

### Model Improvement Opportunities
- Add more trees (models still weren't converging at 3000 — best_iteration ≈ 2990)
- Include triage vital signs (from triage.csv) — these ARE taken at triage time and would significantly boost accuracy
- Try LightGBM as an alternative to XGBoost
- Ensemble methods (stacking multiple models)
- Neural network with learned embeddings for complaint text

---

## Known Issues

1. **Windows console Unicode:** CrewAI's internal logging uses emojis that cause `charmap` encoding errors on Windows cp1252 console. These are cosmetic warnings from CrewAI's event bus, not from our code. They don't affect functionality.

2. **Windows Application Control:** Pandas DLL loading can occasionally be blocked by Windows Application Control policies. Restarting the terminal or IDE usually resolves this.

3. **Training time:** With 3000 trees and 2023 features on 334K training samples, training takes ~90 minutes. Increasing to 5000 trees would proportionally increase this.
