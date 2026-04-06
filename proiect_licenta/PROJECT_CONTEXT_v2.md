# Multi-Agent Medical Decision Support System — Project Context (v2)

## Overview

This is a **Bachelor's Thesis** project implementing a **Multi-Agent Architecture for Medical Decision Support** using **CrewAI** and **supervised machine learning** trained on the **MIMIC-IV Emergency Department** dataset (~425K real patient encounters).

The system simulates a clinical emergency department workflow: a patient describes their symptoms in natural language, and a pipeline of specialized AI agents processes the input through sequential steps of triage, diagnosis, and hospital disposition.

**Tech stack:** Python 3.13, CrewAI 1.9.3, Gemini 2.5 Flash (LLM), XGBoost, scikit-learn, PyTorch, sentence-transformers (ClinicalBERT), pandas.

---

## System Architecture

```
Patient (free text) → NLP Parser Agent → Triage Agent → Doctor Agent → [Nurse Agent]
                         (LLM)            (ML model)      (ML model)     (future)
```

### Full Agent Pipeline (4 agents total)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT MEDICAL SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. NLP Parser Agent (LLM - Gemini)                                 │
│     Input:  Free-text patient description                           │
│     Output: JSON { chief_complaints[], pain_score, age,             │
│                    gender, arrival_transport }                      │
│                                                                     │
│  2. Triage Agent (ML - XGBoost + TF-IDF)                            │
│     Input:  Structured complaints + pain + demographics             │
│     Output: ESI acuity level (1-5), admission/discharge prediction  │
│     Tool:   triage_prediction_tool (wraps trained ML models)        │
│                                                                     │
│  3. Doctor Agent (ML - XGBoost + ClinicalBERT)                      │
│     Input:  Triage outputs + patient demographics                   │
│     Output: Primary Diagnosis, department prediction (for admitted) │
│     Tool:   doctor_prediction_tool (wraps trained ML + BERT)        │
│                                                                     │
│  4. Nurse/Assistant Agent (FUTURE - not yet implemented)            │
│     Input:  Requests from Doctor Agent                              │
│     Output: Vital signs, patient history, clinical parameters       │
│     Data:   vitalsign.csv, medrecon.csv from MIMIC-IV               │
│                                                                     │
│  5. Text Generation Agent (FUTURE - not yet implemented)            │
│     Purpose: Generate synthetic patient utterances from structured  │
│              data for testing and ML model validation               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What's Been Accomplished 

### 1. NLP Parser Agent
- **Type:** LLM-based (Gemini via CrewAI)
- **What it does:** Extracts structured vectors directly from free-text conversations.

### 2. Triage Agent (Phase 1)
- **Type:** CrewAI agent with `TriagePredictionTool` wrapping two XGBoost models.
- **Goal:** Predict ESI acuity (1-5) and admission/discharge.
- **Engine details:** Relies on a 2000-dimensional **TF-IDF array** to tokenize symptom unigrams/bigrams alongside structured demographic markers. Utilizes soft class weighting.
- **Performance:** **66.7% Acuity Accuracy**, **75.9% Disposition Accuracy**.

### 3. Doctor Agent (Phase 2)
- **Type:** CrewAI agent with `DoctorPredictionTool` utilizing PyTorch-accelerated Deep Learning Embeddings + XGBoost.
- **Goal:** Formulate a 19-class structured Clinical Diagnosis, and an overarching exact prediction of 16 potential Hospital Departments (SURG, MED, NEURO, etc.).
- **Engine Details:** 
  - Instead of TF-IDF word-matching, it projects the patient's inputs using **Sentence Transformers (`pritamdeka/S-PubMedBert-MS-MARCO`)**, creating a 768-dimensional mathematical clinical understanding of the texts.
  - Injects inverse-frequency square-root class weighting.
  - Cascades Triage assumptions (`acuity` & `disposition`) deeply into the Doctor model's features.
- **Performance (100k Benchmark Dataset):**
  - **Diagnosis Accuracy:** **57.28%** (Excellent for 19 classes without Vitals) with specialized pockets like *Injury and Poisoning* reaching 72.5% Precision.
  - **Service Accuracy:** **58.37%** with remarkably structured minority rescue flags (e.g. Surgery recall improved from 10% to 48%, Cardiology recall improved from 44% to 65% because the algorithms actively monitor high-risk dense semantic vectors).


## Trained ML Models (stored in `src/proiect_licenta/models/`)

- `acuity_model.joblib` — Triage ESI predictor (XGBoost).
- `disposition_model.joblib` — Triage ADMIT predictor (XGBoost).
- `diagnosis_model.joblib` — Doctor Diagnosis Categorical predictor (XGBoost).
- `service_model.joblib` — Doctor Hospital admission department predictor (XGBoost).
- `doctor_bert_name.joblib` — Contains the cached string referencing the HuggingFace transformer path.
- `tfidf_vectorizer.joblib` / `severity_map.joblib` — Core Phase 1 data artifacts.


---

## Project Structure

```
proiect_licenta/
├── pyproject.toml                # Dependencies: crewai, pandas, xgboost, thefuzz, joblib, torch, sentence-transformers
├── src/proiect_licenta/
│   ├── main.py                   # Entry point — interactive patient input 
│   ├── crew.py                   # CrewAI crew definition — 3 agents, 3 tasks, sequential process
│   ├── data_pipeline.py          # Phase 1 Triage ML script
│   ├── doctor_data_pipeline.py   # Phase 2 Doctor ML script (Downsampled 100k Benchmark via ClinicalBERT)
│   ├── tools/
│   │   ├── triage_tool.py        # Triage inference engine
│   │   └── doctor_tool.py        # Doctor inference engine (Instantiates torch models lazily)
...
```

---

## Design Decisions

1. **No vital signs at triage stage** — By design, the Triage Agent uses only complaints + pain + demographics. Vital signs are deferred to the Nurse/Assistant Agent, which the Doctor Agent can request. This simulates the real-world flow where detailed vitals come after initial triage.

2. **Semantic Divergence** — The Triage Agent actively utilizes simple TF-IDF and normalized term lists to keep inference lightning fast. The Doctor Agent uses massive Deep Learning transformers (ClinicalBERT framework) to detect advanced clinical context clues that standard keyword lexicons drop.

3. **Multi-Model Sequential Cascading** — Triage computes Acuity. Acuity is passed to Admissions. Both Triage signals are concatenated with BERT vectors and passed directly into the Doctor system so the algorithm natively "listens" to the Triage Agent exactly like a human Doctor would.

4. **Class Correction Constraints** — XGBoost inherently drifts toward "Major General Medicine", thereby causing high global accuracy but mismanaging life-or-death surgical categories. We imposed math-heavy class-balancing logic via square root counts directly on the `sample_weight` fields.

---

## How to Run

```bash
# Execute Triage Pipeline Training
uv run python src/proiect_licenta/data_pipeline.py

# Execute Doctor Pipeline Training (Requires PyTorch + Transformers)
uv run python src/proiect_licenta/doctor_data_pipeline.py

# Run the live interactive medical system (NLP -> Triage -> Doctor)
uv run run_crew
```

---

## Future Goals (Next Implementations)

### Immediate Doctor Pipeline Optimization
- Utilize **Optuna** to cycle hundreds of hyperparameters (`max_depth`, `learning_rate` constraints) against the ClinicalBERT models to extract optimal F1 values.
- Uncap the 100,000 benchmark limit and evaluate over the ultimate 425K row ecosystem. 

### Phase 3: Nurse/Assistant Agent
- Provides supplementary patient information on request from the Doctor Agent.
- Has access to: vital signs (`vitalsign.csv`), medication history (`medrecon.csv`), clinical parameters.
- Simulates a real nurse taking vitals and reporting to the doctor.
- **Impact:** When vital signs (SpO2, SBP/DBP, Heart Rate) are injected into the XGBoost layers, classification accuracy bounds across all three AI pathways will scale exponentially.

### Phase 4: Text Generation Agent
- Generates synthetic natural language patient descriptions from structured data.
- Foundational element for system validation protocols.
