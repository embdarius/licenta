# System Architecture

This document describes the end-to-end architecture of the Multi-Agent Medical Decision Support System: the agent pipeline, task flow, cascading ML design, shared text preprocessing, and project layout on disk.

For per-agent deep dives (tools, models, benchmarks, training evolution) see:
- [`agents/nlp-parser-agent.md`](agents/nlp-parser-agent.md)
- [`agents/triage-agent.md`](agents/triage-agent.md)
- [`agents/doctor-agent.md`](agents/doctor-agent.md)
- [`agents/nurse-agent.md`](agents/nurse-agent.md)

---

## High-Level Flow

```
Patient -> NLP Parser -> Triage -> Doctor v1 -> Nurse -> Doctor v2
           (LLM)        (ML)      (ML)         (interactive) (ML+vitals+meds)
```

The system simulates a clinical emergency department workflow: a patient describes their symptoms in natural language, and a pipeline of specialized AI agents processes the input through triage, initial diagnosis, nurse data collection, and enhanced reassessment.

---

## Full Agent Pipeline (4 operational agents, 5 tasks)

```
+---------------------------------------------------------------------+
|                  MULTI-AGENT MEDICAL SYSTEM                          |
+---------------------------------------------------------------------+
|                                                                      |
|  1. NLP Parser Agent (LLM - Gemini)               [OPERATIONAL]     |
|     Task:   parse_symptoms_task                                      |
|     Input:  Free-text patient description                            |
|     Output: JSON { chief_complaints[], pain_score, age,              |
|                    gender, arrival_transport,                         |
|                    vitals (ambulance/helicopter only) }               |
|     Tool:   ask_patient (interactive follow-up questions)            |
|     Note:   Collects EMS vitals for ambulance/helicopter patients    |
|                                                                      |
|  2. Triage Agent (ML - XGBoost v4)                 [OPERATIONAL]     |
|     Task:   triage_assessment_task                                   |
|     Input:  Complaints + pain + demographics + vitals (if available) |
|     Output: ESI acuity level (1-5), admission/discharge prediction   |
|     Tool:   triage_prediction_tool (wraps 2 XGBoost models)         |
|     Note:   Vitals used for ambulance/helicopter; masked for walk-in |
|                                                                      |
|  3. Doctor Agent - Initial (ML - XGBoost v1)       [OPERATIONAL]     |
|     Task:   doctor_assessment_task                                   |
|     Input:  Triage results + patient data                            |
|     Output: Preliminary diagnosis (14 classes) + department (11)     |
|     Tool:   doctor_prediction_tool (wraps 2 XGBoost v1 models)      |
|                                                                      |
|  4. Nurse Agent (Interactive)                      [OPERATIONAL]     |
|     Task:   nurse_data_collection_task                               |
|     Input:  Patient context from triage + initial assessment         |
|     Output: Vital signs (temp, HR, RR, O2, BP) + medication list    |
|     Tool:   nurse_data_collection (interactive stdin collection)     |
|     Note:   Each field can be skipped ("I don't know")              |
|                                                                      |
|  5. Doctor Agent - Enhanced (ML - XGBoost v2)      [OPERATIONAL]     |
|     Task:   doctor_reassessment_task                                 |
|     Input:  All prior data + vital signs + medications               |
|     Output: Enhanced diagnosis + department + comparison with v1     |
|     Tool:   doctor_prediction_tool_v2 (wraps 2 XGBoost v2 models)  |
|                                                                      |
|  5b. Doctor Disposition Refinement                 [OPERATIONAL]     |
|       (ML - XGBoost v3, isotonic-calibrated)                         |
|     Task:   doctor_disposition_task  (between nurse + reassessment)  |
|     Input:  Triage softmax + nurse vitals + meds + longitudinal+PMH  |
|     Output: Calibrated P(admit), refines triage's admit/discharge.   |
|             The post-nurse reassessment gates on THIS task's verdict |
|             rather than triage's screening prediction.                |
|     Tool:   doctor_disposition_tool                                   |
|     Note:   Plan section 3 / Option B. +3.77pp accuracy over triage  |
|             v3 cascade dispo on the same test rows. ECE = 0.0036.    |
|                                                                      |
|  Case Generation Agent (LLM)            [OFFLINE / BENCHMARK-ONLY]   |
|     Purpose: Translate MIMIC-IV tabular rows into grounded NL        |
|              patient cases to benchmark the NL→parser→models path.   |
|     NOT part of the live patient pipeline. See                       |
|     docs/agents/case-generation-agent.md.                            |
|                                                                      |
+---------------------------------------------------------------------+
```

### Offline tooling — Case Generation + end-to-end benchmark (Phase 4)

Separate from the live runtime above, the **Case Generation Agent**
(`src/proiect_licenta/case_generation.py`) turns real MIMIC-IV tabular rows into
grounded natural-language patient cases, and `benchmarks/benchmark_pipeline_e2e.py`
runs them back through the *whole* live crew to measure how much accuracy the
NL-translation layer costs versus feeding the models tabular columns directly.
The benchmark reports four columns on the same stay_ids — tool-direct,
feature-vector, feature-vector-gated, and E2E — which between them separate the
disposition-gate cost, the runtime feature-degradation cost, and the LLM cost.
This is the only LLM-generation component and it never sits in the
patient-facing pipeline. It also drove a live-pipeline improvement — the
**multi-reading-vitals fix** (the nurse tool can collect one or more additional readings, optionally timestamped, so the
disposition + v3 doctor tools build real longitudinal-vital features via
`build_longitudinal_block` instead of a single-snapshot fallback). Full design
in [`agents/case-generation-agent.md`](agents/case-generation-agent.md).

There are 4 distinct agents. The Doctor Agent runs **three** times in the live runtime — initial assessment (v3_base), disposition refinement (new binary admit/discharge model from plan section 3, Option B), and enhanced reassessment (v3 with nurse data) — yielding a 6-task pipeline. The reassessment gates on the disposition task's refined `is_admitted` rather than triage's screening verdict. All tasks execute sequentially via `Process.sequential`.

---

## Cascading Prediction Architecture

The system uses a cascading design where each model's output feeds into the next:

```
Chief complaints + demographics (+ EMS vitals for ambulance/helicopter) + PMH (all patients)
         |
         v  [2070 features for triage v3: 23 structured + 28 vital + 19 PMH + 2000 TF-IDF]
  [Acuity Model v3] -> predicted_acuity (ESI 1-5, ordinal-aware)
         |
         v
  [Disposition Model v3] -> P(admit)  -- the SOFT cascade source
         |                   (uses predicted_acuity int as feature; ROC AUC 0.86)
         |
    +----+----+
    |         |
    v         v
[Diagnosis  [Diagnosis  <- + snapshot vitals + longitudinal vitals + rhythm + meds + PMH
 Model v1]   Model v2]      (Doctor v3 nurse — 13 classes, the production diag/dept models)
    |         |
    v         v
[Department [Department
 Model v1]   Model v2]
    |         |
    v         v
 INITIAL    ENHANCED
 ASSESSMENT REASSESSMENT (with nurse data comparison)

         +---  PARALLEL post-nurse refinement (plan section 3, Option B, NEW)  ---+
         |                                                                          |
         |     [Doctor Disposition v3]                                              |
         |          - Soft cascade: 5 acuity softmax + 1 dispo probability          |
         |          - + snapshot vitals + longitudinal + rhythm + meds + PMH        |
         |          - Isotonic-calibrated, ECE 0.0036                               |
         |          - Trained on FULL 418K (admit + discharge), not admitted-only   |
         |          - +3.77pp accuracy over the triage v3 cascade dispo on the      |
         |            same 83,617-row test split. **Live in the crew as of 2026-05-30** |
         |            (`doctor_disposition_tool` between nurse + reassessment).     |
         +---------------------------------------------------------------------------+
```

- **Triage path (v3)** uses 2070 features: 23 v1 structured + 28 vital-sign features + 19 PMH + 2000 TF-IDF. Vitals are real for ambulance/helicopter patients and walk-in-masked for walk-ins (the `_missing` flag tells the model). PMH is sourced from prior MIMIC encounters at training and, at inference, from either the real prior-encounter record via the MRN/`subject_id` EHR lookup (`PatientHistoryLookupTool`, for returning patients — added 2026-06-03) or patient self-report with zero-fill fallback (first-time/unknown patients).
- **v1 doctor path** uses 2025 features (triage 2023 base + predicted_acuity + predicted_disposition).
- **v2 doctor path** uses 2056 features (doctor v1 2025 + 20 vital-sign features + 11 medication features).
- **Doctor v3 nurse path** uses ~2116 features (v2 base + longitudinal vitals + rhythm + PMH + Tier A softmax cascade).
- **Doctor disposition v3 (plan section 3 Option B, live since 2026-05-30)** uses **2128 features** = 2069 from the v3 triage input vector (calling `train_triage_v3.build_features(fit=False)` produces structured + v3 vitals + PMH + TF-IDF) + 6 soft cascade cols + 11 medication + 41 longitudinal vital + rhythm + 1 cascade housekeeping col. Trained on the FULL 418K stays (admit + discharge), unlike diagnosis/department which are admitted-only.
- Both doctor paths run sequentially so the user sees initial results before providing nurse data. The new disposition task (when wired) sits between the nurse step and the reassessment task; the reassessment will gate its tool call on *its* `is_admitted` flag rather than the triage one.
- The Department model always consumes the predicted diagnosis from its matching-version diagnosis model (diagnosis v1 -> department v1, diagnosis v2 -> department v2).

---

## Text Preprocessing Pipeline (shared across all models)

Both the training pipelines and inference tools apply identical preprocessing:

1. **Lowercase + strip** the chief complaint text.
2. **Replace separators** (`,`, `;`, `/`, `-`, `(`, `)`, `.`) with spaces.
3. **Expand abbreviations:** 45+ medical abbreviations mapped (e.g., `abd` -> `abdominal`, `sob` -> `shortness of breath`, `cp` -> `chest pain`, `cva` -> `cerebrovascular accident stroke`, etc.).
4. **TF-IDF vectorization** with the saved vectorizer (2000 features, unigram/bigram/trigram).
5. **Severity priors** computed by looking up each word's mean acuity from training data.

The same abbreviation map, vectorizer, and severity map are reused at inference by every prediction tool — guaranteeing feature parity between training and runtime.

---

## Project Structure

```
proiect_licenta/
|-- .env                          # GEMINI_API_KEY, MODEL config
|-- pyproject.toml                # Dependencies + script entry points
|-- README.md                     # Quick-start + doc map
|-- CLAUDE.md                     # Onboarding pointer for Claude Code sessions
|-- PROJECT_CONTEXT.md            # Top-level overview + doc index
|-- benchmarks/                   # Evaluation scripts (run with `uv run python benchmarks/<name>.py`)
|   |-- benchmark_triage_v1.py    # Triage v3b model benchmark (v1 pipeline)
|   |-- benchmark_triage_v2.py    # Triage v4 model benchmark (with vitals)
|   |-- benchmark_triage_v2_realistic.py  # v2 under realistic missing-vitals scenario
|   |-- benchmark_doctor.py       # Doctor v1 model benchmark
|   +-- benchmark_nurse.py        # Doctor v1 vs v2 comparison benchmark
|-- docs/
|   |-- architecture.md           # This file
|   |-- datasets.md               # MIMIC-IV dataset reference
|   |-- future-work.md            # Roadmap + v2 analysis + improvements
|   +-- agents/
|       |-- nlp-parser-agent.md
|       |-- triage-agent.md
|       |-- doctor-agent.md
|       +-- nurse-agent.md
|
|-- artifacts/                    # Trained model artifacts (gitignored)
|   |-- triage/
|   |   |-- v1/                   # Triage v3b (no vitals)
|   |   |   |-- acuity_model.joblib
|   |   |   |-- disposition_model.joblib
|   |   |   |-- tfidf_vectorizer.joblib
|   |   |   |-- severity_map.joblib
|   |   |   +-- model_metadata.json
|   |   +-- v2/                   # Triage v4 (with vitals — used at runtime)
|   |       |-- acuity_model.joblib
|   |       |-- disposition_model.joblib
|   |       |-- tfidf_vectorizer.joblib
|   |       |-- severity_map.joblib
|   |       |-- vital_medians.joblib   # Per-vital training medians for imputation
|   |       +-- model_metadata.json
|   +-- doctor/
|       |-- v1/                   # Doctor v1 (triage data only)
|       |   |-- diagnosis_model.joblib
|       |   |-- department_model.joblib
|       |   +-- metadata.json
|       +-- v2/                   # Doctor v2 (with nurse vitals + medications)
|           |-- diagnosis_model.joblib
|           |-- department_model.joblib
|           +-- metadata.json
|
|-- data/                         # Raw MIMIC-IV CSVs (gitignored)
|   |-- mimic-iv-ed/              # Emergency Department data (PRIMARY)
|   |   |-- triage.csv            # 425K rows: complaints, pain, vitals, acuity
|   |   |-- edstays.csv           # 425K rows: stay tracking, gender, arrival, disposition
|   |   |-- diagnosis.csv         # 900K rows: ICD-9/10 diagnosis codes per stay
|   |   |-- vitalsign.csv         # 1.4M rows: longitudinal vital signs during stay
|   |   |-- medrecon.csv          # 3M rows: medication reconciliation (pre-admission meds)
|   |   |-- pyxis.csv             # Medications dispensed during stay
|   |   +-- files_created/
|   |       +-- categorized_diagnosis.csv  # ICD codes grouped into categories
|   |-- mimic-iv/hosp/            # Hospital data
|   |   |-- patients.csv          # Demographics (age, gender, death date)
|   |   |-- admissions.csv        # Hospital admissions
|   |   |-- services.csv          # Department assignments (MED, SURG, NEURO, etc.)
|   |   |-- diagnoses_icd.csv     # Hospital-wide ICD diagnosis codes
|   |   +-- d_icd_diagnoses.csv   # ICD code dictionary
|   |-- mimic-iv/note/            # Clinical notes (unused -- see datasets.md)
|   |   |-- discharge.csv         # ~3.3 GB discharge summaries
|   |   |-- radiology.csv         # ~2.7 GB radiology reports
|   |   |-- discharge_detail.csv
|   |   +-- radiology_detail.csv
|   +-- mimic-iv/icu/             # ICU data (NOT USED)
|
+-- src/proiect_licenta/
    |-- main.py                   # Entry point -- interactive patient input -> crew kickoff
    |-- crew.py                   # CrewAI crew definition -- 4 agents, 5 tasks, sequential
    |-- paths.py                  # Canonical project paths (data/, artifacts/, ...)
    |-- preprocessing.py          # Shared complaint normalization + abbreviation map
    |-- config/
    |   |-- agents.yaml           # Agent definitions (nlp_parser, triage, doctor, nurse)
    |   +-- tasks.yaml            # Task definitions (5 tasks: parse, triage, doctor v1, nurse, doctor v2)
    |-- training/
    |   |-- train_triage_v1.py    # Triage ML training pipeline v3b (acuity + disposition, no vitals)
    |   |-- train_triage_v2.py    # Triage ML training pipeline v4 (+ vital signs, ambulance/helicopter)
    |   |-- train_doctor.py       # Doctor v1 ML training pipeline (diagnosis + department)
    |   +-- train_nurse.py        # Doctor v2 ML training pipeline (+ vitals + medications)
    +-- tools/
        |-- __init__.py           # Exports all tools
        |-- triage_tool.py        # CrewAI BaseTool wrapping triage ML models
        |-- doctor_tool.py        # CrewAI BaseTool wrapping doctor v1 ML models
        |-- doctor_tool_v2.py     # CrewAI BaseTool wrapping doctor v2 ML models (+ nurse data)
        |-- nurse_tool.py         # CrewAI BaseTool for interactive vital/medication collection
        +-- ask_patient_tool.py   # Interactive follow-up question tool
```

### Where things live (quick reference)

- **Paths** — `src/proiect_licenta/paths.py` is the single source of truth. Every dataset CSV path and every artifact directory is exported as a constant. Tools, training pipelines, and benchmarks all import from there; no path is hardcoded.
- **Preprocessing** — `src/proiect_licenta/preprocessing.py` owns `ABBREVIATIONS` and `normalize_complaint_text`. Triage v1, triage v2, doctor, nurse, and the runtime tools import the same function so training and inference cannot drift.
- **Doctor v1 vs v2 layout parity** — `artifacts/doctor/v1/` and `artifacts/doctor/v2/` use the same filenames (`diagnosis_model.joblib`, `department_model.joblib`, `metadata.json`). The version is the directory, not the filename.
- **Doctor v2 still loads triage v1 base artifacts** (TF-IDF vectorizer, severity map, acuity/disposition models). This is intentional: doctor v2's cascading features must match the same TF-IDF/severity space the doctor models were trained on, so it reads from `artifacts/triage/v1/`. Only the diagnosis and department weights differ between v1 and v2.
