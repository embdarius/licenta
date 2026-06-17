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
Patient -> NLP Parser -> Triage -> Doctor (initial) -> Nurse -> Doctor (disposition) -> Doctor (reassessment)
           (LLM)        (ML v3)   (ML v3_base)        (interactive) (ML, calibrated)   (ML v3 +vitals+meds)
```

The system simulates a clinical emergency department workflow: a patient describes their symptoms in natural language, and a pipeline of specialized AI agents processes the input through triage, initial diagnosis, nurse data collection, a calibrated disposition refinement, and enhanced reassessment. The live runtime is the **v3 stack** (rewired 2026-05-30); v1/v2 models are retained as thesis baselines.

---

## Full Agent Pipeline (4 operational agents, 6 tasks)

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
|  2. Triage Agent (ML - XGBoost v3)                 [OPERATIONAL]     |
|     Task:   triage_assessment_task                                   |
|     Input:  Complaints + pain + demographics + vitals + PMH          |
|     Output: ESI acuity level (1-5), admission/discharge prediction   |
|     Tool:   TriagePredictionTool (wraps 2 XGBoost v3 models),        |
|             PatientHistoryLookupTool (MRN/EHR PMH lookup)            |
|     Note:   Vitals used for ambulance/helicopter; masked for walk-in |
|                                                                      |
|  3. Doctor Agent - Initial (ML - XGBoost v3_base)  [OPERATIONAL]     |
|     Task:   doctor_assessment_task                                   |
|     Input:  Triage results + patient data                            |
|     Output: Preliminary diagnosis (13 classes) + department (11),    |
|             top-3 with probabilities                                  |
|     Tool:   DoctorPredictionToolV3Base (2 XGBoost v3_base models)    |
|                                                                      |
|  4. Nurse Agent (Interactive)                      [OPERATIONAL]     |
|     Task:   nurse_data_collection_task                               |
|     Input:  Patient context from triage + initial assessment         |
|     Output: Vital signs (temp, HR, RR, O2, BP) + medication list    |
|     Tool:   NurseDataCollectionTool (interactive stdin collection)   |
|     Note:   Each field can be skipped ("I don't know"); can collect  |
|             a 2nd vitals reading for longitudinal features           |
|                                                                      |
|  5. Doctor Disposition Refinement                  [OPERATIONAL]     |
|       (ML - XGBoost v3, isotonic-calibrated)                         |
|     Task:   doctor_disposition_task  (between nurse + reassessment)  |
|     Input:  Triage softmax + nurse vitals + meds + longitudinal+PMH  |
|     Output: Calibrated P(admit), refines triage's admit/discharge.   |
|             The post-nurse reassessment gates on THIS task's verdict |
|             rather than triage's screening prediction.                |
|     Tool:   DoctorDispositionTool                                    |
|     Note:   Plan section 3 / Option B. +3.77pp accuracy over triage  |
|             v3 cascade dispo on the same test rows. ECE = 0.0036.    |
|                                                                      |
|  6. Doctor Agent - Enhanced (ML - XGBoost v3)      [OPERATIONAL]     |
|     Task:   doctor_reassessment_task                                 |
|     Input:  All prior data + vital signs + medications               |
|     Output: Enhanced diagnosis + department + comparison with initial |
|     Tool:   DoctorPredictionToolV3 (2 XGBoost v3 nurse models)      |
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
 v3_base]    v3 nurse]      (both 13 classes — the live production diag/dept models)
    |         |
    v         v
[Department [Department
 v3_base]    v3 nurse]
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
- **Doctor v3_base path (live initial)** uses the v1 feature set with the catch-all class excluded (13-class label space, no nurse data yet).
- **Doctor v3 nurse path (live reassessment)** uses ~2116 features (v3_base base + snapshot vitals + longitudinal vitals + rhythm + meds + PMH + Tier A softmax cascade).
- **Doctor disposition v3 (plan section 3 Option B, live since 2026-05-30)** uses **2128 features** = 2069 from the v3 triage input vector (calling `train_triage_v3.build_features(fit=False)` produces structured + v3 vitals + PMH + TF-IDF) + 6 soft cascade cols + 11 medication + 41 longitudinal vital + rhythm + 1 cascade housekeeping col. Trained on the FULL 418K stays (admit + discharge), unlike diagnosis/department which are admitted-only.
- The retained **v1/v2 doctor baselines** use 2025 / 2056 features respectively (v1 = triage base + predicted_acuity + predicted_disposition; v2 = v1 + vital-sign + medication features) — kept on disk for thesis comparison, not in the live crew.
- The doctor passes run sequentially so the user sees initial results before providing nurse data. The disposition task sits between the nurse step and the reassessment task; the reassessment gates its tool call on *its* `is_admitted` flag rather than the triage one.
- The Department model always consumes the predicted diagnosis from its matching-version diagnosis model (diagnosis v3_base -> department v3_base, diagnosis v3 nurse -> department v3 nurse).

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
|   |-- benchmark_triage_v1.py    # Triage v1 model benchmark (pre-vital baseline)
|   |-- benchmark_triage_v2.py    # Triage v2 model benchmark (with vitals)
|   |-- benchmark_triage_v2_realistic.py  # v2 under realistic missing-vitals scenario
|   |-- benchmark_triage_v3.py    # Triage v3 (vitals + PMH) — head-to-head vs v2 (live model)
|   |-- benchmark_doctor.py       # Doctor v1 model benchmark
|   |-- benchmark_nurse.py        # Doctor v1 vs v2 comparison benchmark
|   |-- benchmark_doctor_v3.py    # Doctor v3 base (13 classes, catch-all excluded)
|   |-- benchmark_nurse_v3.py     # Doctor v3 base vs v3 with-nurse
|   |-- benchmark_doctor_disposition.py  # Doctor disposition v3 vs triage v3 cascade dispo
|   |-- benchmark_icd_resolution.py      # Stage-2 exact-ICD (rollup + full-code)
|   |-- benchmark_pipeline_e2e.py # End-to-end crew vs tabular baselines (Phase 4, 20 cases)
|   +-- compare_all_versions.py   # Four-way table v1/v2/v3-base/v3-nurse
|-- docs/
|   |-- architecture.md           # This file
|   |-- datasets.md               # MIMIC-IV dataset reference
|   |-- llm-backend.md            # Switchable LLM backend (Gemini Flash ↔ MedGemma)
|   |-- future-work.md            # Roadmap + v2 analysis + improvements
|   +-- agents/
|       |-- nlp-parser-agent.md
|       |-- triage-agent.md
|       |-- doctor-agent.md
|       |-- nurse-agent.md
|       +-- case-generation-agent.md
|
|-- artifacts/                    # Trained model artifacts (gitignored)
|   |-- triage/
|   |   |-- v1/                   # Triage v1 (no vitals) — baseline
|   |   |   |-- acuity_model.joblib
|   |   |   |-- disposition_model.joblib
|   |   |   |-- tfidf_vectorizer.joblib
|   |   |   |-- severity_map.joblib
|   |   |   +-- model_metadata.json
|   |   |-- v2/                   # Triage v2 (with vitals) — baseline
|   |   |   |-- acuity_model.joblib
|   |   |   |-- disposition_model.joblib
|   |   |   |-- tfidf_vectorizer.joblib
|   |   |   |-- severity_map.joblib
|   |   |   |-- vital_medians.joblib   # Per-vital training medians for imputation
|   |   |   +-- model_metadata.json
|   |   +-- v3/                   # Triage v3 (vitals + PMH) — LIVE runtime model
|   |       |-- acuity_model.joblib
|   |       |-- disposition_model.joblib
|   |       |-- tfidf_vectorizer.joblib
|   |       |-- severity_map.joblib
|   |       |-- vital_medians.joblib
|   |       +-- model_metadata.json
|   |-- doctor/
|   |   |-- v1/                   # Doctor v1 (triage data only) — baseline
|   |   |   |-- diagnosis_model.joblib
|   |   |   |-- department_model.joblib
|   |   |   +-- metadata.json
|   |   |-- v2/                   # Doctor v2 (with nurse vitals + meds) — baseline
|   |   |   |-- diagnosis_model.joblib
|   |   |   |-- department_model.joblib
|   |   |   +-- metadata.json
|   |   |-- v3_base/              # Doctor v3 base (13 classes) — LIVE initial assessment
|   |   |   |-- diagnosis_model.joblib
|   |   |   |-- department_model.joblib
|   |   |   +-- metadata.json
|   |   +-- v3/                   # Doctor v3 with nurse — LIVE reassessment + disposition
|   |       |-- diagnosis_model.joblib
|   |       |-- department_model.joblib
|   |       |-- disposition_model.joblib       # Calibrated binary admit/discharge (+ _raw sibling)
|   |       |-- metadata.json
|   |       +-- icd_resolver/      # Stage-2 exact-ICD candidate indices (offline)
|   +-- history/
|       +-- pmh_index.joblib      # Prebuilt PMH/med EHR index for MRN lookup (build_history_index)
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
    |-- crew.py                   # CrewAI crew definition -- 4 agents, 6 tasks, sequential
    |-- paths.py                  # Canonical project paths (data/, artifacts/, ...)
    |-- preprocessing.py          # Shared complaint normalization + abbreviation map
    |-- llm_config.py             # Switchable LLM backend (get_llm, LLM_BACKEND flash↔medgemma)
    |-- pmh_features.py           # Past Medical History feature derivation + EHR-lookup helpers
    |-- pmh_vocab.py              # PMH vocabulary -> diagnosis-group mapping
    |-- icd_resolution.py         # Stage-2 exact-ICD retrieval cascade
    |-- vital_trajectory.py       # Longitudinal-vital aggregation helpers
    |-- case_generation.py        # Phase 4 offline synthetic-case generator (generate_cases)
    |-- config/
    |   |-- agents.yaml           # Live agent definitions (nlp_parser, triage, doctor, nurse)
    |   |-- tasks.yaml            # Live task definitions (6 tasks: parse, triage, doctor initial,
    |   |                         #   nurse, doctor disposition, doctor reassessment)
    |   |-- case_generation_agents.yaml   # Dedicated offline case-generator agent
    |   +-- case_generation_tasks.yaml    # Dedicated offline case-generator task
    |-- training/
    |   |-- train_triage_v1.py    # Triage v1 (acuity + disposition, no vitals) — baseline
    |   |-- train_triage_v2.py    # Triage v2 (+ vital signs) — baseline
    |   |-- train_triage_v3.py    # Triage v3 (+ vitals + PMH) — LIVE
    |   |-- train_doctor.py       # Doctor v1 (diagnosis + department) — baseline
    |   |-- train_nurse.py        # Doctor v2 (+ vitals + meds) — baseline
    |   |-- train_doctor_v3.py    # Doctor v3 base (13 classes) — LIVE initial
    |   |-- train_nurse_v3.py     # Doctor v3 with nurse (longitudinal vitals + rhythm) — LIVE
    |   |-- train_doctor_disposition.py   # Calibrated binary admit/discharge — LIVE disposition
    |   +-- train_icd_resolver.py # Stage-2 exact-ICD candidate indices (offline)
    +-- tools/
        |-- __init__.py           # Exports all tools
        |-- triage_tool.py        # TriagePredictionTool (triage v3) — LIVE
        |-- doctor_tool.py        # DoctorPredictionTool (doctor v1) — baseline
        |-- doctor_tool_v2.py     # DoctorPredictionToolV2 (doctor v2) — baseline
        |-- doctor_tool_v3_base.py  # DoctorPredictionToolV3Base — LIVE initial
        |-- doctor_tool_v3.py     # DoctorPredictionToolV3 (+ nurse data) — LIVE reassessment
        |-- doctor_disposition_tool.py  # DoctorDispositionTool (calibrated) — LIVE disposition
        |-- nurse_tool.py         # NurseDataCollectionTool — interactive vital/med collection
        |-- ask_patient_tool.py   # AskPatientTool — interactive follow-up questions
        |-- patient_history_lookup_tool.py  # PatientHistoryLookupTool — MRN/EHR PMH + med lookup
        |-- med_vocab.py          # Medication vocabulary + feature helpers
        +-- vital_trajectory_io.py  # Longitudinal-vital block assembly for the tools
```

### Where things live (quick reference)

- **Paths** — `src/proiect_licenta/paths.py` is the single source of truth. Every dataset CSV path and every artifact directory is exported as a constant. Tools, training pipelines, and benchmarks all import from there; no path is hardcoded.
- **Preprocessing** — `src/proiect_licenta/preprocessing.py` owns `ABBREVIATIONS` and `normalize_complaint_text`. Every triage tier (v1/v2/v3), every doctor tier (v1/v2/v3), nurse, and the runtime tools import the same function so training and inference cannot drift.
- **Doctor layout parity** — `artifacts/doctor/{v1,v2,v3_base,v3}/` use the same filenames (`diagnosis_model.joblib`, `department_model.joblib`, `metadata.json`); `v3/` adds `disposition_model.joblib` and the `icd_resolver/` subdir. The version is the directory, not the filename.
- **Live cascade artifact wiring** — the live doctor v3_base / v3 tools and the disposition tool consume the triage **v3** TF-IDF/severity/PMH space (`artifacts/triage/v3/`) so their cascading features match what the v3 models were trained on. (The retained doctor v2 baseline still reads triage **v1** base artifacts for the same parity reason against its own training space.)
