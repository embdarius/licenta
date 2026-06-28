# Claude Code Onboarding

**Start here:** read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — it's the slim top-level overview with benchmark numbers, design decisions, and links into `docs/`.

## Doc map

| Where | What's in it |
|---|---|
| [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) | Overview, benchmarks, design decisions, run commands, status of each phase |
| [`docs/architecture.md`](docs/architecture.md) | Full pipeline diagram, cascading prediction design, on-disk layout |
| [`docs/agents/nlp-parser-agent.md`](docs/agents/nlp-parser-agent.md) | NLP Parser (LLM): role, I/O contract, `AskPatientTool` |
| [`docs/agents/triage-agent.md`](docs/agents/triage-agent.md) | Triage: acuity + disposition models, training evolution v1 → v3b, benchmarks, constrained Group-2 Optuna HPO (reporting-only) |
| [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md) | Doctor v1/v2/v3 + disposition + Stage-2 ICD: XGBoost models, grouping tables, vital/medication processing, longitudinal vitals + rhythm, graded near-miss metrics |
| [`docs/agents/nurse-agent.md`](docs/agents/nurse-agent.md) | Nurse: interactive collection flow, partial data handling |
| [`docs/agents/case-generation-agent.md`](docs/agents/case-generation-agent.md) | Case Generation (Phase 4, offline/benchmark-only): tabular row → grounded NL patient case; 4-way E2E vs tabular benchmark (incl. gate-isolating column) + the runtime multi-reading-vitals fix |
| [`docs/llm-backend.md`](docs/llm-backend.md) | Switchable LLM backend (Gemini Flash 2.5 ↔ self-hosted MedGemma): `llm_config` design, vLLM/Colab serving, same-mode `parser-llm` benchmark, 20-case Experiment A results |
| [`docs/datasets.md`](docs/datasets.md) | MIMIC-IV table reference (used + inspected-but-unused, leakage notes) |
| [`docs/results/rebenchmark_v3/`](docs/results/rebenchmark_v3/) | Detailed v3 re-benchmark audit (2026-06-28): regression PASS vs `ceiling.json` + expanded metrics (per-class, calibration, disposition per-threshold table, graded ICD, Flash-vs-MedGemma + parser-cost). Raw artifacts: `benchmarks/audit/2026-06-28/` |
| [`docs/future-work.md`](docs/future-work.md) | Why v2 gains were modest, prioritized roadmap, known issues, planned phases 4 + 5 |
| [`webapp/README.md`](webapp/README.md) | Web interface — live conversational runtime (drives the real crew over SSE; injected interactive-tool channel + event-bus telemetry + intake-confirmation gate) + retained stage/parity test path |

## Key facts to internalize before changing code

- **Pipeline:** Patient → NLP Parser → Triage → Doctor (initial) → Nurse → Doctor (disposition) → Doctor (reassessment). Four agents, six tasks (Doctor runs three times). The live runtime is the **v3 stack** (rewired 2026-05-30).
- **The before/after-nurse split is intentional**, not legacy. The thesis explicitly compares Doctor predictions before vs. after nurse-collected vitals/meds (design decision #1 in `PROJECT_CONTEXT.md`). v1/v2 tool files + artifacts are retained on disk as 14-class thesis baselines but are no longer registered with the live crew.
- **Models (live):** triage v3 acuity + disposition, doctor v3_base diagnosis + department (initial), doctor disposition v3 (calibrated binary admit/discharge), doctor v3-nurse diagnosis + department (reassessment), plus the offline Stage-2 exact-ICD resolver. v1/v2 models remain as baselines. All trained on MIMIC-IV-ED.
- **NLP = LLM, prediction = ML.** Don't replace XGBoost with the LLM.
- **Datasets and trained model artifacts are not committed.** Raw MIMIC-IV CSVs live under `data/` (gitignored). Trained `.joblib` files live under `artifacts/triage/{v1,v2,v3}/` and `artifacts/doctor/{v1,v2,v3_base,v3}/` (gitignored; `doctor/v3/` also holds `disposition_model.joblib` and the Stage-2 `icd_resolver/`).
- **Single source of truth for paths.** `src/proiect_licenta/paths.py` exports every dataset and artifact directory constant — `TRIAGE_V1_DIR`, `TRIAGE_V2_DIR`, `TRIAGE_V3_DIR`, `DOCTOR_V1_DIR`, `DOCTOR_V2_DIR`, `DOCTOR_V3_BASE_DIR`, `DOCTOR_V3_DIR`, `DOCTOR_V3_ICD_RESOLVER_DIR`, `TRIAGE_CSV`, `EDSTAYS_CSV`, etc. Every tool, training pipeline, and benchmark imports from there. Never hard-code paths.
- **Shared complaint preprocessing.** `src/proiect_licenta/preprocessing.py` owns `ABBREVIATIONS` and `normalize_complaint_text`. Triage (v1/v2/v3), doctor (v1/v2/v3), nurse, and the runtime tools all import from here so the model tiers stay peers (no tier imports from another).
- **Case Generation Agent is offline/benchmark-only** (`src/proiect_licenta/case_generation.py`, `uv run generate_cases`). It is NOT in the live patient crew. It uses **dedicated** `config/case_generation_{agents,tasks}.yaml` — do NOT add its agent/task to the shared `agents.yaml`/`tasks.yaml`, because `@CrewBase` maps every task in a file to an `@agent` method and that would break the live `ProiectLicenta` crew. Generated cases live under `data/derived/synthetic_cases/` (gitignored — MIMIC DUA).

## Source layout

- `src/proiect_licenta/main.py` — entry points (`run`, `train_*`).
- `src/proiect_licenta/crew.py` — CrewAI agent + task wiring.
- `src/proiect_licenta/paths.py` — canonical project paths (data/, artifacts/, etc.).
- `src/proiect_licenta/preprocessing.py` — shared complaint-text normalization.
- `src/proiect_licenta/llm_config.py` — switchable LLM backend (`get_llm`, `LLM_BACKEND` flash↔medgemma).
- `src/proiect_licenta/pmh_features.py`, `pmh_vocab.py` — Past Medical History feature derivation + vocabulary.
- `src/proiect_licenta/icd_resolution.py`, `vital_trajectory.py` — Stage-2 exact-ICD resolver + longitudinal-vital helpers.
- `src/proiect_licenta/icd_similarity.py` — evaluation-only graded near-miss metrics for Stage-2 (TF-IDF + Gemini title cosine, ICD-tree distance); used by `benchmark_icd_resolution.py` + `build_title_embeddings.py`.
- `src/proiect_licenta/config/{agents,tasks}.yaml` — live agent prompts + task definitions; `config/case_generation_{agents,tasks}.yaml` — dedicated offline case-generation crew.
- `src/proiect_licenta/tools/` — CrewAI BaseTool implementations (live: triage v3, doctor v3_base, doctor disposition, doctor v3, nurse, ask-patient, patient-history-lookup; v1/v2 tools retained as baselines).
- `src/proiect_licenta/training/` — training pipelines: `train_{triage_v1,triage_v2,triage_v3,doctor,nurse,doctor_v3,nurse_v3,doctor_disposition,icd_resolver}.py`.
- `benchmarks/` — evaluation scripts.
- `data/` — raw MIMIC-IV CSVs (gitignored).
- `artifacts/triage/{v1,v2,v3}/`, `artifacts/doctor/{v1,v2,v3_base,v3}/` — trained `.joblib` model files + `metadata.json` (gitignored; `doctor/v3/` adds `disposition_model.joblib` + `icd_resolver/`).
- `docs/_archive/` — completed one-off scripts kept for thesis defense; not part of the live system.
