# Claude Code Onboarding

**Start here:** read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — it's the slim top-level overview with benchmark numbers, design decisions, and links into `docs/`.

## Doc map

| Where | What's in it |
|---|---|
| [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) | Overview, benchmarks, design decisions, run commands, status of each phase |
| [`docs/architecture.md`](docs/architecture.md) | Full pipeline diagram, cascading prediction design, on-disk layout |
| [`docs/agents/nlp-parser-agent.md`](docs/agents/nlp-parser-agent.md) | NLP Parser (LLM): role, I/O contract, `AskPatientTool` |
| [`docs/agents/triage-agent.md`](docs/agents/triage-agent.md) | Triage: acuity + disposition models, training evolution v1 → v3b, benchmarks |
| [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md) | Doctor v1 + v2: 4 XGBoost models, grouping tables, vital/medication processing |
| [`docs/agents/nurse-agent.md`](docs/agents/nurse-agent.md) | Nurse: interactive collection flow, partial data handling |
| [`docs/datasets.md`](docs/datasets.md) | MIMIC-IV table reference (used + inspected-but-unused, leakage notes) |
| [`docs/future-work.md`](docs/future-work.md) | Why v2 gains were modest, prioritized roadmap, known issues, planned phases 4 + 5 |

## Key facts to internalize before changing code

- **Pipeline:** Patient → NLP Parser → Triage → Doctor v1 → Nurse → Doctor v2. Four agents, five tasks (Doctor runs twice).
- **v1/v2 split is intentional**, not legacy. The thesis explicitly compares Doctor predictions before vs. after nurse-collected vitals/meds (design decision #1 in `PROJECT_CONTEXT.md`).
- **Models:** 6 total XGBoost models (acuity, disposition, diagnosis v1, department v1, diagnosis v2, department v2). All trained on MIMIC-IV-ED.
- **NLP = LLM, prediction = ML.** Don't replace XGBoost with the LLM.
- **Datasets and trained model artifacts are not committed.** Raw MIMIC-IV CSVs live under `data/` (gitignored). Trained `.joblib` files live under `artifacts/{triage,doctor}/{v1,v2}/` (gitignored).
- **Single source of truth for paths.** `src/proiect_licenta/paths.py` exports every dataset and artifact directory constant — `TRIAGE_V1_DIR`, `TRIAGE_V2_DIR`, `DOCTOR_V1_DIR`, `DOCTOR_V2_DIR`, `TRIAGE_CSV`, `EDSTAYS_CSV`, etc. Every tool, training pipeline, and benchmark imports from there. Never hard-code paths.
- **Shared complaint preprocessing.** `src/proiect_licenta/preprocessing.py` owns `ABBREVIATIONS` and `normalize_complaint_text`. Triage v1, v2, doctor, nurse, and the runtime tools all import from here so v1 and v2 stay peers (neither imports from the other).

## Source layout

- `src/proiect_licenta/main.py` — entry points (`run`, `train_*`).
- `src/proiect_licenta/crew.py` — CrewAI agent + task wiring.
- `src/proiect_licenta/paths.py` — canonical project paths (data/, artifacts/, etc.).
- `src/proiect_licenta/preprocessing.py` — shared complaint-text normalization.
- `src/proiect_licenta/config/{agents,tasks}.yaml` — agent prompts and task definitions.
- `src/proiect_licenta/tools/` — CrewAI BaseTool implementations (one per agent).
- `src/proiect_licenta/training/train_{triage_v1,triage_v2,doctor,nurse}.py` — training pipelines.
- `benchmarks/` — evaluation scripts.
- `data/` — raw MIMIC-IV CSVs (gitignored).
- `artifacts/{triage,doctor}/{v1,v2}/` — trained `.joblib` model files + `metadata.json` (gitignored).
- `docs/_archive/` — completed one-off scripts kept for thesis defense; not part of the live system.
