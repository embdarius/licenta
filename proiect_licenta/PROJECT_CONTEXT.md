# Multi-Agent Medical Decision Support System — Project Context

## Overview

This is a **Bachelor's Thesis** project implementing a **Multi-Agent Architecture for Medical Decision Support** using **CrewAI** and **supervised machine learning** trained on the **MIMIC-IV Emergency Department** dataset (~425K real patient encounters).

The system simulates a clinical emergency department workflow: a patient describes their symptoms in natural language, and a pipeline of specialized AI agents processes the input through triage, initial diagnosis, nurse data collection, and enhanced reassessment.

**Tech stack:** Python 3.13, CrewAI 1.9.3, Gemini 2.5 Flash (LLM), XGBoost, scikit-learn, pandas, thefuzz.

---

## Documentation Index

This file is the slim top-level overview. Detailed documentation is split per topic under `docs/`:

| File | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Full agent pipeline diagram, cascading prediction design, shared text preprocessing pipeline, project structure on disk |
| [`docs/agents/nlp-parser-agent.md`](docs/agents/nlp-parser-agent.md) | NLP Parser Agent (LLM): role, input/output contract, `AskPatientTool` |
| [`docs/agents/triage-agent.md`](docs/agents/triage-agent.md) | Triage Agent: acuity + disposition XGBoost models, 2023 features, 66.7% / 75.9% benchmarks, training evolution v1 -> v3b |
| [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md) | Doctor Agent v1 + v2: 4 XGBoost models, diagnosis / department grouping tables, v1 vs v2 comparison, medication classification, vital sign processing |
| [`docs/agents/nurse-agent.md`](docs/agents/nurse-agent.md) | Nurse Agent: interactive collection flow, partial data handling, why a dedicated agent |
| [`docs/datasets.md`](docs/datasets.md) | MIMIC-IV table reference — used tables + inspected-but-unused tables (`vitalsign.csv`, `pyxis.csv`, `admissions.csv`, clinical notes) with leakage considerations |
| [`docs/future-work.md`](docs/future-work.md) | Why the Doctor v2 gains were modest, prioritized next-step recommendations, Phase 4 Text Generation, Phase 5 Hospital Infrastructure, model-level improvements, known issues |

**If you're new:** read `docs/architecture.md` first, then the four agent files in order (NLP Parser -> Triage -> Doctor -> Nurse).

### Repo Conventions (must-knows for editing)

- **`src/proiect_licenta/paths.py`** is the single source of truth for filesystem paths. All dataset CSVs and artifact directories (`TRIAGE_V1_DIR`, `DOCTOR_V2_DIR`, `TRIAGE_CSV`, ...) are exported as constants. Never hard-code paths.
- **`src/proiect_licenta/preprocessing.py`** owns `ABBREVIATIONS` and `normalize_complaint_text`. Triage v1/v2, doctor, nurse, and runtime tools all import from here so training and inference can't drift.
- **Datasets and trained model weights are gitignored.** Datasets live in `data/`, artifacts in `artifacts/{triage,doctor}/{v1,v2}/`. Retrain with `uv run train_*` (see "How to Run").

---

## System at a Glance

```
Patient -> NLP Parser -> Triage -> Doctor v1 -> Nurse -> Doctor v2
           (LLM)        (ML)      (ML)         (interactive) (ML+vitals+meds)
```

- **4 agents**, **5 tasks** (the Doctor runs twice).
- **6 XGBoost models total**: acuity, disposition, diagnosis v1, department v1, diagnosis v2, department v2.
- **Cascading:** each model's output feeds the next.
- See [`docs/architecture.md`](docs/architecture.md) for the full diagram and pipeline design.

### Headline Benchmark Numbers

| Model | Top-1 | Top-3 | Notes |
|---|---|---|---|
| Triage — Acuity (5 classes) | 66.7% | — | 97.7% within-1-level |
| Triage — Disposition (2 classes) | 75.9% | — | ROC AUC 0.84 |
| Doctor v1 — Diagnosis (14 classes) | 50.2% | 83.6% | 3.05x over random |
| Doctor v1 — Department (11 classes) | 59.1% | 92.5% | majority baseline 59.9% |
| Doctor v2 — Diagnosis (14 classes) | **52.4%** | **84.9%** | +2.3pp over v1 |
| Doctor v2 — Department (11 classes) | **65.0%** | **93.7%** | +5.9pp over v1; beats majority baseline |

Full per-class metrics and confusion matrices live in the per-agent docs.

---

## Design Decisions

1. **Two-phase doctor assessment** — The Doctor Agent runs twice: v1 (triage data only) and v2 (with nurse data). This allows direct comparison of predictions before and after vital signs/medication data, demonstrating the clinical value of additional data collection.
2. **LLM for NLP parsing, ML for prediction** — The NLP Parser uses Gemini (LLM) for natural language understanding. All prediction models use supervised ML (XGBoost) trained on 400K+ real encounters for reliable, auditable predictions.
3. **Cascading prediction** — Each model's output feeds into the next: acuity -> disposition -> diagnosis -> department. This mirrors clinical flow and improves downstream predictions.
4. **Soft class weighting** — Uses sqrt(inverse_frequency) to balance minority class recall without destroying majority class accuracy.
5. **Category grouping** — Small diagnosis/department categories are merged to ensure sufficient training samples per class. The "Nervous System" ICD-9/10 split is an artifact that was corrected by merging.
6. **100K training cap for doctor models** — To keep training times reasonable while benchmarking. Can be increased to full 157K for production.
7. **Admitted-only for doctor models** — Diagnosis and department prediction only applies to admitted patients. Discharged patients get a discharge summary instead.
8. **No ICU data** — Project focuses entirely on the Emergency Department pathway.

---

## How to Run

```bash
# Install dependencies
uv sync

# Train/retrain triage ML models (~90 minutes)
uv run train_models

# Train/retrain doctor v1 ML models (~30 minutes)
uv run train_doctor

# Train/retrain doctor v2 ML models with nurse data (~45 minutes)
uv run train_nurse

# Run the full 4-agent, 5-task system interactively
uv run crewai run

# Or equivalently
uv run run_crew

# Run benchmarks
uv run python benchmarks/benchmark_triage_v1.py             # Triage v1 models
uv run python benchmarks/benchmark_triage_v2.py             # Triage v2 models (with vitals)
uv run python benchmarks/benchmark_triage_v2_realistic.py   # Triage v2 under realistic missing-vitals scenario
uv run python benchmarks/benchmark_doctor.py                # Doctor v1 models
uv run python benchmarks/benchmark_nurse.py                 # Doctor v1 vs v2 comparison
```

### Environment Variables (`.env`)
```
MODEL=gemini/gemini-2.5-flash
GEMINI_API_KEY=<key>
```

---

## Status Summary

- **Phase 1 — Triage System:** Complete. See [`docs/agents/triage-agent.md`](docs/agents/triage-agent.md).
- **Phase 2 — Doctor v1 (initial assessment):** Complete. See [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md).
- **Phase 3 — Nurse Agent + Doctor v2:** Complete. See [`docs/agents/nurse-agent.md`](docs/agents/nurse-agent.md) and [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md).
- **Phase 4 — Text Generation Agent:** Planned. See [`docs/future-work.md`](docs/future-work.md).
- **Phase 5 — Hospital Infrastructure:** Planned. See [`docs/future-work.md`](docs/future-work.md).

Known issues, model-level improvement opportunities, and the full prioritized roadmap — including the analysis of *why* the v2 gains were modest and which unused MIMIC-IV tables could close the gap — are consolidated in [`docs/future-work.md`](docs/future-work.md).
