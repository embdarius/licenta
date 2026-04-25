# Multi-Agent Medical Decision Support System

Bachelor's thesis project implementing a multi-agent architecture for emergency-department decision support, built on **CrewAI** + **XGBoost** and trained on the **MIMIC-IV-ED** dataset (~425K real patient encounters).

A patient describes their symptoms in natural language; a pipeline of four specialized agents (NLP Parser → Triage → Doctor → Nurse → Doctor v2) walks the case through triage, initial diagnosis, vital-sign collection, and an enhanced reassessment.

```
Patient → NLP Parser → Triage → Doctor v1 → Nurse → Doctor v2
          (LLM)        (ML)     (ML)        (interactive)  (ML+vitals+meds)
```

## Documentation

- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** — top-level overview, benchmarks, design decisions, status.
- **[docs/architecture.md](docs/architecture.md)** — full pipeline diagram, cascading prediction design, on-disk layout.
- **[docs/agents/](docs/agents/)** — per-agent deep dives (NLP Parser, Triage, Doctor, Nurse).
- **[docs/datasets.md](docs/datasets.md)** — MIMIC-IV table reference.
- **[docs/future-work.md](docs/future-work.md)** — known issues, roadmap, planned phases 4 + 5.

If you're new, read `PROJECT_CONTEXT.md` first.

## Quick start

```bash
# Install
uv sync

# Configure: create .env with
#   MODEL=gemini/gemini-2.5-flash
#   GEMINI_API_KEY=<your key>

# Run the full 4-agent pipeline
uv run run_crew

# Train models (see PROJECT_CONTEXT.md for timings)
uv run train_models       # triage v1
uv run train_triage_v2    # triage v2 (with vitals)
uv run train_doctor       # doctor v1
uv run train_nurse        # doctor v2 (uses nurse data)

# Benchmarks
uv run python benchmarks/benchmark_triage_v2.py
uv run python benchmarks/benchmark_doctor.py
uv run python benchmarks/benchmark_nurse.py
```

Datasets are not committed — place the MIMIC-IV CSVs under `src/proiect_licenta/datasets/datasets_mimic-iv/` (see `docs/datasets.md`).

## Tech stack

Python 3.13 · CrewAI 1.9.3 · Gemini 2.5 Flash · XGBoost · scikit-learn · pandas · thefuzz
