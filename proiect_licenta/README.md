# Multi-Agent Medical Decision Support System

Bachelor's thesis project implementing a multi-agent architecture for emergency-department decision support, built on **CrewAI** + **XGBoost** and trained on the **MIMIC-IV-ED** dataset (~425K real patient encounters).

A patient describes their symptoms in natural language; a pipeline of four specialized agents walks the case through triage, initial diagnosis, vital-sign collection, a calibrated disposition refinement, and an enhanced reassessment. The Doctor runs three times (6 tasks total).

```
Patient → NLP Parser → Triage → Doctor (initial) → Nurse → Doctor (disposition) → Doctor (reassessment)
          (LLM)        (ML v3)  (ML v3_base)       (interactive) (ML, calibrated)  (ML v3 +vitals+meds)
```

Live models are the **v3 stack** (rewired 2026-05-30); the v1/v2 models are retained as thesis baselines.

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

# Run the full 4-agent / 6-task pipeline
uv run run_crew

# Train models — live v3 stack (see PROJECT_CONTEXT.md "How to Run" for the full list + timings)
uv run train_triage_v3         # triage v3 (vitals + PMH) — live
uv run train_doctor_v3         # doctor v3 base (catch-all excluded) — live initial
uv run train_nurse_v3          # doctor v3 with nurse data — live reassessment
uv run train_doctor_disposition  # calibrated binary admit/discharge — live disposition
# v1/v2 baselines: uv run train_models / train_triage_v2 / train_doctor / train_nurse

# Benchmarks
uv run python benchmarks/benchmark_triage_v3.py
uv run python benchmarks/benchmark_doctor_v3.py
uv run python benchmarks/benchmark_nurse_v3.py
```

Datasets are not committed — place the MIMIC-IV CSVs under `data/` (e.g. `data/mimic-iv-ed/`, `data/mimic-iv/hosp/`), the layout encoded in `src/proiect_licenta/paths.py` (see `docs/datasets.md`).

## Tech stack

Python 3.13 · CrewAI 1.9.3 · Gemini 2.5 Flash · XGBoost · scikit-learn · pandas · thefuzz
