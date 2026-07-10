# Multi-Agent Medical Decision Support System

Bachelor's thesis project: a multi-agent architecture for emergency-department
decision support, built on CrewAI and XGBoost and trained on MIMIC-IV-ED
(~425K real patient encounters).

A patient describes their symptoms in natural language, and a pipeline of four
specialized agents walks the case through triage, an initial diagnosis, vital-sign
collection, a calibrated disposition refinement, and an enhanced reassessment. The
Doctor runs three times, for six tasks total.

```
Patient -> NLP Parser -> Triage -> Doctor (initial) -> Nurse -> Doctor (disposition) -> Doctor (reassessment)
           (LLM)         (ML v3)  (ML v3_base)        (interactive) (ML, calibrated)   (ML v3 +vitals+meds)
```

The live models are the v3 stack; the v1/v2 models are retained as thesis
baselines. NLP parsing is done by the LLM; every prediction is done by the ML
models (the LLM does not replace XGBoost).

## Documentation

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — overview, benchmarks, design decisions, status.
- [docs/architecture.md](docs/architecture.md) — pipeline diagram, cascading-prediction design, on-disk layout.
- [docs/agents/](docs/agents/) — per-agent deep dives (NLP Parser, Triage, Doctor, Nurse).
- [docs/datasets.md](docs/datasets.md) — MIMIC-IV table reference.
- [docs/future-work.md](docs/future-work.md) — known issues, roadmap, planned phases.

If you are new, read PROJECT_CONTEXT.md first.

## Install

```bash
uv sync

# Create .env with:
#   MODEL=gemini/gemini-2.5-flash
#   GEMINI_API_KEY=<your key>
```

Datasets are not committed. Place the MIMIC-IV CSVs under `data/` (e.g.
`data/mimic-iv-ed/`, `data/mimic-iv/hosp/`); the exact layout is encoded in
`src/proiect_licenta/paths.py` and documented in [docs/datasets.md](docs/datasets.md).

## Training the models

```bash
# Live v3 stack
uv run train_triage_v3          # triage v3 (vitals + PMH)
uv run train_doctor_v3          # doctor v3 base (initial assessment)
uv run train_nurse_v3           # doctor v3 with nurse data (reassessment)
uv run train_doctor_disposition # calibrated binary admit/discharge

# v1/v2 baselines
uv run train_models             # triage v1
uv run train_triage_v2
uv run train_doctor
uv run train_nurse

# Benchmarks
uv run python benchmarks/benchmark_triage_v3.py
uv run python benchmarks/benchmark_doctor_v3.py
uv run python benchmarks/benchmark_nurse_v3.py
```

See PROJECT_CONTEXT.md ("How to Run") for the full command list and timings.

## Running the CLI

```bash
uv run run_crew
```

This runs the full four-agent, six-task pipeline interactively at the terminal:
the agents ask their questions on stdin and print their assessments to stdout.

## Running the web interface

The web interface drives the same crew as `uv run run_crew` and renders it as a
live clinical conversation: you see which agent is active, answer questions in the
browser, and get the structured tool outputs as inline result cards. The backend
runs the actual `ProiectLicenta().crew().kickoff(...)`; only the transport of the
interactive `input()` calls and the agent/tool telemetry moves from stdin/stdout
to the browser over SSE.

Run it as two processes from the repo root:

```bash
# Backend (FastAPI). Reads .env (GEMINI_API_KEY, MODEL, LLM_BACKEND).
uv sync --extra web
uv run uvicorn webapp.backend.main:app --port 8000 --reload

# Frontend (Vite dev server, proxies /api -> :8000)
cd webapp/frontend
npm install
npm run dev          # http://localhost:5173
```

If the backend runs on a different port, set `BACKEND_URL` for the Vite proxy:
`BACKEND_URL=http://127.0.0.1:8071 npm run dev`.

The live API is a small set of SSE + POST endpoints under `/api/live`
(`start`, `stream/{sid}`, `answer/{sid}`, `cancel/{sid}`); see
`webapp/backend/live.py`. An earlier stage-by-stage path
(`src/proiect_licenta/pipeline.py` plus the `/api/parse`, `/api/triage`, ...
endpoints) is retained as a test/utility and checked by
`webapp/backend/test_parity.py`, which confirms the direct-tool pipeline matches
the crew path.

## Tech stack

Python 3.13, CrewAI, Gemini 2.5 Flash, XGBoost, scikit-learn, pandas, thefuzz;
web interface on FastAPI + Vite + React + TypeScript + Tailwind.
