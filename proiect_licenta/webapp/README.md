# Web UI — Live Runtime Inference

An interactive website for the multi-agent ED decision-support pipeline. It runs
the **same live inference as the CLI crew** (`uv run run_crew`) but stage by
stage with a polished dashboard, so it's suitable for a thesis demo.

```
Intake (LLM parse) → Triage → Initial Dx → Nurse → Disposition refinement → Reassessment
```

## How it mirrors the live crew exactly

The backend does **not** re-implement any model logic. It calls the **same tool
classes** the CrewAI crew uses, via the same argument wiring proven in
`benchmarks/benchmark_pipeline_e2e.py` (`run_tool_direct`). All of this lives in
the shared module [`src/proiect_licenta/pipeline.py`](../src/proiect_licenta/pipeline.py):

- `TriagePredictionTool` · `DoctorPredictionToolV3Base` · `DoctorDispositionTool`
  · `DoctorPredictionToolV3` · `PatientHistoryLookupTool`
- MRN/EHR lookup overrides self-reported PMH/meds (tasks.yaml step 1b/2b).
- The nurse step collects **one or more chronological readings** and builds the
  identical payload via `build_nurse_payload` (shared with the stdin nurse tool).
- The reassessment **gates on the refined disposition**, not the triage verdict.

Only the stdin/LLM-ask *interactivity* is replaced by web forms. Predictions are
byte-identical — see the parity test below.

## Run it

Two processes. From the repo root:

```bash
# 1) Backend (FastAPI). Reads .env (GEMINI_API_KEY, MODEL, LLM_BACKEND).
uv sync --extra web
uv run uvicorn webapp.backend.main:app --port 8000 --reload

# 2) Frontend (Vite dev server, proxies /api -> :8000)
cd webapp/frontend
npm install
npm run dev          # http://localhost:5173
```

If the backend runs on a different port, set `BACKEND_URL` for the Vite proxy:
`BACKEND_URL=http://127.0.0.1:8071 npm run dev`.

## Verify parity (web == crew)

```bash
uv run python webapp/backend/test_parity.py 12
# -> "PARITY OK — web pipeline is byte-identical to the crew path."
```

It drives synthetic cases through the same stage functions the API calls and
asserts acuity, triage admit, refined admit, and the top-1 diagnosis/department
match the benchmark's `run_tool_direct`.

## Layout

```
webapp/
  backend/
    main.py          FastAPI app: in-memory session + 6 stage endpoints + CORS
    schemas.py       pydantic request/response models
    test_parity.py   web-vs-benchmark parity test
  frontend/          Vite + React + TS + Tailwind SPA
    src/
      App.tsx        stepper shell + progress rail; orchestrates the stages
      api.ts         typed fetch client (POST per stage)
      components/    AcuityGauge, DispositionCard/Flip, Top3Bars, NurseForm,
                     IntakePanel, IcdDifferential, ComparisonPanel, ProgressRail
```

## API (session-scoped, sequential)

| Method + path | Purpose |
|---|---|
| `POST /api/session` | new session id |
| `POST /api/parse` | LLM parse free text → prefilled intake fields |
| `POST /api/triage` | EHR lookup + triage acuity/disposition |
| `POST /api/doctor-initial?session_id=` | initial diagnosis/department (v3_base) |
| `POST /api/nurse` | assemble nurse payload (multi-reading) |
| `POST /api/disposition?session_id=` | calibrated admit/discharge refinement |
| `POST /api/reassessment?session_id=` | enhanced diagnosis/department + ICD |

The session store is in-memory and process-local (fine for a single-user demo).
