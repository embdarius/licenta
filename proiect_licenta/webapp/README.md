# Web UI — Live Conversational Runtime

An interactive website that **drives the real agentic crew** — the same 4-agent /
6-task pipeline as `uv run run_crew` — and renders it as a clean, live clinical
conversation. You watch which agent is active, what they ask, and which tools run
in real time; you answer their questions in the browser; the structured tool
outputs appear inline as rich result cards.

```
Intake (NLP parse + confirm) → Triage → Initial Dx → Nurse → Disposition refinement → Reassessment
```

This is **not** a re-implementation or a simplified demo: the backend runs the
actual `ProiectLicenta().crew().kickoff(...)`. The only thing replaced is the
*transport* of the interactive `input()` calls and the agent/tool telemetry —
from stdin/stdout to the browser over SSE.

## How it drives the real crew

The terminal runtime blocks on `input()` inside `ask_patient` / `nurse` /
`confirm_intake`, and `kickoff()` runs synchronously to a single text blob. The
web layer bridges this without forking the pipeline:

- **Per-session background thread.** Each session runs its own crew in a thread,
  so FastAPI stays responsive.
- **Interactive I/O via an injected channel.** The interactive tools gained a
  `channel` field ([`src/proiect_licenta/interaction.py`](../src/proiect_licenta/interaction.py),
  `make_ask`). When set, a tool's question is emitted to the browser and the crew
  thread **blocks** until the user POSTs an answer; when unset (the CLI), it falls
  back to `print()/input()` — terminal behavior is byte-preserved. Each ask carries
  a `kind` hint (`text`/`number`/`bp`/`rhythm`/`yesno`/`integer`/`intake_form`) so
  the UI renders the right input widget (the "hybrid smart prompts").
- **Live telemetry via the CrewAI event bus.** Global listeners on
  `crewai_event_bus` (`TaskStarted/Completed`, `AgentExecutionStarted/Completed`,
  `ToolUsageStarted/Finished`) stream agent/tool/task activity to the active
  session, attributed via a `ContextVar` that the bus propagates through
  `copy_context`. Finished-tool JSON is forwarded so the frontend renders the
  matching visualization card.
- **Reasoning capture.** `step_callback` accumulates each agent's thoughts into a
  per-task collapsible "Show reasoning" panel (available after the fact, not
  streamed token-by-token).
- **Intake confirmation.** A new `confirm_intake` tool runs as the parser's final
  step: the parsed fields surface as an editable form so the user can correct any
  extraction error **before triage**. It is a no-op at the terminal (returns the
  JSON unchanged), so the CLI flow is unaffected.

## Run it

Two processes, from the repo root:

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

## Live API (SSE)

| Method + path | Purpose |
|---|---|
| `POST /api/live/start` `{narrative}` | spawn the crew thread; returns `session_id` |
| `GET  /api/live/stream/{sid}` | **SSE** stream of agent/tool/task events + questions |
| `POST /api/live/answer/{sid}` `{text}` | unblocks the crew thread's pending question |
| `POST /api/live/cancel/{sid}` | abort the session |

Event types: `task_started`, `agent_started`, `agent_message`, `tool_started`,
`tool_finished` (carries parsed tool JSON), `question` (`kind` + `meta`),
`reasoning`, `task_completed`, `final`, `error`, `done`.

## Frontend

Vite + React + TS + Tailwind, light clinical theme. `App.tsx` consumes the SSE
stream into a conversation transcript with a live workflow rail:

- `AgentRoster` — the 6 tasks / 4 agents with active/done highlighting.
- `AgentMessage` / `UserMessage` — chat bubbles (collapsible long reports).
- `ToolActivity` — live "running… → done" chips, expandable to raw args/output.
- `QuestionPrompt` — kind-aware widgets + the editable `intake_form`.
- `RichResultCard` — maps a finished tool's JSON to the reused viz components
  (`AcuityGauge`, `DispositionCard`/`Flip`, `Top3Bars`, `IcdDifferential`,
  `ComparisonPanel`).
- `ReasoningPanel` — collapsible per-task chain-of-thought.

## Retained non-interactive test path (Phase 1)

The earlier stage-by-stage path is kept as a **backend test/utility**, not the
website's flow:

- [`src/proiect_licenta/pipeline.py`](../src/proiect_licenta/pipeline.py) — discrete
  stage functions that call the same tools directly (lifted from the benchmark's
  `run_tool_direct`), plus a one-shot LLM intake parse.
- `webapp/backend/main.py` stage endpoints (`/api/parse`, `/api/triage`, …) and
  `webapp/backend/test_parity.py`.

```bash
# Parity test: the direct-tool pipeline is byte-identical to the crew path.
uv run python webapp/backend/test_parity.py 12
# -> "PARITY OK — web pipeline is byte-identical to the crew path."
```

## Layout

```
webapp/
  backend/
    live.py          SessionChannel, crew-thread runner, event-bus listeners,
                     SSE + start/answer/cancel  (THE LIVE RUNTIME)
    main.py          FastAPI app; includes live.router + retained stage endpoints
    schemas.py       pydantic models for the stage endpoints
    test_parity.py   web(stage)-vs-benchmark parity test
  frontend/          Vite + React + TS + Tailwind SPA
    src/App.tsx      live transcript + workflow rail
    src/api.ts       startLive / EventSource / answer / cancel (+ stage methods)
    src/components/  AgentRoster, AgentMessage, UserMessage, ToolActivity,
                     QuestionPrompt, RichResultCard, ReasoningPanel, + reused
                     AcuityGauge, DispositionCard/Flip, Top3Bars, IcdDifferential,
                     ComparisonPanel
```

The session store is in-memory and process-local (fine for a single-user demo).
