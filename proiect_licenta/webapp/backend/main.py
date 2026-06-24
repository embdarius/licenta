"""FastAPI backend for the interactive live-inference website.

Drives the same prediction tools the live CrewAI crew uses, one pipeline stage
per endpoint, via ``proiect_licenta.pipeline``. Predictions are byte-identical
to the crew (see ``webapp/backend/test_parity.py``); only the stdin/LLM-ask
interactivity is replaced by web-native forms.

Run (from the repo root):
    uv run uvicorn webapp.backend.main:app --reload --port 8000
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from proiect_licenta import pipeline
from proiect_licenta.tools.nurse_tool import build_nurse_payload

from webapp.backend import live
from webapp.backend.schemas import (
    NurseRequest, ParseRequest, ParseResponse, SessionResponse,
    StageResponse, TriageRequest,
)

app = FastAPI(title="ED Decision-Support — Live Inference")

# Live conversational runtime (drives the real agentic crew over SSE). The
# stage endpoints below remain as a non-interactive test path (pipeline.py).
app.include_router(live.router)

# Dev CORS: the Vite dev server proxies /api, but allow direct cross-origin too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store. A session accumulates state across stages exactly as
# the crew passes context task-to-task. Process-local + ephemeral — fine for a
# single-user demo; swap for Redis if this ever needs to scale.
# ---------------------------------------------------------------------------
_SESSIONS: dict[str, dict] = {}


def _session(session_id: str) -> dict:
    s = _SESSIONS.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. POST /api/session first.")
    return s


@app.on_event("startup")
def _warm_models() -> None:
    """Load the joblib model artifacts once at startup so the first request is fast,
    and register the live event-bus listeners."""
    pipeline.get_pipeline_tools()
    live.register_listeners()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    sid = uuid.uuid4().hex
    _SESSIONS[sid] = {"core": None, "nurse": None}
    return SessionResponse(session_id=sid)


@app.post("/api/parse", response_model=ParseResponse)
def parse(req: ParseRequest) -> ParseResponse:
    """Stage 0 — LLM intake parse (free text -> prefilled, editable fields)."""
    if not req.narrative.strip():
        raise HTTPException(status_code=400, detail="Empty narrative.")
    fields = pipeline.parse_intake(req.narrative)
    return ParseResponse(**fields)


@app.post("/api/triage", response_model=StageResponse)
def triage(req: TriageRequest) -> StageResponse:
    """Stage 1 — EHR lookup (returning patients) + triage acuity/disposition."""
    s = _session(req.session_id)

    # MRN/EHR lookup: a returning patient's real prior-encounter PMH + med blocks
    # OVERRIDE self-report downstream (matches tasks.yaml step 1b/2b).
    pmh_json, med_json = pipeline.lookup_blocks(req.subject_id, req.chief_complaints)

    ems = req.ems_vitals.model_dump() if req.ems_vitals else None
    # EMS vitals are only available for ambulance/helicopter at triage.
    if req.arrival_transport not in ("ambulance", "helicopter"):
        ems = None

    triage_j = pipeline.run_triage(
        chief_complaints=req.chief_complaints, pain_score=req.pain_score,
        age=req.age, gender=req.gender, arrival_transport=req.arrival_transport,
        ems_vitals=ems, prior_history=req.prior_history,
        n_prior_admissions=req.n_prior_admissions, pmh_lookup_json=pmh_json,
    )
    acuity = pipeline.acuity_of(triage_j)
    triage_admit = pipeline.triage_admit_of(triage_j)

    # Persist the patient "core" so downstream stages only need the session_id.
    s["core"] = {
        "subject_id": req.subject_id,
        "chief_complaints": req.chief_complaints,
        "pain_score": req.pain_score,
        "age": req.age,
        "gender": req.gender,
        "arrival_transport": req.arrival_transport,
        "predicted_acuity": acuity,
        "triage_admit": triage_admit,
        "pmh_json": pmh_json,
        "med_json": med_json,
    }
    return StageResponse(
        session_id=req.session_id, stage="triage", payload=triage_j,
        predicted_acuity=acuity, triage_admit=triage_admit,
    )


@app.post("/api/doctor-initial", response_model=StageResponse)
def doctor_initial(session_id: str) -> StageResponse:
    """Stage 2 — initial doctor assessment (v3_base), gated on triage admit."""
    s = _session(session_id)
    c = s["core"]
    if c is None:
        raise HTTPException(status_code=409, detail="Run /api/triage first.")
    payload = pipeline.run_doctor_initial(
        chief_complaints=c["chief_complaints"], pain_score=c["pain_score"],
        age=c["age"], gender=c["gender"], arrival_transport=c["arrival_transport"],
        predicted_acuity=c["predicted_acuity"], is_admitted=c["triage_admit"],
    )
    return StageResponse(
        session_id=session_id, stage="doctor_initial", payload=payload,
        predicted_acuity=c["predicted_acuity"], triage_admit=c["triage_admit"],
    )


@app.post("/api/nurse", response_model=StageResponse)
def nurse(req: NurseRequest) -> StageResponse:
    """Stage 3 — nurse multi-reading collection (web-native, same payload shape)."""
    s = _session(req.session_id)
    if s["core"] is None:
        raise HTTPException(status_code=409, detail="Run /api/triage first.")

    # Normalize skip-equivalent free text to None, matching the stdin tool.
    def _blank(v):
        return None if v is None or str(v).strip() == "" else v

    rounds = [r.model_dump() for r in req.readings]
    payload = build_nurse_payload(
        rounds, _blank(req.medications_raw), _blank(req.prior_history),
        req.n_prior_admissions,
    )
    s["nurse"] = payload
    return StageResponse(session_id=req.session_id, stage="nurse", payload=payload)


@app.post("/api/disposition", response_model=StageResponse)
def disposition(session_id: str) -> StageResponse:
    """Stage 4 — calibrated disposition refinement (the before/after-nurse flip)."""
    s = _session(session_id)
    c = s["core"]
    n = s["nurse"]
    if c is None:
        raise HTTPException(status_code=409, detail="Run /api/triage first.")
    if n is None:
        raise HTTPException(status_code=409, detail="Run /api/nurse first.")
    payload = pipeline.run_disposition(
        chief_complaints=c["chief_complaints"], pain_score=c["pain_score"],
        age=c["age"], gender=c["gender"], arrival_transport=c["arrival_transport"],
        predicted_acuity=c["predicted_acuity"], triage_is_admitted=c["triage_admit"],
        nurse=n, pmh_lookup_json=c["pmh_json"], med_lookup_json=c["med_json"],
    )
    refined = pipeline.refined_admit_of(payload)
    c["refined_admit"] = refined
    return StageResponse(
        session_id=session_id, stage="disposition", payload=payload,
        predicted_acuity=c["predicted_acuity"], triage_admit=c["triage_admit"],
        refined_admit=refined,
    )


@app.post("/api/reassessment", response_model=StageResponse)
def reassessment(session_id: str) -> StageResponse:
    """Stage 5 — enhanced diagnosis/department, gated on the REFINED disposition."""
    s = _session(session_id)
    c = s["core"]
    n = s["nurse"]
    if c is None or n is None:
        raise HTTPException(status_code=409, detail="Run /api/triage and /api/nurse first.")
    if "refined_admit" not in c:
        raise HTTPException(status_code=409, detail="Run /api/disposition first.")
    payload = pipeline.run_reassessment(
        chief_complaints=c["chief_complaints"], pain_score=c["pain_score"],
        age=c["age"], gender=c["gender"], arrival_transport=c["arrival_transport"],
        predicted_acuity=c["predicted_acuity"], refined_is_admitted=c["refined_admit"],
        nurse=n, pmh_lookup_json=c["pmh_json"], med_lookup_json=c["med_json"],
    )
    return StageResponse(
        session_id=session_id, stage="reassessment", payload=payload,
        predicted_acuity=c["predicted_acuity"], triage_admit=c["triage_admit"],
        refined_admit=c["refined_admit"],
    )
