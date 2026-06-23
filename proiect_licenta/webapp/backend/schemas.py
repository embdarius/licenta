"""Pydantic request/response models for the web inference API.

These describe the wire format between the React frontend and the FastAPI
backend. The prediction *payloads* themselves are returned as raw tool JSON
(``dict``) so the frontend can render the full rich structure without the
backend having to re-declare every nested field.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    narrative: str = Field(..., description="Patient's free-text symptom description.")


class ParseResponse(BaseModel):
    """LLM-prefilled intake fields the user then confirms/corrects in the form."""
    chief_complaints: str = ""
    age: Optional[int] = None
    gender: Optional[str] = None  # "m" | "f" | None
    arrival_transport: str = "walk_in"
    pain_score: int = -1


class EmsVitals(BaseModel):
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None


class TriageRequest(BaseModel):
    """The confirmed intake form. EMS vitals only matter for ambulance/helicopter."""
    session_id: str
    chief_complaints: str
    pain_score: int = -1
    age: int = 50
    gender: str = "unknown"
    arrival_transport: str = "walk_in"
    subject_id: int = -1  # MRN; -1 = first-time / unknown patient
    prior_history: str = ""
    n_prior_admissions: int = -1
    ems_vitals: Optional[EmsVitals] = None


class StageResponse(BaseModel):
    """Generic stage envelope: the raw tool JSON + a few derived flags."""
    session_id: str
    stage: str
    payload: dict[str, Any]
    # Derived conveniences the frontend uses for gating/headlines.
    predicted_acuity: Optional[int] = None
    triage_admit: Optional[bool] = None
    refined_admit: Optional[bool] = None


class NurseReading(BaseModel):
    """One chronological set of vitals + optional rhythm + optional timestamp."""
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    rhythm: Optional[str] = None
    ts: Optional[float] = None  # sort key (minutes); None -> entry order


class NurseRequest(BaseModel):
    session_id: str
    readings: list[NurseReading] = Field(default_factory=list)
    medications_raw: Optional[str] = None
    prior_history: Optional[str] = None
    n_prior_admissions: int = -1


class SessionResponse(BaseModel):
    session_id: str
