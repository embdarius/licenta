"""Stage-by-stage orchestration of the live inference pipeline.

Drives the four prediction tools the same way the CrewAI crew does, so the web
backend's predictions match the crew path: the same tool instances, the same
argument wiring, the same "disposition gates the reassessment" gating, and the
same MRN lookup override. Only the stdin/LLM interactivity is replaced by
web-native collection. Nothing here touches the trained models; it just calls
the tools.
"""
from __future__ import annotations

import json
import re

# The six snapshot vitals in the tool's expected order. Hardcoded rather than
# imported from case_generation to avoid pulling the heavy MIMIC loaders at
# import time; this list is the stable tool contract.
VITAL_COLS = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp")

_TOOLS: dict = {}


def get_pipeline_tools() -> dict:
    """Construct and cache the five live tool instances.

    Construction loads the joblib artifacts, so build them once and reuse
    across requests.
    """
    if not _TOOLS:
        from proiect_licenta.tools.triage_tool import TriagePredictionTool
        from proiect_licenta.tools.doctor_tool_v3_base import DoctorPredictionToolV3Base
        from proiect_licenta.tools.doctor_disposition_tool import DoctorDispositionTool
        from proiect_licenta.tools.doctor_tool_v3 import DoctorPredictionToolV3
        from proiect_licenta.tools.patient_history_lookup_tool import PatientHistoryLookupTool

        _TOOLS["triage"] = TriagePredictionTool()
        _TOOLS["base"] = DoctorPredictionToolV3Base()
        _TOOLS["dispo"] = DoctorDispositionTool()
        _TOOLS["v3"] = DoctorPredictionToolV3()
        _TOOLS["lookup"] = PatientHistoryLookupTool()
    return _TOOLS


def _vital_arg(v) -> float:
    """Tool vital arg: the measured value, or -1.0 (missing) when absent."""
    return -1.0 if v is None or v == "" else float(v)


def _ems(ems_vitals: dict | None) -> dict:
    """Normalize an EMS-vitals dict to all six keys (missing -> None)."""
    base = {k: None for k in VITAL_COLS}
    if ems_vitals:
        for k in VITAL_COLS:
            if ems_vitals.get(k) is not None:
                base[k] = ems_vitals[k]
    return base


def lookup_blocks(subject_id: int, chief_complaints: str,
                  current_intime: str = "now") -> tuple[str, str]:
    """Return ``(pmh_json, med_json)`` for a returning patient, else ``("","")``.

    A live patient arrives now, so the leakage cutoff is anchored to the
    subject's most recent recorded encounter (the "now" sentinel in
    PatientHistoryLookupTool). Each block is the lookup's pmh_block / med_block
    as a JSON string, ready to pass into the prediction tools as their
    pmh_lookup_json / med_lookup_json overrides.
    """
    if subject_id is None or int(subject_id) < 0:
        return "", ""
    tools = get_pipeline_tools()
    res = json.loads(tools["lookup"]._run(
        subject_id=int(subject_id),
        current_intime=current_intime,
        chief_complaints=chief_complaints,
    ))
    if not res.get("known_patient"):
        return "", ""
    pmh_json = json.dumps(res["pmh_block"]) if res.get("pmh_block") else ""
    med_json = json.dumps(res["med_block"]) if res.get("med_block") else ""
    return pmh_json, med_json


def run_triage(chief_complaints: str, pain_score: int, age: int, gender: str,
               arrival_transport: str, ems_vitals: dict | None = None,
               prior_history: str = "", n_prior_admissions: int = -1,
               pmh_lookup_json: str = "") -> dict:
    """Stage 1 triage: acuity plus screening disposition.

    EMS vitals only matter for ambulance/helicopter patients; pass
    ems_vitals=None for walk-ins (the tool treats -1 as missing). A non-empty
    pmh_lookup_json overrides the self-reported PMH with the real record.
    """
    tools = get_pipeline_tools()
    ems = _ems(ems_vitals)
    return json.loads(tools["triage"]._run(
        chief_complaints=chief_complaints, pain_score=pain_score,
        age=age, gender=gender, arrival_transport=arrival_transport,
        temperature=_vital_arg(ems["temperature"]), heartrate=_vital_arg(ems["heartrate"]),
        resprate=_vital_arg(ems["resprate"]), o2sat=_vital_arg(ems["o2sat"]),
        sbp=_vital_arg(ems["sbp"]), dbp=_vital_arg(ems["dbp"]),
        prior_history=prior_history, n_prior_admissions=n_prior_admissions,
        pmh_lookup_json=pmh_lookup_json,
    ))


def acuity_of(triage_json: dict) -> int:
    return int(triage_json["acuity_prediction"]["predicted_esi_level"])


def triage_admit_of(triage_json: dict) -> bool:
    s = str(triage_json["disposition_prediction"]["prediction"]).upper()
    return "ADMIT" in s and "NOT ADMIT" not in s and "DISCHARGE" not in s


def run_doctor_initial(chief_complaints: str, pain_score: int, age: int,
                       gender: str, arrival_transport: str,
                       predicted_acuity: int, is_admitted: bool) -> dict:
    """Stage 2 initial doctor assessment (v3_base): top-3 diagnosis and department.

    The tool short-circuits to a NOT_ADMITTED payload when is_admitted is False,
    matching the crew's gating.
    """
    tools = get_pipeline_tools()
    return json.loads(tools["base"]._run(
        chief_complaints=chief_complaints, pain_score=pain_score,
        predicted_acuity=predicted_acuity, is_admitted=is_admitted,
        age=age, gender=gender, arrival_transport=arrival_transport,
    ))


def run_disposition(chief_complaints: str, pain_score: int, age: int,
                    gender: str, arrival_transport: str, predicted_acuity: int,
                    triage_is_admitted: bool, nurse: dict,
                    pmh_lookup_json: str = "", med_lookup_json: str = "") -> dict:
    """Stage 4 calibrated disposition refinement with all post-nurse signals.

    ``nurse`` is the dict from build_nurse_payload. Always called, for both
    triage-admit and triage-discharge, because the model can flip either way.
    """
    tools = get_pipeline_tools()
    nv = nurse["vital_signs"]
    traj_json = json.dumps(nurse.get("vital_trajectory") or {})
    return json.loads(tools["dispo"]._run(
        chief_complaints=chief_complaints, pain_score=pain_score,
        predicted_acuity=predicted_acuity, triage_is_admitted=triage_is_admitted,
        age=age, gender=gender, arrival_transport=arrival_transport,
        temperature=_vital_arg(nv["temperature"]), heartrate=_vital_arg(nv["heartrate"]),
        resprate=_vital_arg(nv["resprate"]), o2sat=_vital_arg(nv["o2sat"]),
        sbp=_vital_arg(nv["sbp"]), dbp=_vital_arg(nv["dbp"]),
        rhythm=(nurse.get("rhythm") or ""),
        medications_raw=(nurse.get("medications_raw") or "unknown"),
        prior_history=(nurse.get("prior_history") or ""),
        n_prior_admissions=nurse.get("n_prior_admissions", -1),
        vital_trajectory_json=traj_json, pmh_lookup_json=pmh_lookup_json,
        med_lookup_json=med_lookup_json,
    ))


def refined_admit_of(dispo_json: dict) -> bool:
    return bool(dispo_json["disposition_prediction"]["is_admitted"])


def run_reassessment(chief_complaints: str, pain_score: int, age: int,
                     gender: str, arrival_transport: str, predicted_acuity: int,
                     refined_is_admitted: bool, nurse: dict,
                     pmh_lookup_json: str = "", med_lookup_json: str = "") -> dict:
    """Stage 5 enhanced reassessment (v3 nurse): diagnosis, department, exact-ICD.

    Gated on the refined disposition verdict, not the triage one. The tool
    returns a NOT_ADMITTED payload when False. Same nurse and lookup args as the
    disposition stage.
    """
    tools = get_pipeline_tools()
    nv = nurse["vital_signs"]
    traj_json = json.dumps(nurse.get("vital_trajectory") or {})
    return json.loads(tools["v3"]._run(
        chief_complaints=chief_complaints, pain_score=pain_score,
        predicted_acuity=predicted_acuity, is_admitted=refined_is_admitted,
        age=age, gender=gender, arrival_transport=arrival_transport,
        temperature=_vital_arg(nv["temperature"]), heartrate=_vital_arg(nv["heartrate"]),
        resprate=_vital_arg(nv["resprate"]), o2sat=_vital_arg(nv["o2sat"]),
        sbp=_vital_arg(nv["sbp"]), dbp=_vital_arg(nv["dbp"]),
        medications_raw=(nurse.get("medications_raw") or "unknown"),
        rhythm=(nurse.get("rhythm") or ""),
        prior_history=(nurse.get("prior_history") or ""),
        n_prior_admissions=nurse.get("n_prior_admissions", -1),
        vital_trajectory_json=traj_json, pmh_lookup_json=pmh_lookup_json,
        med_lookup_json=med_lookup_json,
    ))


# Stage 0 LLM intake parse (free text to structured fields). Same lay-to-clinical
# ED-terminology steering as the live parser (tasks.yaml parse_symptoms_task).
_PARSE_PROMPT = """You are a clinical intake parser. Read the patient's free-text \
description and extract ONLY what is stated. Respond with a single JSON object and \
nothing else, using exactly these keys:
  "chief_complaints": short comma-separated string of the presenting complaint(s), \
using STANDARD emergency-department terminology rather than the patient's lay \
wording (this is how the complaint is charted at triage). Map everyday \
descriptions to the clinical term, e.g. "really bad stomach pain" -> "abdominal \
pain"; "can't catch my breath"/"winded" -> "dyspnea"; "throwing up" -> "nausea, \
vomiting"; "passed out" -> "syncope"; "brain bleed" -> "intracranial hemorrhage"; \
"bright red blood in my stool" -> "BRBPR"; "blood in my urine" -> "hematuria"; \
"I'm suicidal" -> "suicidal ideation". Keep EVERY distinct complaint the patient \
mentions (e.g. "chest pain, dyspnea").
  "age": integer years, or null if not stated
  "gender": "m" or "f", or null if not stated
  "arrival_transport": one of "ambulance", "helicopter", or "walk_in". Use "walk_in" \
unless the patient explicitly says they were brought by ambulance or helicopter \
(walk-in is the default; there is no "unknown").
  "pain_score": integer 0-10 if a pain level is stated, else -1
Do not invent vitals, diagnoses, or history. Patient description:
{narrative}"""


def _coerce_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_intake(narrative: str) -> dict:
    """One LLM completion to normalized intake fields (best-effort).

    Returns a dict with chief_complaints, age, gender, arrival_transport, and
    pain_score. The web form pre-fills from this and lets the user correct
    anything. Reuses get_parse_llm so it respects LLM_BACKEND (Flash by default).
    """
    from proiect_licenta.llm_config import get_parse_llm

    llm = get_parse_llm()
    raw = llm.call(_PARSE_PROMPT.format(narrative=narrative))
    s = str(raw).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    parsed = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            parsed = {}

    out = {"chief_complaints": "", "age": None, "gender": None,
           "arrival_transport": "walk_in", "pain_score": -1}

    cc = parsed.get("chief_complaints")
    if isinstance(cc, list):
        cc = ", ".join(str(x) for x in cc)
    if isinstance(cc, str) and cc.strip():
        out["chief_complaints"] = cc.strip()

    age = _coerce_int(parsed.get("age"))
    if age is not None:
        out["age"] = age

    g = str(parsed.get("gender") or "").strip().lower()
    if g in ("m", "f"):
        out["gender"] = g

    # Live parser convention: only ambulance/helicopter are non-default.
    tr = str(parsed.get("arrival_transport") or "").strip().lower().replace(" ", "_")
    out["arrival_transport"] = tr if tr in ("ambulance", "helicopter") else "walk_in"

    ps = _coerce_int(parsed.get("pain_score"))
    if ps is not None:
        out["pain_score"] = ps

    return out
