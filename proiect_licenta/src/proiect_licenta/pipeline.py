"""Stage-by-stage orchestration of the live inference pipeline.

Single source of truth for driving the four prediction tools the same way the
live CrewAI crew does — extracted from the proven, deterministic wiring in
``benchmarks/benchmark_pipeline_e2e.py`` (``run_tool_direct`` / ``_lookup_blocks``
/ ``_llm_parse_triage``). The web backend (``webapp/backend``) calls these stage
functions one at a time and accumulates session state, so the website's
predictions are byte-identical to the crew's: the SAME tool instances, the SAME
argument wiring, the SAME "disposition gates the reassessment" gating, and the
SAME MRN/EHR-lookup override. Only the stdin/LLM-ask *interactivity* is replaced
by web-native collection.

Nothing here touches the trained models or the tools — it just calls them.
"""
from __future__ import annotations

import json
import re

# The six snapshot vitals, in the tool's expected order. Hardcoded (rather than
# imported from case_generation) to avoid pulling the heavy MIMIC loaders at
# import time; this list is the stable tool contract.
VITAL_COLS = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp")

# ---------------------------------------------------------------------------
# Tool singletons (mirror benchmark _direct_tools)
# ---------------------------------------------------------------------------
_TOOLS: dict = {}


def get_pipeline_tools() -> dict:
    """Lazily construct and cache the five live tool instances.

    Construction loads the joblib model artifacts, so we build them once and
    reuse across requests (same pattern as the benchmark's ``_direct_tools``).
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


# ---------------------------------------------------------------------------
# EHR / MRN lookup (mirror benchmark _lookup_blocks, but with the live 'now'
# anchor instead of a historical intime — matches tasks.yaml step 1b/2b).
# ---------------------------------------------------------------------------
def lookup_blocks(subject_id: int, chief_complaints: str,
                  current_intime: str = "now") -> tuple[str, str]:
    """Return ``(pmh_json, med_json)`` for a returning patient, else ``("","")``.

    A live patient is arriving *now*, so we anchor the leakage cutoff to the
    subject's most recent recorded encounter ("now" sentinel — see
    ``PatientHistoryLookupTool._run``). Each block is the lookup's `pmh_block` /
    `med_block` serialized to a JSON string (empty when not on record / index
    not built), ready to pass straight into the prediction tools as their
    ``pmh_lookup_json`` / ``med_lookup_json`` overrides.
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


# ---------------------------------------------------------------------------
# Stage 1 — Triage (acuity + screening disposition)
# ---------------------------------------------------------------------------
def run_triage(chief_complaints: str, pain_score: int, age: int, gender: str,
               arrival_transport: str, ems_vitals: dict | None = None,
               prior_history: str = "", n_prior_admissions: int = -1,
               pmh_lookup_json: str = "") -> dict:
    """``TriagePredictionTool`` with EMS vitals + PMH self-report / EHR override.

    EMS vitals are only meaningful for ambulance/helicopter patients; pass
    ``ems_vitals=None`` for walk-ins (the tool treats -1 as missing/masked).
    ``pmh_lookup_json`` (when non-empty) OVERRIDES the self-reported PMH with the
    real prior-encounter record. Mirrors benchmark lines 236-244.
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


# ---------------------------------------------------------------------------
# Stage 2 — Doctor initial assessment (v3_base, pre-nurse)
# ---------------------------------------------------------------------------
def run_doctor_initial(chief_complaints: str, pain_score: int, age: int,
                       gender: str, arrival_transport: str,
                       predicted_acuity: int, is_admitted: bool) -> dict:
    """``DoctorPredictionToolV3Base`` — initial top-3 diagnosis + department.

    The tool itself short-circuits to a NOT_ADMITTED payload when
    ``is_admitted`` is False, matching the live crew's gating.
    """
    tools = get_pipeline_tools()
    return json.loads(tools["base"]._run(
        chief_complaints=chief_complaints, pain_score=pain_score,
        predicted_acuity=predicted_acuity, is_admitted=is_admitted,
        age=age, gender=gender, arrival_transport=arrival_transport,
    ))


# ---------------------------------------------------------------------------
# Stage 4 — Doctor disposition refinement (calibrated, post-nurse)
# ---------------------------------------------------------------------------
def run_disposition(chief_complaints: str, pain_score: int, age: int,
                    gender: str, arrival_transport: str, predicted_acuity: int,
                    triage_is_admitted: bool, nurse: dict,
                    pmh_lookup_json: str = "", med_lookup_json: str = "") -> dict:
    """``DoctorDispositionTool`` with all post-nurse signals.

    ``nurse`` is the dict from ``build_nurse_payload`` (vital_signs,
    vital_trajectory, rhythm, medications_raw, prior_history,
    n_prior_admissions). Always called — both for triage-admit and
    triage-discharge — because the model can flip either way. Mirrors benchmark
    lines 253-264.
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


# ---------------------------------------------------------------------------
# Stage 5 — Doctor enhanced reassessment (v3 nurse). Gated on REFINED admit.
# ---------------------------------------------------------------------------
def run_reassessment(chief_complaints: str, pain_score: int, age: int,
                     gender: str, arrival_transport: str, predicted_acuity: int,
                     refined_is_admitted: bool, nurse: dict,
                     pmh_lookup_json: str = "", med_lookup_json: str = "") -> dict:
    """``DoctorPredictionToolV3`` — enhanced diagnosis + department + exact-ICD.

    Gated on the REFINED disposition verdict (``refined_is_admitted``), NOT the
    triage one. The tool returns a NOT_ADMITTED payload when False. Same nurse +
    lookup args as the disposition stage. Mirrors benchmark lines 269-280.
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


# ---------------------------------------------------------------------------
# Stage 0 — LLM intake parse (free text -> structured fields)
# ---------------------------------------------------------------------------
# Same lay->clinical ED-terminology steering as the live parser (tasks.yaml
# parse_symptoms_task) and the benchmark's parser-llm mode (_PARSE_PROMPT).
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
    """One direct LLM completion -> normalized intake fields (best-effort).

    Returns a dict with keys ``chief_complaints`` (str), ``age`` (int|None),
    ``gender`` ("m"/"f"/None), ``arrival_transport`` (walk_in/ambulance/
    helicopter), ``pain_score`` (int, -1 if unstated). The web form pre-fills
    from this and lets the user correct anything. Reuses ``get_parse_llm`` so it
    respects ``LLM_BACKEND`` (Flash by default). Normalization matches the
    benchmark's ``run_parser_llm_bypass`` (lines 454-474).
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
