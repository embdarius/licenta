"""
End-to-End Pipeline Benchmark — synthetic NL cases vs the tabular benchmark
===========================================================================

Runs the synthetic cases produced by `uv run generate_cases` through THREE
prediction modes on the SAME stay_ids and scores each against MIMIC ground
truth:

  1. E2E (real crew)      — feed only the generated free-text narrative into
                            the full ProiectLicenta crew (NLP parser + triage +
                            doctor v3_base + nurse + disposition + reassessment).
                            The interactive tools are monkeypatched to answer
                            from the case's structured fields; the 4 prediction
                            tools are wrapped to tee their exact JSON output.
  2. Tool-direct          — call the same crew tools directly with the exact
                            tabular field values (no LLM), replicating the
                            crew's gating in Python. Isolates the NL layer.
  3. Feature-vector       — predict from the cached build_features rows + the
                            trained models (the existing benchmark methodology),
                            restricted to the 20 stay_ids.

Targets scored: ESI acuity, refined disposition (admit/discharge), diagnosis
top-1/top-3, department top-1/top-3, and the Stage-2 exact-ICD resolver
(doctor+nurse, 3-char rollup): exact @1/@5, flat@10, and union — comparable to the
E2E flat-10 / E2E union metrics in benchmark_icd_resolution.py (the runtime tool
resolves over top-5 cats / k_per_cat=5 / k_flat=10, the same parameters). The
exact-ICD candidates come from the v3 tool's `exact_diagnoses` block (E2E +
tool-direct) or from running the resolver on the cached features (feature-vector).
Diagnosis/department/exact-ICD are scored over the admitted ground-truth cases
(with a coverage note for cases the pipeline routed to discharge). Also prints a
per-case dump and an NL-fidelity report (what the parser extracted vs tabular truth).

Usage:
    uv run python benchmarks/benchmark_pipeline_e2e.py [--limit N] [--skip-feature-vector] [--skip-e2e]

LLM backend comparison (Flash vs MedGemma):
    # Flash (default) — full agentic crew, unchanged baseline:
    uv run python benchmarks/benchmark_pipeline_e2e.py --dump-json artifacts/benchmarks/e2e_flash.json
    # MedGemma — full agentic crew if it can tool-call, else add the bypass mode:
    uv run python benchmarks/benchmark_pipeline_e2e.py --llm-backend medgemma --parser-llm \
        --dump-json artifacts/benchmarks/e2e_medgemma.json
The orchestration-independent NL-FIDELITY section is the fair parser-quality
anchor across backends (printed for both the E2E and parser-llm modes).
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import joblib

# Force UTF-8 stdout/stderr before crewai/tool imports wrap the streams — the
# loaders + LLM narratives emit non-cp1252 chars that crash the Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import (
    ARTIFACTS_DIR, DOCTOR_V3_DIR, DOCTOR_V3_ICD_RESOLVER_DIR, DIAGNOSIS_CSV,
    TRIAGE_V1_DIR,
)
from proiect_licenta.preprocessing import (
    normalize_complaint_text, clinicalize_complaint,
)
from proiect_licenta.case_generation import load_cases, load_feature_cache, VITAL_COLS
from proiect_licenta import icd_resolution as icdr
from proiect_licenta.tools.doctor_tool_v3_base import DEPARTMENT_NAMES  # noqa: F401

# Tool classes (patched for E2E capture; instantiated for tool-direct)
from proiect_licenta.tools import triage_tool, doctor_tool_v3_base, \
    doctor_disposition_tool, doctor_tool_v3, ask_patient_tool, nurse_tool
from proiect_licenta.tools.triage_tool import TriagePredictionTool
from proiect_licenta.tools.doctor_tool_v3_base import DoctorPredictionToolV3Base
from proiect_licenta.tools.doctor_disposition_tool import (
    DoctorDispositionTool, DECISION_THRESHOLD,
)
from proiect_licenta.tools.doctor_tool_v3 import DoctorPredictionToolV3
from proiect_licenta.tools.ask_patient_tool import AskPatientTool
from proiect_licenta.tools.nurse_tool import NurseDataCollectionTool
from proiect_licenta.tools.patient_history_lookup_tool import (
    PatientHistoryLookupTool, get_history_index,
)


# ===========================================================================
# Shared helpers
# ===========================================================================
def print_section(title):
    print(f"\n{'='*78}\n  {title}\n{'='*78}")


def _to_admit(text_or_bool) -> bool:
    if isinstance(text_or_bool, bool):
        return text_or_bool
    s = str(text_or_bool).upper()
    return "ADMIT" in s and "NOT ADMIT" not in s and "DISCHARGE" not in s


def _diag_top3(tool_json) -> list:
    return [d["category"] for d in tool_json["diagnosis_prediction"]["top_3_categories"]]


def _dept_top3(tool_json) -> list:
    return [d["code"] for d in tool_json["department_prediction"]["top_3_departments"]]


def _vital_arg(v):
    """Tool vital arg: measured value, or -1.0 if the patient didn't have it."""
    return -1.0 if v is None else float(v)


def _exact_icd_from_block(block) -> dict:
    """Normalize a tool's `exact_diagnoses` JSON to the common scoring shape
    ``{"flat": [rollup_code, ...] (ordered), "union": {rollup_code, ...}}``.
    Returns None when no resolver block was produced."""
    if not block:
        return None
    flat = [d["icd_3char"] for d in block.get("flat_ranking", [])]
    union = {
        c["icd_3char"]
        for pc in block.get("top_per_category", [])
        for c in pc.get("candidates", [])
    }
    return {"flat": flat, "union": union}


def _exact_icd_from_resolved(resolved) -> dict:
    """Same as `_exact_icd_from_block` but for the resolver's raw return dict
    (keys `flat_top` / `per_category`), used by the feature-vector mode."""
    if not resolved:
        return None
    flat = [d["code"] for d in resolved.get("flat_top", [])]
    union = {
        c["code"]
        for pc in resolved.get("per_category", [])
        for c in pc.get("codes", [])
    }
    return {"flat": flat, "union": union}


def true_rollups(stay_ids) -> dict:
    """Map each stay_id -> its primary-diagnosis 3-char ICD rollup, derived from
    MIMIC (the cases carry only diagnosis_group, not the exact code). Mirrors
    `_attach_icd_version` in benchmark_icd_resolution.py."""
    import pandas as pd
    want = {int(s) for s in stay_ids}
    diag = pd.read_csv(DIAGNOSIS_CSV, dtype={"icd_code": str, "icd_version": str})
    diag = diag[(diag["seq_num"] == 1) & (diag["stay_id"].isin(want))]
    diag = diag.drop_duplicates("stay_id")
    out = {}
    for _, r in diag.iterrows():
        code = str(r["icd_code"] or "").strip().upper()
        ver = str(r["icd_version"] or "").strip()
        out[int(r["stay_id"])] = icdr.rollup_icd(code, ver)
    return out


# ===========================================================================
# Mode 2 — TOOL-DIRECT (exact tabular values, crew gating replicated, no LLM)
# ===========================================================================
_DIRECT_TOOLS = {}


def _direct_tools():
    if not _DIRECT_TOOLS:
        _DIRECT_TOOLS["triage"] = TriagePredictionTool()
        _DIRECT_TOOLS["base"] = DoctorPredictionToolV3Base()
        _DIRECT_TOOLS["dispo"] = DoctorDispositionTool()
        _DIRECT_TOOLS["v3"] = DoctorPredictionToolV3()
        _DIRECT_TOOLS["lookup"] = PatientHistoryLookupTool()
    return _DIRECT_TOOLS


def _lookup_blocks(case) -> tuple:
    """Call the PatientHistoryLookupTool with the case's REAL subject_id + intime
    and return ``(pmh_json, med_json)`` — the `pmh_block` and `med_block` as JSON
    strings (each "" if unknown / not on record / index not built). This is what
    an EHR lookup for a returning patient would supply at runtime."""
    # The benchmark reconstructs a HISTORICAL visit, so it must pass that stay's
    # exact intime as the leakage cutoff — never the 'now' sentinel (which would
    # anchor to the subject's latest encounter and could include this stay).
    # If the case bundle predates the `intime` field, skip the lookup entirely.
    intime = str(case.get("intime", "")).strip()
    if not intime:
        return "", ""
    tools = _direct_tools()
    res = json.loads(tools["lookup"]._run(
        subject_id=int(case.get("subject_id", -1)),
        current_intime=intime,
        chief_complaints=case["triage_inputs"]["chief_complaints"],
    ))
    if not res.get("known_patient"):
        return "", ""
    pmh_json = json.dumps(res["pmh_block"]) if res.get("pmh_block") else ""
    med_json = json.dumps(res["med_block"]) if res.get("med_block") else ""
    return pmh_json, med_json


def run_tool_direct(case, pmh_lookup_json: str = "", med_lookup_json: str = "") -> dict:
    """Replicate the crew pipeline with direct tool calls on exact fields.

    `pmh_lookup_json` / `med_lookup_json` (empty by default) feed the tools the
    real prior-encounter PMH block (triage + disposition + v3) and reconciled
    home-med block (disposition + v3) from an EHR lookup; with "" they use the
    ask-the-patient self-report path. The two settings give the no-lookup and
    with-lookup benchmark columns. (Triage has no medication features, so
    med_lookup_json is not passed to the triage call.)"""
    t = case["triage_inputs"]
    n = case["nurse_inputs"]
    tools = _direct_tools()
    out = {"acuity": None, "triage_admit": None, "refined_admit": None,
           "diag_top3": None, "dept_top3": None, "exact_icd": None}

    # EMS vitals only for ambulance/helicopter (matches live triage).
    ems = t["ems_vitals"] or {v: None for v in VITAL_COLS}

    # Triage also receives the EHR lookup block (when present) so its
    # acuity/disposition models use a returning patient's real prior-encounter
    # numerics instead of zero-filling — matching the live crew, where the
    # triage agent calls patient_history_lookup_tool itself (tasks.yaml step 1b).
    # With "" (the plain tool_direct column) it falls back to self-report.
    triage_j = json.loads(tools["triage"]._run(
        chief_complaints=t["chief_complaints"], pain_score=t["pain_score"],
        age=t["age"], gender=t["gender"], arrival_transport=t["arrival_transport"],
        temperature=_vital_arg(ems["temperature"]), heartrate=_vital_arg(ems["heartrate"]),
        resprate=_vital_arg(ems["resprate"]), o2sat=_vital_arg(ems["o2sat"]),
        sbp=_vital_arg(ems["sbp"]), dbp=_vital_arg(ems["dbp"]),
        prior_history=t["prior_history"], n_prior_admissions=t["n_prior_admissions"],
        pmh_lookup_json=pmh_lookup_json,
    ))
    out["acuity"] = int(triage_j["acuity_prediction"]["predicted_esi_level"])
    out["triage_admit"] = _to_admit(triage_j["disposition_prediction"]["prediction"])

    nv = n["vitals"]
    # Real multi-reading trajectory the runtime would have if it collected
    # several readings. Passed as the doctor tools' vital_trajectory_json so
    # they build real longitudinal features instead of the snapshot fallback.
    traj_json = json.dumps(n.get("vital_trajectory") or {})
    dispo_j = json.loads(tools["dispo"]._run(
        chief_complaints=t["chief_complaints"], pain_score=t["pain_score"],
        predicted_acuity=out["acuity"], triage_is_admitted=out["triage_admit"],
        age=t["age"], gender=t["gender"], arrival_transport=t["arrival_transport"],
        temperature=_vital_arg(nv["temperature"]), heartrate=_vital_arg(nv["heartrate"]),
        resprate=_vital_arg(nv["resprate"]), o2sat=_vital_arg(nv["o2sat"]),
        sbp=_vital_arg(nv["sbp"]), dbp=_vital_arg(nv["dbp"]),
        rhythm=n["rhythm"], medications_raw=(n["medications_raw"] or "unknown"),
        prior_history=n["prior_history"], n_prior_admissions=n["n_prior_admissions"],
        vital_trajectory_json=traj_json, pmh_lookup_json=pmh_lookup_json,
        med_lookup_json=med_lookup_json,
    ))
    out["refined_admit"] = bool(dispo_j["disposition_prediction"]["is_admitted"])

    # Diagnosis/department reassessment gates on the REFINED verdict.
    if out["refined_admit"]:
        v3_j = json.loads(tools["v3"]._run(
            chief_complaints=t["chief_complaints"], pain_score=t["pain_score"],
            predicted_acuity=out["acuity"], is_admitted=True,
            age=t["age"], gender=t["gender"], arrival_transport=t["arrival_transport"],
            temperature=_vital_arg(nv["temperature"]), heartrate=_vital_arg(nv["heartrate"]),
            resprate=_vital_arg(nv["resprate"]), o2sat=_vital_arg(nv["o2sat"]),
            sbp=_vital_arg(nv["sbp"]), dbp=_vital_arg(nv["dbp"]),
            medications_raw=(n["medications_raw"] or "unknown"), rhythm=n["rhythm"],
            prior_history=n["prior_history"], n_prior_admissions=n["n_prior_admissions"],
            vital_trajectory_json=traj_json, pmh_lookup_json=pmh_lookup_json,
            med_lookup_json=med_lookup_json,
        ))
        out["diag_top3"] = _diag_top3(v3_j)
        out["dept_top3"] = _dept_top3(v3_j)
        out["exact_icd"] = _exact_icd_from_block(
            v3_j["diagnosis_prediction"].get("exact_diagnoses"))
    return out


def run_tool_direct_lookup(case) -> dict:
    """tool-direct WITH the EHR history lookup: fetch the real prior-encounter
    PMH block AND prior reconciled home-med block by subject_id and feed them to
    the triage (PMH only) + disposition + v3 tools. The gap vs the plain
    tool_direct column = the value of EHR access for returning patients."""
    pmh_json, med_json = _lookup_blocks(case)
    return run_tool_direct(case, pmh_lookup_json=pmh_json, med_lookup_json=med_json)


# ===========================================================================
# Mode — PARSER-LLM bypass (MedGemma-only fallback for broken agentic tool-calling)
# ===========================================================================
# When an LLM can't reliably drive CrewAI's agentic tool-calling (the main risk
# for a non-function-calling medical model like MedGemma), this mode isolates the
# part we actually want to compare — clinical NL *parsing* — by calling the LLM
# as a single direct completion for the parse step only, then feeding the parsed
# fields into the deterministic, LLM-free tool-direct path. The interactive
# fields the live crew would collect via (monkeypatched) tools — EMS vitals, PMH,
# prior-admission count — are kept from the case, exactly as in the E2E run, so
# only the free-text-parsed fields vary. Gemini is never run here (it always
# drives the full agentic crew); this is MedGemma's evaluation path.
# Two prompt variants so the clinical-term steering (Track 1) can be measured in
# isolation: `--plain-prompt` selects _PARSE_PROMPT_PLAIN (reproduces the
# documented n150 baseline); the default _PARSE_PROMPT adds the lay->clinical
# steering. Combined with `--clinicalize`, this gives a clean 2x2 attribution
# (prompt on/off x deterministic map on/off).
_PARSE_PROMPT_PLAIN = """You are a clinical intake parser. Read the patient's free-text \
description and extract ONLY what is stated. Respond with a single JSON object and \
nothing else, using exactly these keys:
  "chief_complaints": short comma-separated string of the presenting complaint(s)
  "age": integer years, or null if not stated
  "gender": "m" or "f", or null if not stated
  "arrival_transport": one of "ambulance", "helicopter", or "walk_in". Use "walk_in" \
unless the patient explicitly says they were brought by ambulance or helicopter \
(walk-in is the default; there is no "unknown").
  "pain_score": integer 0-10 if a pain level is stated, else -1
Do not invent vitals, diagnoses, or history. Patient description:
{narrative}"""

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

# The prompt actually used for parsing (and for the cache key). main() may swap
# this to the plain variant via --plain-prompt.
_ACTIVE_PARSE_PROMPT = _PARSE_PROMPT


def _coerce_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --- Parse cache -----------------------------------------------------------
# The LLM parse of a narrative is deterministic (temperature=0) and depends only
# on the prompt + narrative, so we cache it keyed by (prompt hash, narrative).
# This lets system-side iterations (e.g. toggling the lay->clinical clinicalize
# map, which is applied AFTER the parse) re-run with ZERO LLM calls. A prompt
# change alters the key automatically, so the cache self-invalidates per prompt
# revision. Cache file is per backend.
import hashlib

_PARSE_CACHE: dict = {}
_PARSE_CACHE_PATH = None
_PARSE_CACHE_DIRTY = False
_PARSE_CACHE_HITS = 0
_PARSE_CACHE_MISSES = 0


def _parse_cache_key(narrative: str) -> str:
    h = hashlib.sha1()
    h.update(_ACTIVE_PARSE_PROMPT.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(narrative).encode("utf-8"))
    return h.hexdigest()


def load_parse_cache(backend: str) -> None:
    global _PARSE_CACHE, _PARSE_CACHE_PATH
    _PARSE_CACHE_PATH = ARTIFACTS_DIR / "benchmarks" / f"parse_cache_{backend}.json"
    if _PARSE_CACHE_PATH.exists():
        try:
            _PARSE_CACHE = json.loads(_PARSE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _PARSE_CACHE = {}
    else:
        _PARSE_CACHE = {}


def save_parse_cache() -> None:
    global _PARSE_CACHE_DIRTY
    if _PARSE_CACHE_PATH is None or not _PARSE_CACHE_DIRTY:
        return
    _PARSE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write so a kill mid-write can't corrupt the cache.
    tmp = _PARSE_CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_PARSE_CACHE, indent=0), encoding="utf-8")
    tmp.replace(_PARSE_CACHE_PATH)
    _PARSE_CACHE_DIRTY = False


def _llm_parse_triage(narrative: str, llm) -> dict:
    """One direct LLM completion -> structured triage fields (best-effort).

    Cached by (prompt, narrative); a cache hit makes no LLM call.
    """
    global _PARSE_CACHE_DIRTY, _PARSE_CACHE_HITS, _PARSE_CACHE_MISSES
    key = _parse_cache_key(narrative)
    if key in _PARSE_CACHE:
        _PARSE_CACHE_HITS += 1
        return _PARSE_CACHE[key]
    _PARSE_CACHE_MISSES += 1
    raw = llm.call(_ACTIVE_PARSE_PROMPT.format(narrative=narrative))
    s = str(raw).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = {}
    if obj:  # only cache successful parses, so failures can be retried later
        _PARSE_CACHE[key] = obj
        _PARSE_CACHE_DIRTY = True
        # Flush incrementally so a killed run (e.g. machine sleep) loses at most
        # a few parses; the next run resumes cheaply from the cache.
        if _PARSE_CACHE_MISSES % 20 == 0:
            save_parse_cache()
    return obj


def run_parser_llm_bypass(case: dict, llm, clinicalize: bool = False) -> dict:
    """Parse the narrative with a direct LLM call, then score via tool-direct.
    Overrides only the free-text-parsed triage fields; everything else (EMS
    vitals, PMH, prior-admission count, all nurse inputs) is kept from the case,
    matching how the E2E crew sources them from interactive (monkeypatched) tools.

    When ``clinicalize`` is True, the parsed chief complaint is passed through the
    deterministic lay->clinical map (``clinicalize_complaint``) before it reaches
    the tools. This is the only difference from the baseline parser-llm run, so
    the accuracy delta isolates the map's value. The map is applied ONLY to the
    parsed free text here — never to the tabular complaint in ``tool_direct`` —
    so the reference modes stay clean."""
    parsed = _llm_parse_triage(case["narrative"], llm)
    ti = dict(case["triage_inputs"])

    cc = parsed.get("chief_complaints")
    if isinstance(cc, list):
        cc = ", ".join(str(x) for x in cc)
    cc_raw = cc.strip() if isinstance(cc, str) else ""
    if cc_raw:
        ti["chief_complaints"] = clinicalize_complaint(cc_raw) if clinicalize else cc_raw

    age = _coerce_int(parsed.get("age"))
    if age is not None:
        ti["age"] = age
    g = str(parsed.get("gender") or "").strip().lower()
    if g in ("m", "f"):
        ti["gender"] = g
    # Live parser convention (tasks.yaml): only ambulance/helicopter are non-default;
    # anything else (incl. unstated) is walk_in. No "unknown" category exists, so we
    # don't offer the LLM one (offering it penalised the more literal model).
    tr = str(parsed.get("arrival_transport") or "").strip().lower().replace(" ", "_")
    ti["arrival_transport"] = tr if tr in ("ambulance", "helicopter") else "walk_in"
    ps = _coerce_int(parsed.get("pain_score"))
    if ps is not None:
        ti["pain_score"] = ps

    case2 = dict(case)
    case2["triage_inputs"] = ti
    # Feed the SAME EHR lookup (prior-encounter PMH + reconciled home meds) that
    # the feature_vector path's build_features draws on, so the ONLY feature that
    # differs between this mode and feature_vector is the LLM-parsed chief
    # complaint — a fair like-for-like parser comparison (equal feature sets).
    # The lookup uses the original case (its same-complaint-as-prior signal stays
    # on the tabular complaint, keeping PMH/med parity exact with feature_vector).
    pmh_json, med_json = _lookup_blocks(case)
    out = run_tool_direct(case2, pmh_lookup_json=pmh_json, med_lookup_json=med_json)
    # Surface parser output for the NL-fidelity report (same shape as E2E).
    # `complaints` is what the tools saw (clinicalized when enabled); `complaints_raw`
    # is the verbatim LLM parse, so the report can measure the map's coverage.
    out["parser"] = {
        "complaints": [ti["chief_complaints"]],
        "complaints_raw": [cc_raw] if cc_raw else [ti["chief_complaints"]],
        "age": ti["age"], "gender": ti["gender"],
        "arrival_transport": ti["arrival_transport"], "pain": ti["pain_score"],
    }
    return out


# ===========================================================================
# Mode 3 — FEATURE-VECTOR (cached build_features rows + trained models)
# ===========================================================================
_FV = {}


def _fv_models():
    if not _FV:
        _FV["dispo"] = joblib.load(DOCTOR_V3_DIR / "disposition_model.joblib")
        _FV["diag"] = joblib.load(DOCTOR_V3_DIR / "diagnosis_model.joblib")
        _FV["dept"] = joblib.load(DOCTOR_V3_DIR / "department_model.joblib")
        meta = json.loads((DOCTOR_V3_DIR / "metadata.json").read_text(encoding="utf-8"))
        _FV["diag_labels"] = meta["diagnosis_labels"]
        _FV["dept_labels"] = meta["department_labels"]
        _FV["cascade_cols"] = meta.get("diag_cascade_cols")
        # Stage-2 exact-ICD resolver + the shared TF-IDF vectorizer, so the
        # feature-vector column scores exact-ICD the same way the live tool does.
        try:
            _FV["resolver"] = icdr.load_resolver(DOCTOR_V3_ICD_RESOLVER_DIR)
            _FV["tfidf"] = joblib.load(TRIAGE_V1_DIR / "tfidf_vectorizer.joblib")
        except (FileNotFoundError, OSError):
            _FV["resolver"] = None
            _FV["tfidf"] = None
    return _FV


def _fv_exact_icd(m, case, Xn, diag_proba) -> dict:
    """Feature-vector exact-ICD: run the Stage-2 resolver on the FV diagnosis
    softmax + the cached physio/PMH columns, mirroring the runtime tool's path
    selection (gated-PMH -> vitals -> text+prevalence). Returns the common
    {"flat","union"} shape, or None when the resolver isn't available."""
    resolver = m.get("resolver")
    if resolver is None:
        return None
    gindex = resolver["granularities"]["rollup"]
    top5 = np.argsort(diag_proba)[::-1][:5]
    cat_probs = [(m["diag_labels"][i], float(diag_proba[i])) for i in top5]
    q_vec = icdr.vectorize_query(
        normalize_complaint_text(case["triage_inputs"]["chief_complaints"]), m["tfidf"])
    weights = resolver.get("weights")
    weights_pmh = resolver.get("weights_pmh")
    standardizer = resolver.get("standardizer")
    pmh_standardizer = resolver.get("pmh_standardizer")
    physio_q = (icdr.physio_matrix(Xn, standardizer)[0]
                if standardizer is not None else None)
    has_pmh = ("no_history" in Xn.columns) and (int(Xn["no_history"].iloc[0]) == 0)
    if has_pmh and weights_pmh and pmh_standardizer is not None and physio_q is not None:
        pmh_q = icdr.physio_matrix(Xn, pmh_standardizer)[0]
        resolved = icdr.resolve_exact_diagnoses_v3(
            cat_probs, q_vec, physio_q, pmh_q, True, gindex,
            weights_pmh, weights, k_per_cat=5, k_flat=10)
    elif physio_q is not None and weights:
        resolved = icdr.resolve_exact_diagnoses_v2(
            cat_probs, q_vec, physio_q, gindex, weights, k_per_cat=5, k_flat=10)
    else:
        resolved = icdr.resolve_exact_diagnoses(
            cat_probs, q_vec, gindex, resolver["alpha"], k_per_cat=5, k_flat=10)
    return _exact_icd_from_resolved(resolved)


def run_feature_vector(case, cache_entry) -> dict:
    m = _fv_models()
    out = {"acuity": None, "triage_admit": None, "refined_admit": None,
           "diag_top3": None, "dept_top3": None, "exact_icd": None}

    Xd = cache_entry["dispo_features"]
    # Triage acuity (feature-vector) from the soft-cascade columns build_features
    # already computed via the triage v3 acuity model.
    acu_probs = [float(Xd[f"triage_acuity_proba_{k}"].iloc[0]) for k in range(1, 6)]
    out["acuity"] = int(np.argmax(acu_probs)) + 1
    # Gate at the SAME threshold the live pipeline uses (DECISION_THRESHOLD),
    # so feature_vector_gated is apples-to-apples with the tool/E2E columns.
    # (triage_admit is the raw triage-v3 screening verdict, kept at its own 0.5.)
    out["triage_admit"] = float(Xd["triage_disposition_proba_admit"].iloc[0]) >= 0.5
    out["refined_admit"] = float(m["dispo"].predict_proba(Xd)[0, 1]) >= DECISION_THRESHOLD

    if "nurse_v3_features" in cache_entry:
        Xn = cache_entry["nurse_v3_features"]
        diag_proba = m["diag"].predict_proba(Xn)[0]
        top3 = np.argsort(diag_proba)[::-1][:3]
        out["diag_top3"] = [m["diag_labels"][i] for i in top3]

        Xdept = Xn.copy()
        if m["cascade_cols"]:
            for k, col in enumerate(m["cascade_cols"]):
                Xdept[col] = diag_proba[k]
        else:
            Xdept["predicted_diagnosis"] = int(np.argmax(diag_proba))
        dept_proba = m["dept"].predict_proba(Xdept)[0]
        top3d = np.argsort(dept_proba)[::-1][:3]
        out["dept_top3"] = [m["dept_labels"][i] for i in top3d]

        out["exact_icd"] = _fv_exact_icd(m, case, Xn, diag_proba)
    return out


# ===========================================================================
# Mode 1 — E2E (real crew, patched I/O, captured tool outputs)
# ===========================================================================
CAPTURE = {}        # tool_name -> list of parsed JSON outputs (current case)
CURRENT_CASE = {}   # the case whose scripted answers the patched tools serve


def _answer_ask(question: str, case: dict) -> str:
    """Route an NLP-parser follow-up question to the case's structured fields."""
    q = question.lower()
    t = case["triage_inputs"]
    ems = t["ems_vitals"] or {v: None for v in VITAL_COLS}

    def num(x):
        return "skip" if x is None else str(x)

    # MRN / patient-identity question (the parser asks this FIRST). Must precede
    # the generic "hospital before" / "medical" branches below, which would
    # otherwise swallow it. Returns the case's real subject_id so the live
    # MRN -> structured-data -> patient_history_lookup_tool chain is exercised
    # end-to-end (the leakage-cutoff intime is forced to the case's stay intime
    # by a separate E2E patch, so this column reproduces tool_direct_lookup).
    if any(k in q for k in (
        "mrn", "medical record", "record number", "patient id",
        "patient identifier", "patient number", "treated here before",
        "treated at this hospital before", "been here before",
        "been treated here", "in our system", "in the system", "registered",
        "seen you before", "seen here before", "first time here",
    )):
        sid = int(case.get("subject_id", -1) or -1)
        return f"Yes, my MRN is {sid}" if sid > 0 else "No, this is my first time here"

    if any(k in q for k in ("pain", "hurt", "out of 10", "/10")):
        return "skip" if t["pain_score"] < 0 else str(t["pain_score"])
    if any(k in q for k in ("how old", "age", "year")):
        return str(t["age"])
    if any(k in q for k in ("gender", "sex", "male or female", "man or woman")):
        return t["gender"]
    if any(k in q for k in ("arrive", "arrived", "transport", "get to", "ambulance", "walk", "come in", "brought")):
        return {"walk_in": "I walked in", "ambulance": "by ambulance",
                "helicopter": "by helicopter"}.get(t["arrival_transport"], t["arrival_transport"])
    if "temperature" in q or "temp" in q or "fever" in q:
        return num(ems["temperature"])
    if "heart rate" in q or "pulse" in q or "bpm" in q or "heartrate" in q:
        return num(ems["heartrate"])
    if "breath" in q or "respir" in q:
        return num(ems["resprate"])
    if "oxygen" in q or "o2" in q or "sat" in q:
        return num(ems["o2sat"])
    if "blood pressure" in q or "bp" in q:
        if ems["sbp"] is None or ems["dbp"] is None:
            return "skip"
        return f"{int(ems['sbp'])}/{int(ems['dbp'])}"
    if any(k in q for k in ("admitted before", "prior admission", "hospital before",
                            "admissions", "times have you been")):
        return "skip" if t["n_prior_admissions"] < 0 else str(t["n_prior_admissions"])
    if any(k in q for k in ("history", "chronic", "condition", "medical", "past")):
        return t["prior_history"] if t["prior_history"] else "no"
    if "medication" in q or "meds" in q or "taking" in q:
        return case["nurse_inputs"]["medications_raw"] or "none"
    return "skip"


def _nurse_json(case: dict) -> str:
    """Return the JSON the real nurse tool would emit, from the case's
    deterministic nurse_inputs (None for any field the patient lacked)."""
    n = case["nurse_inputs"]
    v = n["vitals"]
    result = {
        "vital_signs": {k: v[k] for k in VITAL_COLS},
        # Real multi-reading trajectory (what a nurse collecting several
        # readings would record). The doctor agent copies this into the
        # disposition / v3 tools' vital_trajectory_json arg.
        "vital_trajectory": n.get("vital_trajectory") or {},
        "rhythm": n["rhythm"] or None,
        "medications_raw": n["medications_raw"] or None,
        "prior_history": n["prior_history"] or None,
        "n_prior_admissions": int(n["n_prior_admissions"]),
    }
    return json.dumps(result, indent=2)


def _install_patches():
    """Monkeypatch interactive tools (answer from fields) and prediction tools
    (tee JSON into CAPTURE). Returns a restore() callable."""
    orig = {
        "ask": AskPatientTool._run,
        "nurse": NurseDataCollectionTool._run,
        "triage": TriagePredictionTool._run,
        "base": DoctorPredictionToolV3Base._run,
        "dispo": DoctorDispositionTool._run,
        "v3": DoctorPredictionToolV3._run,
        "lookup": PatientHistoryLookupTool._run,
    }

    def patched_ask(self, question):
        return _answer_ask(question, CURRENT_CASE)

    def patched_nurse(self, patient_context):
        return _nurse_json(CURRENT_CASE)

    def patched_lookup(self, subject_id, current_intime="now", chief_complaints=""):
        # Option A — force the case's REAL stay intime (the historical leakage
        # cutoff) instead of the live "now" the disposition prompt passes, so the
        # E2E lookup reproduces the tool_direct_lookup / feature_vector_gated
        # scenario rather than live "now" semantics (which for a returning
        # patient anchors to their LATEST encounter — a different cutoff).
        # subject_id + complaints still come from the agent: the parser must have
        # collected the MRN for this to fire, so the full chain is exercised.
        forced_intime = str(CURRENT_CASE.get("intime") or current_intime)
        result = orig["lookup"](
            self, subject_id=subject_id, current_intime=forced_intime,
            chief_complaints=chief_complaints,
        )
        try:
            CAPTURE.setdefault("lookup", []).append(json.loads(result))
        except Exception:
            pass
        return result

    def _tee(name, orig_fn):
        def wrapper(self, *args, **kwargs):
            result = orig_fn(self, *args, **kwargs)
            try:
                CAPTURE.setdefault(name, []).append(json.loads(result))
            except Exception:
                pass
            return result
        return wrapper

    AskPatientTool._run = patched_ask
    NurseDataCollectionTool._run = patched_nurse
    PatientHistoryLookupTool._run = patched_lookup
    TriagePredictionTool._run = _tee("triage", orig["triage"])
    DoctorPredictionToolV3Base._run = _tee("base", orig["base"])
    DoctorDispositionTool._run = _tee("dispo", orig["dispo"])
    DoctorPredictionToolV3._run = _tee("v3", orig["v3"])

    def restore():
        AskPatientTool._run = orig["ask"]
        NurseDataCollectionTool._run = orig["nurse"]
        PatientHistoryLookupTool._run = orig["lookup"]
        TriagePredictionTool._run = orig["triage"]
        DoctorPredictionToolV3Base._run = orig["base"]
        DoctorDispositionTool._run = orig["dispo"]
        DoctorPredictionToolV3._run = orig["v3"]

    return restore


def run_e2e(case: dict) -> dict:
    """Run one case through the real crew; extract predictions from CAPTURE."""
    global CURRENT_CASE
    from proiect_licenta.crew import ProiectLicenta

    CURRENT_CASE = case
    CAPTURE.clear()
    out = {"acuity": None, "triage_admit": None, "refined_admit": None,
           "diag_top3": None, "dept_top3": None, "exact_icd": None,
           "parser": {}, "error": None}

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ProiectLicenta().crew().kickoff(
                inputs={"user_message": case["narrative"]}
            )
    except Exception as e:  # keep the run alive; record the failure
        out["error"] = f"{type(e).__name__}: {e}"

    if CAPTURE.get("triage"):
        tj = CAPTURE["triage"][-1]
        out["acuity"] = int(tj["acuity_prediction"]["predicted_esi_level"])
        out["triage_admit"] = _to_admit(tj["disposition_prediction"]["prediction"])
        # NL-fidelity: what the parser handed the triage tool.
        out["parser"] = {
            "complaints": tj["complaint_analysis"]["input_complaints"],
            "age": tj["patient_info"]["age"],
            "gender": tj["patient_info"]["gender"],
            "arrival_transport": tj["patient_info"]["arrival_transport"],
            "pain": tj.get("pain_score_used"),
        }
    if CAPTURE.get("dispo"):
        out["refined_admit"] = bool(CAPTURE["dispo"][-1]["disposition_prediction"]["is_admitted"])
    # Final diagnosis/department = v3-nurse reassessment if it ran, else None.
    if CAPTURE.get("v3"):
        vj = CAPTURE["v3"][-1]
        if "diagnosis_prediction" in vj:
            out["diag_top3"] = _diag_top3(vj)
            out["dept_top3"] = _dept_top3(vj)
            out["exact_icd"] = _exact_icd_from_block(
                vj["diagnosis_prediction"].get("exact_diagnoses"))
    return out


# ===========================================================================
# Scoring
# ===========================================================================
def _acc(pairs):
    pairs = [p for p in pairs if p is not None]
    return (sum(pairs) / len(pairs)) if pairs else float("nan")


def score(cases, preds_by_mode, modes, true_roll):
    """Aggregate per-target accuracy for each mode. `true_roll` maps stay_id ->
    the primary-diagnosis 3-char ICD rollup (for the Stage-2 exact-ICD targets)."""
    rows = {}
    for mode in modes:
        P = preds_by_mode[mode]
        acuity_hits, dispo_hits = [], []
        d1, d3, p1, p3 = [], [], [], []
        i1, i5, i10, iu = [], [], [], []   # exact-ICD: @1 / @5 / flat@10 / union
        diag_cov, dept_cov, n_admit = 0, 0, 0
        icd_cov, icd_n = 0, 0
        for c in cases:
            sid = c["stay_id"]
            gt = c["ground_truth"]
            pr = P.get(sid)
            if pr is None:
                continue
            if pr.get("acuity") is not None:
                acuity_hits.append(int(pr["acuity"] == gt["acuity"]))
            if pr.get("refined_admit") is not None:
                dispo_hits.append(int(bool(pr["refined_admit"]) == gt["admitted"]))
            if gt["admitted"] and gt["diagnosis_group"]:
                n_admit += 1
                if pr.get("diag_top3"):
                    diag_cov += 1
                    d1.append(int(pr["diag_top3"][0] == gt["diagnosis_group"]))
                    d3.append(int(gt["diagnosis_group"] in pr["diag_top3"][:3]))
                else:
                    d1.append(0); d3.append(0)  # routed to discharge => no Dx
                if pr.get("dept_top3"):
                    dept_cov += 1
                    p1.append(int(pr["dept_top3"][0] == gt["service_group"]))
                    p3.append(int(gt["service_group"] in pr["dept_top3"][:3]))
                else:
                    p1.append(0); p3.append(0)
                # Exact-ICD (rollup) — only scored when MIMIC records a coded
                # primary diagnosis for the stay; miss when no Dx was produced.
                tr = true_roll.get(sid)
                if tr:
                    icd_n += 1
                    ei = pr.get("exact_icd")
                    if ei:
                        icd_cov += 1
                        flat, union = ei["flat"], ei["union"]
                        i1.append(int(bool(flat) and flat[0] == tr))
                        i5.append(int(tr in flat[:5]))
                        i10.append(int(tr in flat[:10]))
                        iu.append(int(tr in union))
                    else:
                        i1.append(0); i5.append(0); i10.append(0); iu.append(0)
        rows[mode] = {
            "acuity": _acc(acuity_hits),
            "dispo": _acc(dispo_hits),
            "diag_top1": _acc(d1), "diag_top3": _acc(d3),
            "dept_top1": _acc(p1), "dept_top3": _acc(p3),
            "icd_top1": _acc(i1), "icd_top5": _acc(i5),
            "icd_top10": _acc(i10), "icd_union": _acc(iu),
            "diag_cov": diag_cov, "dept_cov": dept_cov, "n_admit": n_admit,
            "icd_cov": icd_cov, "icd_n": icd_n,
        }
    return rows


def _fmt(x):
    return "  n/a " if x != x else f"{x*100:5.1f}%"


def print_nl_fidelity(cases, mode_preds):
    """Parser-quality report for any mode that captured a ``parser`` block
    (the E2E crew or the MedGemma parser-llm bypass). Orchestration-independent,
    so it is the fair Flash-vs-MedGemma anchor."""
    age_ok = gen_ok = trans_ok = 0
    comp_jacc = []        # Jaccard on what the tools saw (clinicalized when on)
    comp_jacc_raw = []    # Jaccard on the verbatim LLM parse (map off)
    map_rewrote = 0       # cases where clinicalize changed the parsed complaint
    n = 0
    for c in cases:
        pr = mode_preds.get(c["stay_id"])
        if not pr or not pr.get("parser"):
            continue
        n += 1
        t = c["triage_inputs"]
        ps = pr["parser"]
        age_ok += int(ps.get("age") == t["age"])
        gen_ok += int(str(ps.get("gender", "")).lower() == t["gender"])
        pa = str(ps.get("arrival_transport", "")).lower().replace(" ", "_")
        trans_ok += int(t["arrival_transport"] in pa or pa in t["arrival_transport"])
        truth_tok = set(normalize_complaint_text(t["chief_complaints"]).split())
        got_tok = set(normalize_complaint_text(
            ", ".join(ps.get("complaints", []))).split())
        if truth_tok or got_tok:
            comp_jacc.append(len(truth_tok & got_tok) / len(truth_tok | got_tok))
        raw_norm = normalize_complaint_text(", ".join(ps.get("complaints_raw", [])))
        got_norm = normalize_complaint_text(", ".join(ps.get("complaints", [])))
        raw_tok = set(raw_norm.split())
        if truth_tok or raw_tok:
            comp_jacc_raw.append(len(truth_tok & raw_tok) / len(truth_tok | raw_tok))
        if raw_norm != got_norm:
            map_rewrote += 1
    jacc = float(np.mean(comp_jacc)) if comp_jacc else float("nan")
    jacc_raw = float(np.mean(comp_jacc_raw)) if comp_jacc_raw else float("nan")
    if n:
        print(f"  cases with parser output:        {n}")
        print(f"  age extracted correctly:         {age_ok}/{n}")
        print(f"  gender extracted correctly:      {gen_ok}/{n}")
        print(f"  arrival transport correct:       {trans_ok}/{n}")
        print(f"  complaint Jaccard (raw parse):   {jacc_raw:.3f}"
              if comp_jacc_raw else "  complaint Jaccard (raw parse):   n/a")
        print(f"  complaint Jaccard (-> tools):    {jacc:.3f}"
              if comp_jacc else "  complaint Jaccard (-> tools):    n/a")
        print(f"  lay->clinical map rewrote:       {map_rewrote}/{n} complaints")
    else:
        print("  (no parser output captured)")
    return {"n": n, "age_ok": age_ok, "gender_ok": gen_ok,
            "transport_ok": trans_ok, "complaint_jaccard_mean": jacc,
            "complaint_jaccard_raw_mean": jacc_raw, "map_rewrote": map_rewrote}


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-feature-vector", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--llm-backend", choices=["flash", "medgemma"], default=None,
                        help="Override LLM_BACKEND for the E2E crew + parser-llm "
                             "mode. Default: whatever .env/env says (flash).")
    parser.add_argument("--parser-llm", action="store_true",
                        help="Add the MedGemma-only parser-llm bypass mode (one "
                             "direct LLM parse -> deterministic tool-direct). Use "
                             "when MedGemma can't drive the agentic crew.")
    parser.add_argument("--clinicalize", action="store_true",
                        help="Apply the deterministic lay->clinical chief-complaint "
                             "map (preprocessing.clinicalize_complaint) to the "
                             "parser-llm output before scoring. Re-runs from the "
                             "parse cache, so this costs no extra LLM calls.")
    parser.add_argument("--plain-prompt", action="store_true",
                        help="Use the original parser prompt (no clinical-term "
                             "steering) — reproduces the documented baseline. Omit "
                             "to use the clinical-term prompt (Track 1).")
    parser.add_argument("--dump-json", default=None,
                        help="Write the scored metrics + backend tag to this JSON "
                             "path (for comparing flash vs medgemma runs).")
    args = parser.parse_args()

    # Backend override must be set BEFORE the crew (and its get_llm()) is imported.
    if args.llm_backend:
        os.environ["LLM_BACKEND"] = args.llm_backend
    from proiect_licenta.llm_config import llm_backend
    backend = llm_backend()

    payload = load_cases()
    cases = payload["cases"]
    if args.limit:
        cases = cases[:args.limit]
    cache = load_feature_cache()
    # Ground-truth primary-diagnosis ICD rollup per stay (for exact-ICD targets).
    true_roll = true_rollups([c["stay_id"] for c in cases])

    print("\n" + "#" * 78)
    print("  END-TO-END PIPELINE BENCHMARK  (NL crew vs tabular benchmark)")
    print(f"  Cases: {len(cases)}  (generated {payload.get('generated_at','?')}, "
          f"seed {payload.get('seed','?')})")
    print(f"  LLM backend: {backend.upper()}"
          f"{'  (model ' + os.getenv('MEDGEMMA_MODEL', '?') + ')' if backend == 'medgemma' else ''}")
    print("#" * 78)
    n_admit_gt = sum(1 for c in cases if c["ground_truth"]["admitted"])
    n_flagged = sum(1 for c in cases if c.get("grounding") and not c["grounding"]["ok"])
    print(f"  Ground truth: {n_admit_gt} admitted / {len(cases)-n_admit_gt} discharged")
    print(f"  Grounding-flagged narratives: {n_flagged}")

    modes = []
    preds = {}

    # Tool-direct (no lookup — the live ED with no chart access)
    print_section("MODE: tool-direct (exact tabular values, no LLM)")
    preds["tool_direct"] = {}
    modes.append("tool_direct")
    for c in cases:
        try:
            preds["tool_direct"][c["stay_id"]] = run_tool_direct(c)
            print(f"  stay {c['stay_id']}: ok")
        except Exception as e:
            print(f"  stay {c['stay_id']}: ERROR {type(e).__name__}: {e}")

    # Tool-direct + EHR history lookup (returning patients get real prior data).
    # The gap vs tool_direct = the value of EHR access at triage. Skipped with a
    # note if the history index hasn't been built.
    if get_history_index() is None:
        print_section("MODE: tool-direct-lookup — SKIPPED (no history index)")
        print("  Run `uv run build_history_index` first to enable this column.")
    else:
        print_section("MODE: tool-direct-lookup (EHR history lookup by subject_id)")
        preds["tool_direct_lookup"] = {}
        modes.append("tool_direct_lookup")
        n_known = 0
        n_med = 0
        for c in cases:
            try:
                pmh_json, med_json = _lookup_blocks(c)
                if pmh_json:
                    n_known += 1
                if med_json:
                    n_med += 1
                preds["tool_direct_lookup"][c["stay_id"]] = run_tool_direct(
                    c, pmh_lookup_json=pmh_json, med_lookup_json=med_json,
                )
                hits = []
                if pmh_json:
                    hits.append("pmh")
                if med_json:
                    hits.append("med")
                print(f"  stay {c['stay_id']}: ok{(' [' + '+'.join(hits) + ' lookup hit]') if hits else ''}")
            except Exception as e:
                print(f"  stay {c['stay_id']}: ERROR {type(e).__name__}: {e}")
        print(f"  EHR lookup found prior history for {n_known}/{len(cases)} patients, "
              f"prior med list for {n_med}/{len(cases)}.")

    # Feature-vector
    if not args.skip_feature_vector:
        print_section("MODE: feature-vector (cached build_features + models)")
        preds["feature_vector"] = {}
        modes.append("feature_vector")
        for c in cases:
            try:
                preds["feature_vector"][c["stay_id"]] = run_feature_vector(c, cache[c["stay_id"]])
                print(f"  stay {c['stay_id']}: ok")
            except Exception as e:
                print(f"  stay {c['stay_id']}: ERROR {type(e).__name__}: {e}")

        # Derived: feature-vector GATED by the disposition model's OWN verdict
        # (on real longitudinal features). This makes the feature-vector column
        # apples-to-apples with the gated tool/E2E columns: diagnosis/department
        # only count when the disposition model would have admitted. Comparing
        #   feature_vector  -> feature_vector_gated : cost of the disposition gate
        #   feature_vector_gated -> tool_direct     : cost of runtime feature degradation
        #   tool_direct -> e2e                      : cost of the LLM/NLP layer
        print_section("MODE: feature-vector-gated (gated by disposition model's own verdict)")
        preds["feature_vector_gated"] = {}
        modes.append("feature_vector_gated")
        for c in cases:
            base = preds["feature_vector"].get(c["stay_id"])
            if base is None:
                continue
            g = dict(base)
            if not base.get("refined_admit"):
                g["diag_top3"] = None
                g["dept_top3"] = None
                g["exact_icd"] = None
            preds["feature_vector_gated"][c["stay_id"]] = g
        print(f"  derived for {len(preds['feature_vector_gated'])} cases")

    # Parser-LLM mode: one direct LLM parse -> deterministic tool-direct. Runs for
    # BOTH backends (get_parse_llm() is never None) so Flash and MedGemma can be
    # compared in the SAME mode — isolating parser quality from agentic tool-calling.
    if args.parser_llm:
        from proiect_licenta.llm_config import get_parse_llm
        global _ACTIVE_PARSE_PROMPT
        _ACTIVE_PARSE_PROMPT = _PARSE_PROMPT_PLAIN if args.plain_prompt else _PARSE_PROMPT
        llm = get_parse_llm()
        load_parse_cache(backend)
        label = "parser-llm (direct LLM parse -> tool-direct, no agentic tool-calling)"
        label += "  [plain prompt]" if args.plain_prompt else "  [clinical-term prompt]"
        if args.clinicalize:
            label += "  [+lay->clinical map]"
        print_section(f"MODE: {label}")
        preds["parser_llm"] = {}
        modes.append("parser_llm")
        for i, c in enumerate(cases, 1):
            try:
                preds["parser_llm"][c["stay_id"]] = run_parser_llm_bypass(
                    c, llm, clinicalize=args.clinicalize)
                print(f"  [{i}/{len(cases)}] stay {c['stay_id']}: ok")
            except Exception as e:
                print(f"  [{i}/{len(cases)}] stay {c['stay_id']}: ERROR {type(e).__name__}: {e}")
        save_parse_cache()
        print(f"  parse cache: {_PARSE_CACHE_HITS} hits / {_PARSE_CACHE_MISSES} "
              f"LLM calls -> {_PARSE_CACHE_PATH}")

    # E2E (last — heaviest, runs the LLM crew)
    if not args.skip_e2e:
        print_section("MODE: E2E (real crew, narrative -> NLP parser -> models)")
        restore = _install_patches()
        preds["e2e"] = {}
        modes.append("e2e")
        try:
            for i, c in enumerate(cases, 1):
                r = run_e2e(c)
                preds["e2e"][c["stay_id"]] = r
                status = r["error"] if r["error"] else (
                    f"ESI {r['acuity']}, refined_admit={r['refined_admit']}")
                print(f"  [{i}/{len(cases)}] stay {c['stay_id']}: {status}")
        finally:
            restore()

    # ── Headline table ──
    rows = score(cases, preds, modes, true_roll)
    print_section("ACCURACY ON THE SAME CASES  (per target, per mode)")
    header = f"  {'target':16s}" + "".join(f"{m:>22s}" for m in modes)
    print(header)
    print("  " + "-" * (16 + 22 * len(modes)))
    for target, label in [
        ("acuity", "ESI acuity"),
        ("dispo", "disposition"),
        ("diag_top1", "diagnosis @1"),
        ("diag_top3", "diagnosis @3"),
        ("dept_top1", "department @1"),
        ("dept_top3", "department @3"),
        ("icd_top1", "ICD exact @1"),
        ("icd_top5", "ICD exact @5"),
        ("icd_top10", "ICD flat@10"),
        ("icd_union", "ICD union"),
    ]:
        line = f"  {label:16s}" + "".join(f"{_fmt(rows[m][target]):>22s}" for m in modes)
        print(line)
    # Coverage notes
    print()
    for m in modes:
        r = rows[m]
        print(f"  [{m}] diagnosis/department scored over {r['n_admit']} admitted-GT cases; "
              f"pipeline produced a Dx for {r['diag_cov']}/{r['n_admit']} "
              f"(rest routed to discharge => counted as miss).")
    print()
    for m in modes:
        r = rows[m]
        print(f"  [{m}] exact-ICD (3-char rollup) scored over {r['icd_n']} admitted-GT cases "
              f"with a coded primary Dx; pipeline produced exact-ICD for "
              f"{r['icd_cov']}/{r['icd_n']} (rest counted as miss). Compare ICD union / "
              f"flat@10 with the full-test-set numbers in benchmark_icd_resolution.py.")

    # ── Per-case dump ──
    print_section("PER-CASE DETAIL")
    for c in cases:
        sid = c["stay_id"]
        gt = c["ground_truth"]
        print(f"\n  stay {sid}  [{'ADMIT' if gt['admitted'] else 'DISCHARGE'}]  "
              f"truth: ESI {gt['acuity']}, "
              f"dx={gt['diagnosis_group']}, dept={gt['service_group']}, "
              f"icd={true_roll.get(sid) or '—'}")
        print(f"    complaint (raw): {c['triage_inputs']['chief_complaints']}")
        print(f"    narrative:       {c['narrative']}")
        if not (c.get("grounding") and c["grounding"]["ok"]):
            print(f"    [grounding FLAGGED: {c.get('grounding')}]")
        for m in modes:
            pr = preds[m].get(sid)
            if not pr:
                continue
            dx = pr.get("diag_top3")[0] if pr.get("diag_top3") else "—"
            dp = pr.get("dept_top3")[0] if pr.get("dept_top3") else "—"
            ei = pr.get("exact_icd")
            ic = ei["flat"][0] if (ei and ei.get("flat")) else "—"
            extra = f"  err={pr['error']}" if pr.get("error") else ""
            print(f"    {m:14s}: ESI {pr.get('acuity')}, "
                  f"refined_admit={pr.get('refined_admit')}, dx={dx}, dept={dp}, "
                  f"icd={ic}{extra}")

    # ── NL-fidelity report (parser extraction vs tabular truth) ──
    # This is the orchestration-independent, apples-to-apples parser-quality
    # comparison: it works identically for the E2E crew parser AND the MedGemma
    # parser-llm bypass, so Flash and MedGemma can be compared here even when
    # MedGemma runs in bypass mode and Gemini runs the full agentic crew.
    nl_fidelity = {}
    for fid_mode, fid_label in (("e2e", "E2E crew parser"),
                                ("parser_llm", "parser-llm (direct LLM parse)")):
        if fid_mode in preds:
            print_section(f"NL-FIDELITY [{fid_label}]  (parser extraction vs tabular truth)")
            nl_fidelity[fid_mode] = print_nl_fidelity(cases, preds[fid_mode])

    if args.dump_json:
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_json).write_text(json.dumps(
            {"llm_backend": backend, "clinicalize": bool(args.clinicalize),
             "parse_prompt": "plain" if args.plain_prompt else "clinical",
             "modes": modes, "metrics": rows,
             "nl_fidelity": nl_fidelity, "n_cases": len(cases)}, indent=2), encoding="utf-8")
        print(f"\n  Metrics dumped -> {args.dump_json}")

    print("\n" + "#" * 78)
    print("  BENCHMARK COMPLETE")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
