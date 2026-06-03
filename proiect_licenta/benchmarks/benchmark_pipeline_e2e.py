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
top-1/top-3, department top-1/top-3. Diagnosis/department are scored over the
admitted ground-truth cases (with a coverage note for cases the pipeline routed
to discharge). Also prints a per-case dump and an NL-fidelity report (what the
parser extracted vs the tabular truth).

Usage:
    uv run python benchmarks/benchmark_pipeline_e2e.py [--limit N] [--skip-feature-vector] [--skip-e2e]
"""

import argparse
import contextlib
import io
import json
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

from proiect_licenta.paths import DOCTOR_V3_DIR
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.case_generation import load_cases, load_feature_cache, VITAL_COLS
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


def _lookup_pmh_json(case) -> str:
    """Call the PatientHistoryLookupTool with the case's REAL subject_id +
    intime and return the `pmh_block` as a JSON string (or "" if the patient is
    unknown / the index isn't built). This is what an EHR lookup for a returning
    patient would supply at runtime."""
    # The benchmark reconstructs a HISTORICAL visit, so it must pass that stay's
    # exact intime as the leakage cutoff — never the 'now' sentinel (which would
    # anchor to the subject's latest encounter and could include this stay).
    # If the case bundle predates the `intime` field, skip the lookup entirely.
    intime = str(case.get("intime", "")).strip()
    if not intime:
        return ""
    tools = _direct_tools()
    res = json.loads(tools["lookup"]._run(
        subject_id=int(case.get("subject_id", -1)),
        current_intime=intime,
        chief_complaints=case["triage_inputs"]["chief_complaints"],
    ))
    if res.get("known_patient") and res.get("pmh_block"):
        return json.dumps(res["pmh_block"])
    return ""


def run_tool_direct(case, pmh_lookup_json: str = "") -> dict:
    """Replicate the crew pipeline with direct tool calls on exact fields.

    `pmh_lookup_json` (empty by default) feeds the disposition + v3 tools the
    real prior-encounter PMH block from an EHR lookup; with "" they use the
    ask-the-patient self-report path. The two settings give the no-lookup and
    with-lookup benchmark columns."""
    t = case["triage_inputs"]
    n = case["nurse_inputs"]
    tools = _direct_tools()
    out = {"acuity": None, "triage_admit": None, "refined_admit": None,
           "diag_top3": None, "dept_top3": None}

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
        ))
        out["diag_top3"] = _diag_top3(v3_j)
        out["dept_top3"] = _dept_top3(v3_j)
    return out


def run_tool_direct_lookup(case) -> dict:
    """tool-direct WITH the EHR history lookup: fetch the real prior-encounter
    PMH block by subject_id and feed it to the disposition + v3 tools. The gap
    vs the plain tool_direct column = the value of EHR access at triage for
    returning patients."""
    return run_tool_direct(case, pmh_lookup_json=_lookup_pmh_json(case))


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
    return _FV


def run_feature_vector(case, cache_entry) -> dict:
    m = _fv_models()
    out = {"acuity": None, "triage_admit": None, "refined_admit": None,
           "diag_top3": None, "dept_top3": None}

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
           "diag_top3": None, "dept_top3": None, "parser": {}, "error": None}

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
    return out


# ===========================================================================
# Scoring
# ===========================================================================
def _acc(pairs):
    pairs = [p for p in pairs if p is not None]
    return (sum(pairs) / len(pairs)) if pairs else float("nan")


def score(cases, preds_by_mode, modes):
    """Aggregate per-target accuracy for each mode."""
    rows = {}
    for mode in modes:
        P = preds_by_mode[mode]
        acuity_hits, dispo_hits = [], []
        d1, d3, p1, p3 = [], [], [], []
        diag_cov, dept_cov, n_admit = 0, 0, 0
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
        rows[mode] = {
            "acuity": _acc(acuity_hits),
            "dispo": _acc(dispo_hits),
            "diag_top1": _acc(d1), "diag_top3": _acc(d3),
            "dept_top1": _acc(p1), "dept_top3": _acc(p3),
            "diag_cov": diag_cov, "dept_cov": dept_cov, "n_admit": n_admit,
        }
    return rows


def _fmt(x):
    return "  n/a " if x != x else f"{x*100:5.1f}%"


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-feature-vector", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()

    payload = load_cases()
    cases = payload["cases"]
    if args.limit:
        cases = cases[:args.limit]
    cache = load_feature_cache()

    print("\n" + "#" * 78)
    print("  END-TO-END PIPELINE BENCHMARK  (NL crew vs tabular benchmark)")
    print(f"  Cases: {len(cases)}  (generated {payload.get('generated_at','?')}, "
          f"seed {payload.get('seed','?')})")
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
        for c in cases:
            try:
                pmh_json = _lookup_pmh_json(c)
                if pmh_json:
                    n_known += 1
                preds["tool_direct_lookup"][c["stay_id"]] = run_tool_direct(c, pmh_json)
                print(f"  stay {c['stay_id']}: ok{' [lookup hit]' if pmh_json else ''}")
            except Exception as e:
                print(f"  stay {c['stay_id']}: ERROR {type(e).__name__}: {e}")
        print(f"  EHR lookup found prior history for {n_known}/{len(cases)} patients.")

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
            preds["feature_vector_gated"][c["stay_id"]] = g
        print(f"  derived for {len(preds['feature_vector_gated'])} cases")

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
    rows = score(cases, preds, modes)
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
    ]:
        line = f"  {label:16s}" + "".join(f"{_fmt(rows[m][target]):>22s}" for m in modes)
        print(line)
    # Diagnosis/department coverage note
    print()
    for m in modes:
        r = rows[m]
        print(f"  [{m}] diagnosis/department scored over {r['n_admit']} admitted-GT cases; "
              f"pipeline produced a Dx for {r['diag_cov']}/{r['n_admit']} "
              f"(rest routed to discharge => counted as miss).")

    # ── Per-case dump ──
    print_section("PER-CASE DETAIL")
    for c in cases:
        sid = c["stay_id"]
        gt = c["ground_truth"]
        print(f"\n  stay {sid}  [{'ADMIT' if gt['admitted'] else 'DISCHARGE'}]  "
              f"truth: ESI {gt['acuity']}, "
              f"dx={gt['diagnosis_group']}, dept={gt['service_group']}")
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
            extra = f"  err={pr['error']}" if pr.get("error") else ""
            print(f"    {m:14s}: ESI {pr.get('acuity')}, "
                  f"refined_admit={pr.get('refined_admit')}, dx={dx}, dept={dp}{extra}")

    # ── NL-fidelity report (E2E parser extraction vs tabular truth) ──
    if "e2e" in preds:
        print_section("NL-FIDELITY  (what the parser extracted vs tabular truth)")
        age_ok = gen_ok = trans_ok = 0
        comp_jacc = []
        n = 0
        for c in cases:
            pr = preds["e2e"].get(c["stay_id"])
            if not pr or not pr.get("parser"):
                continue
            n += 1
            t = c["triage_inputs"]
            ps = pr["parser"]
            age_ok += int(ps.get("age") == t["age"])
            gen_ok += int(str(ps.get("gender", "")).lower() == t["gender"])
            pa = str(ps.get("arrival_transport", "")).lower().replace(" ", "_")
            trans_ok += int(t["arrival_transport"] in pa or pa in t["arrival_transport"])
            # complaint token Jaccard (normalized)
            truth_tok = set(normalize_complaint_text(t["chief_complaints"]).split())
            got_tok = set(normalize_complaint_text(
                ", ".join(ps.get("complaints", []))).split())
            if truth_tok or got_tok:
                comp_jacc.append(len(truth_tok & got_tok) / len(truth_tok | got_tok))
        if n:
            print(f"  cases with parser output:        {n}")
            print(f"  age extracted correctly:         {age_ok}/{n}")
            print(f"  gender extracted correctly:      {gen_ok}/{n}")
            print(f"  arrival transport correct:       {trans_ok}/{n}")
            print(f"  mean complaint token Jaccard:    {np.mean(comp_jacc):.3f}"
                  if comp_jacc else "  mean complaint token Jaccard:    n/a")

    print("\n" + "#" * 78)
    print("  BENCHMARK COMPLETE")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
