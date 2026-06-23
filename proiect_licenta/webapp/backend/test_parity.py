"""Parity test — the web pipeline must match the trusted benchmark exactly.

Asserts that driving a synthetic case through ``proiect_licenta.pipeline`` (the
stage functions the FastAPI backend calls) yields the SAME acuity, triage admit,
refined admit, diagnosis top-1 and department top-1 as the benchmark's
``run_tool_direct`` — the reference orchestration the live crew is validated
against. Same tool instances ⇒ identical predictions; any mismatch means the web
wiring drifted from the crew.

Run from the repo root:
    uv run python webapp/backend/test_parity.py            # default: 12 cases
    uv run python webapp/backend/test_parity.py 40         # more cases
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the benchmark importable (it lives outside the package).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "benchmarks"))

from proiect_licenta import pipeline as P  # noqa: E402
from proiect_licenta.case_generation import load_cases  # noqa: E402
import benchmark_pipeline_e2e as B  # noqa: E402


def _web_path(case: dict) -> dict:
    """Replicate exactly what the backend does for a case, no EHR lookup
    (the plain self-report column — matches run_tool_direct's default)."""
    t = case["triage_inputs"]
    n = case["nurse_inputs"]
    tj = P.run_triage(
        chief_complaints=t["chief_complaints"], pain_score=t["pain_score"],
        age=t["age"], gender=t["gender"], arrival_transport=t["arrival_transport"],
        ems_vitals=t["ems_vitals"], prior_history=t["prior_history"],
        n_prior_admissions=t["n_prior_admissions"],
    )
    acuity = P.acuity_of(tj)
    triage_admit = P.triage_admit_of(tj)
    nurse = {
        "vital_signs": n["vitals"],
        "vital_trajectory": n.get("vital_trajectory") or {},
        "rhythm": n["rhythm"], "medications_raw": n["medications_raw"],
        "prior_history": n["prior_history"],
        "n_prior_admissions": n["n_prior_admissions"],
    }
    dj = P.run_disposition(
        chief_complaints=t["chief_complaints"], pain_score=t["pain_score"],
        age=t["age"], gender=t["gender"], arrival_transport=t["arrival_transport"],
        predicted_acuity=acuity, triage_is_admitted=triage_admit, nurse=nurse,
    )
    refined = P.refined_admit_of(dj)
    diag1 = dept1 = None
    if refined:
        vj = P.run_reassessment(
            chief_complaints=t["chief_complaints"], pain_score=t["pain_score"],
            age=t["age"], gender=t["gender"], arrival_transport=t["arrival_transport"],
            predicted_acuity=acuity, refined_is_admitted=refined, nurse=nurse,
        )
        diag1 = vj["diagnosis_prediction"]["top_3_categories"][0]["category"]
        dept1 = vj["department_prediction"]["top_3_departments"][0]["code"]
    return {"acuity": acuity, "triage_admit": triage_admit,
            "refined_admit": refined, "diag1": diag1, "dept1": dept1}


def main(n: int = 12) -> int:
    cases = load_cases()["cases"][:n]
    failures = 0
    for case in cases:
        ref = B.run_tool_direct(case)
        web = _web_path(case)
        ref_diag1 = ref["diag_top3"][0] if ref["diag_top3"] else None
        ref_dept1 = ref["dept_top3"][0] if ref["dept_top3"] else None
        ok = (
            web["acuity"] == ref["acuity"]
            and web["triage_admit"] == ref["triage_admit"]
            and web["refined_admit"] == ref["refined_admit"]
            and web["diag1"] == ref_diag1
            and web["dept1"] == ref_dept1
        )
        if not ok:
            failures += 1
            print(f"  MISMATCH stay {case['stay_id']}: web={web} ref_diag1={ref_diag1} ref_dept1={ref_dept1}")
        else:
            print(f"  OK stay {case['stay_id']}: acuity={web['acuity']} "
                  f"triage_admit={web['triage_admit']} refined={web['refined_admit']} "
                  f"diag={web['diag1']} dept={web['dept1']}")
    print(f"\n{len(cases) - failures}/{len(cases)} cases match the benchmark.")
    if failures:
        print("PARITY FAILED")
        return 1
    print("PARITY OK — web pipeline is byte-identical to the crew path.")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    raise SystemExit(main(n))
