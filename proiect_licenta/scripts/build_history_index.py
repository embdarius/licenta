#!/usr/bin/env python
"""
Offline builder for the patient-history (EHR-simulation) index.
================================================================

Runs the heavy half of the PMH pipeline ONCE
(`pmh_features.build_pmh_index`) over a set of subject_ids and persists the
four per-subject prior-encounter structures to `HISTORY_INDEX_PKL` (joblib,
gitignored — MIMIC DUA). `PatientHistoryLookupTool` then loads this index at
inference and queries it per patient with the strict `< intime` leakage filter,
simulating an EHR record lookup for returning patients WITHOUT re-parsing the
3.3 GB discharge.csv every time.

The index uses the SAME loaders, ICD→group map, catch-all filter, and
leakage-safe construction as `train_nurse_v3` / `train_triage_v3`, so the
PMH features the tool produces match what the models were trained on.

Subject scope (pick one; default `--from-cases`):
  --from-cases     index exactly the subject_ids in the generated cases.json
                   (fast, targeted; the cohort the E2E benchmark queries).
                   Requires `uv run generate_cases` to have run first.
  --all-subjects   index every subject in edstays.csv (deployment-grade; parses
                   the full discharge.csv — minutes, but reusable for any case).

Usage:
    uv run build_history_index                 # from cases.json (default)
    uv run build_history_index --all-subjects  # full population
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import (
    EDSTAYS_CSV, DIAGNOSES_ICD_CSV, ADMISSIONS_CSV, DISCHARGE_NOTES_CSV,
    DIAGNOSIS_CSV, HISTORY_INDEX_DIR, HISTORY_INDEX_PKL,
)
from proiect_licenta.pmh_features import build_pmh_index
from proiect_licenta.training.train_doctor import (
    CATCH_ALL_LABEL, DIAGNOSIS_GROUP_MAP,
)
from proiect_licenta.case_generation import CASES_JSON


def _subjects_from_cases() -> set:
    if not CASES_JSON.exists():
        raise FileNotFoundError(
            f"{CASES_JSON} not found. Run `uv run generate_cases` first, or use "
            f"--all-subjects to index the full population."
        )
    payload = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    subs = {int(c["subject_id"]) for c in payload["cases"]}
    print(f"  Subjects from cases.json: {len(subs):,}")
    return subs


def main():
    parser = argparse.ArgumentParser(description="Build the patient-history index.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-cases", action="store_true", default=True,
                       help="Index the subject_ids in the generated cases.json (default).")
    group.add_argument("--all-subjects", action="store_true",
                       help="Index every subject in edstays.csv (full population).")
    args = parser.parse_args()

    print("=" * 70)
    print("  BUILD PATIENT-HISTORY INDEX (EHR-simulation, offline)")
    print("=" * 70)

    print("\n[1/3] Loading edstays.csv...")
    edstays = pd.read_csv(EDSTAYS_CSV)

    if args.all_subjects:
        subjects = set(edstays["subject_id"].astype(int).tolist())
        print(f"  Indexing FULL population: {len(subjects):,} subjects")
    else:
        subjects = _subjects_from_cases()

    print("\n[2/3] Building index (heaviest step — parses discharge.csv)...")
    index = build_pmh_index(
        subjects=subjects,
        edstays_full=edstays,
        diagnoses_icd_csv_path=DIAGNOSES_ICD_CSV,
        admissions_csv_path=ADMISSIONS_CSV,
        discharge_csv_path=DISCHARGE_NOTES_CSV,
        diagnosis_csv_path=DIAGNOSIS_CSV,
        diagnosis_group_map=DIAGNOSIS_GROUP_MAP,
        catch_all_label=CATCH_ALL_LABEL,
    )

    print("\n[3/3] Persisting index...")
    HISTORY_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "all_subjects" if args.all_subjects else "from_cases",
        "n_subjects": len(subjects),
        "catch_all_label": CATCH_ALL_LABEL,
        "index": index,
    }
    joblib.dump(payload, HISTORY_INDEX_PKL)

    print(f"\n  Saved index -> {HISTORY_INDEX_PKL}")
    print(f"  adm_by_subject : {len(index['adm_by_subject']):,} subjects with prior admissions")
    print(f"  ed_by_subject  : {len(index['ed_by_subject']):,} subjects with ED visits")
    print(f"  icd_pmh_by_hadm: {len(index['icd_pmh_by_hadm']):,} admissions with ICD PMH")
    print(f"  note_pmh_by_hadm:{len(index['note_pmh_by_hadm']):,} admissions with note PMH")
    print(f"  med_by_stay    : {len(index.get('med_by_stay', {})):,} stays with a medrecon block")
    print("=" * 70)


if __name__ == "__main__":
    main()
