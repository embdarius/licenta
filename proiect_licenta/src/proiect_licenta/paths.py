"""Canonical filesystem paths for datasets and trained-model artifacts.

All pipeline, tool, and benchmark code imports paths from here so the on-disk
layout lives in one place. Raw MIMIC-IV CSVs sit under data/ and trained models
under artifacts/{triage,doctor}/{v1,v2,v3,...} (both gitignored). This module is
at src/proiect_licenta/paths.py, so parents[2] resolves to the project root.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Datasets
DATA_DIR = PROJECT_ROOT / "data"
MIMIC_IV_DIR = DATA_DIR / "mimic-iv"
MIMIC_IV_ED_DIR = DATA_DIR / "mimic-iv-ed"
MIMIC_IV_NOTES_DIR = DATA_DIR / "mimic-iv-notes"
HOSP_DIR = MIMIC_IV_DIR / "hosp"

# Common CSV files
TRIAGE_CSV = MIMIC_IV_ED_DIR / "triage.csv"
VITALSIGN_CSV = MIMIC_IV_ED_DIR / "vitalsign.csv"
EDSTAYS_CSV = MIMIC_IV_ED_DIR / "edstays.csv"
DIAGNOSIS_CSV = MIMIC_IV_ED_DIR / "files_created" / "categorized_diagnosis.csv"
MEDRECON_CSV = MIMIC_IV_ED_DIR / "medrecon.csv"
PATIENTS_CSV = HOSP_DIR / "patients.csv"
SERVICES_CSV = HOSP_DIR / "services.csv"

# Hospital-wide tables used as PMH sources. diagnoses_icd.csv holds ICD codes per
# prior hospital admission; the ICD-to-group map is derived from
# categorized_diagnosis.csv (ED) and reused here.
DIAGNOSES_ICD_CSV = HOSP_DIR / "diagnoses_icd.csv"
ADMISSIONS_CSV = HOSP_DIR / "admissions.csv"

# Free-text discharge summaries. The PMH section is parsed per prior admission,
# mapped through pmh_vocab.py, and OR'd with the ICD-derived flags. The path is
# doubly nested by the MIMIC notes distribution.
DISCHARGE_NOTES_CSV = (
    MIMIC_IV_NOTES_DIR / "mimic-iv-notes" / "note" / "discharge.csv" / "discharge.csv"
)

# Derived/cached artifacts (e.g. pre-parsed PMH flags), created on first parse.
DERIVED_DIR = DATA_DIR / "derived"

# Triage-v3 HPO feature cache. Kept separate from the doctor's tune_cache so the
# two sweeps never share a cache.
TRIAGE_TUNE_CACHE_DIR = DERIVED_DIR / "triage_tune_cache"

# Doctor-disposition HPO feature cache. Separate from the diagnosis/department
# tune_cache because disposition uses the full 425K dataset (no admitted-only
# filter) and the triage v3 soft cascade.
DOCTOR_DISPO_TUNE_CACHE_DIR = DERIVED_DIR / "doctor_dispo_tune_cache"

# Trained model artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRIAGE_V1_DIR = ARTIFACTS_DIR / "triage" / "v1"
TRIAGE_V2_DIR = ARTIFACTS_DIR / "triage" / "v2"
# v3 adds the 19-feature PMH block on top of the v2 vital-augmented features.
TRIAGE_V3_DIR = ARTIFACTS_DIR / "triage" / "v3"
# HPO outputs in a dedicated subdir so the reporting-only sweep never overwrites
# the live model joblibs in TRIAGE_V3_DIR.
TRIAGE_V3_HPO_DIR = TRIAGE_V3_DIR / "hpo"
DOCTOR_V1_DIR = ARTIFACTS_DIR / "doctor" / "v1"
DOCTOR_V2_DIR = ARTIFACTS_DIR / "doctor" / "v2"
# v3 tier: catch-all class excluded, full admitted-patient dataset. v3_base
# mirrors v1's features; v3 mirrors v2's plus longitudinal vitals and rhythm.
# DOCTOR_V3_DIR also holds disposition_model.joblib, a binary admit/discharge
# classifier trained on the full 425K dataset and consumed by
# doctor_disposition_tool.
DOCTOR_V3_BASE_DIR = ARTIFACTS_DIR / "doctor" / "v3_base"
DOCTOR_V3_DIR = ARTIFACTS_DIR / "doctor" / "v3"
# HPO outputs for the department + disposition heads, in a dedicated subdir so
# the reporting-only sweep never overwrites the live joblibs in DOCTOR_V3_DIR.
DOCTOR_V3_HPO_DIR = DOCTOR_V3_DIR / "hpo"
# Stage-2 exact-ICD resolver built offline by `uv run train_icd_resolver` from
# the doctor v3 training split. See src/proiect_licenta/icd_resolution.py.
DOCTOR_V3_ICD_RESOLVER_DIR = DOCTOR_V3_DIR / "icd_resolver"

# Patient-history lookup index: per-subject prior-encounter data (admissions, ED
# visits, ICD, discharge-note PMH) built offline by `uv run build_history_index`.
# PatientHistoryLookupTool loads it at inference and queries each patient with a
# strict `< intime` leakage filter, avoiding a re-parse of the 3.3 GB
# discharge.csv. Gitignored (MIMIC DUA).
HISTORY_INDEX_DIR = ARTIFACTS_DIR / "history"
HISTORY_INDEX_PKL = HISTORY_INDEX_DIR / "pmh_index.joblib"
