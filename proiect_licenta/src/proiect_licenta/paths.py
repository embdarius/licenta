"""Canonical filesystem paths for datasets and trained-model artifacts.

All pipeline, tool, and benchmark code should import paths from here rather
than constructing them locally — this keeps the on-disk layout in one place.

Layout (relative to the project root):

    data/                       # Raw MIMIC-IV CSVs (gitignored)
    +-- mimic-iv/
    +-- mimic-iv-ed/
    +-- mimic-iv-notes/

    artifacts/                  # Trained model artifacts (gitignored)
    +-- triage/
    |   +-- v1/                 # acuity, disposition, tfidf, severity_map, metadata
    |   +-- v2/                 # v1 + vital_medians (with vital signs)
    +-- doctor/
        +-- v1/                 # diagnosis_model, department_model, metadata (14 classes)
        +-- v2/                 # same names; trained with nurse data (14 classes)
        +-- v3_base/            # 13 classes (catch-all dropped), full data, no nurse
        +-- v3/                 # 13 classes, full data, longitudinal vitals + rhythm + meds

This module sits at src/proiect_licenta/paths.py — `parents[2]` therefore
resolves to the project root (proiect_licenta/), one level above src/.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Datasets ------------------------------------------------------------
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

# ---- Trained model artifacts --------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRIAGE_V1_DIR = ARTIFACTS_DIR / "triage" / "v1"
TRIAGE_V2_DIR = ARTIFACTS_DIR / "triage" / "v2"
DOCTOR_V1_DIR = ARTIFACTS_DIR / "doctor" / "v1"
DOCTOR_V2_DIR = ARTIFACTS_DIR / "doctor" / "v2"
# v3 tier: catch-all class ("Symptoms, Signs, Ill-Defined") excluded, full
# admitted-patient dataset (no 100K sub-sample), v3_base mirrors v1's
# feature set, v3 mirrors v2's plus longitudinal vitals + rhythm.
DOCTOR_V3_BASE_DIR = ARTIFACTS_DIR / "doctor" / "v3_base"
DOCTOR_V3_DIR = ARTIFACTS_DIR / "doctor" / "v3"
