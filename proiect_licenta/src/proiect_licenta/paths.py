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
    |   +-- v3/                 # v2 + PMH features (Doctor Change 1 recipe)
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

# Hospital-wide tables used as PMH sources (Doctor v3 nurse Change 1).
# diagnoses_icd.csv: ICD codes per prior hospital admission; the ICD->group map
# is derived from categorized_diagnosis.csv (ED) and reused here.
DIAGNOSES_ICD_CSV = HOSP_DIR / "diagnoses_icd.csv"
ADMISSIONS_CSV = HOSP_DIR / "admissions.csv"

# Free-text discharge summaries. The PMH section is parsed per prior admission
# for the same patient, mapped through pmh_vocab.py, and OR'd with the ICD-
# derived flags. Nested layout: data/mimic-iv-notes/mimic-iv-notes/note/...
DISCHARGE_NOTES_CSV = (
    MIMIC_IV_NOTES_DIR / "mimic-iv-notes" / "note" / "discharge.csv" / "discharge.csv"
)

# Derived/cached artifacts (e.g., pre-parsed PMH flags) — small files, kept
# out of memory between training runs. Created on first parse.
DERIVED_DIR = DATA_DIR / "derived"

# Triage-v3 hyperparameter-search feature cache (float32 parquet: the outer
# train/test feature matrices + targets). Distinct from the doctor's
# `derived/tune_cache/` so the two sweeps never share a cache. Built by
# scripts/tune_triage_v3.py on first run; reused in seconds thereafter.
TRIAGE_TUNE_CACHE_DIR = DERIVED_DIR / "triage_tune_cache"

# Doctor-disposition hyperparameter-search feature cache (float32 parquet: the
# full 425K feature matrix X + the binary admit label). Distinct from the
# doctor diagnosis/department `derived/tune_cache/` because the disposition
# model uses a different dataset (full 425K, no admitted-only filter) and the
# triage v3 SOFT cascade. Built by scripts/tune_doctor_v3_heads.py on first run
# (parses discharge.csv once); reused in seconds thereafter. The department
# stage of that script reuses the existing `derived/tune_cache/` instead.
DOCTOR_DISPO_TUNE_CACHE_DIR = DERIVED_DIR / "doctor_dispo_tune_cache"

# ---- Trained model artifacts --------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRIAGE_V1_DIR = ARTIFACTS_DIR / "triage" / "v1"
TRIAGE_V2_DIR = ARTIFACTS_DIR / "triage" / "v2"
# Triage v3 adds the same 19-feature PMH block used by Doctor v3 nurse
# (Change 1) on top of the v2 vital-augmented feature set.
TRIAGE_V3_DIR = ARTIFACTS_DIR / "triage" / "v3"
# Triage-v3 hyperparameter-search outputs (Optuna SQLite study, per-trial JSON
# logs, tuned-params + final report). A dedicated subdir so the constrained
# Group-2 regularization sweep (scripts/tune_triage_v3.py) never overwrites the
# live triage v3 model joblibs that sit directly in TRIAGE_V3_DIR. Reporting-
# only: the sweep reads the live models but never writes them.
TRIAGE_V3_HPO_DIR = TRIAGE_V3_DIR / "hpo"
DOCTOR_V1_DIR = ARTIFACTS_DIR / "doctor" / "v1"
DOCTOR_V2_DIR = ARTIFACTS_DIR / "doctor" / "v2"
# v3 tier: catch-all class ("Symptoms, Signs, Ill-Defined") excluded, full
# admitted-patient dataset (no 100K sub-sample), v3_base mirrors v1's
# feature set, v3 mirrors v2's plus longitudinal vitals + rhythm.
# DOCTOR_V3_DIR also holds the peer disposition model
# (disposition_model.joblib) added under plan section 3 — a binary
# admit/discharge classifier trained on the FULL 425K dataset (not the
# admitted-only ~102K slice used by diagnosis + department) and consumed
# by doctor_disposition_tool to refine the triage disposition after the
# nurse step. metadata.json gains a "disposition" sub-block.
DOCTOR_V3_BASE_DIR = ARTIFACTS_DIR / "doctor" / "v3_base"
DOCTOR_V3_DIR = ARTIFACTS_DIR / "doctor" / "v3"
# Doctor-v3 hyperparameter-search outputs (Optuna SQLite study, per-trial JSON
# logs, tuned-params block + final report) for the department + disposition
# heads. A dedicated subdir — mirrors TRIAGE_V3_HPO_DIR — so the constrained
# Group-2 regularization sweep (scripts/tune_doctor_v3_heads.py) never overwrites
# the live doctor v3 model joblibs that sit directly in DOCTOR_V3_DIR.
# Reporting-only: the sweep reads the live models but never writes them. (Kept
# separate from the older write-mode diagnosis sweep's optuna_study.db, which
# lives directly in DOCTOR_V3_DIR.)
DOCTOR_V3_HPO_DIR = DOCTOR_V3_DIR / "hpo"
# Stage-2 exact-ICD resolver (per-category prototype centroids + prevalence
# priors) built offline by `uv run train_icd_resolver` from the Doctor v3
# training split. Consumed by benchmarks/benchmark_icd_resolution.py (and,
# later, the runtime doctor tool). See src/proiect_licenta/icd_resolution.py.
DOCTOR_V3_ICD_RESOLVER_DIR = DOCTOR_V3_DIR / "icd_resolver"

# ---- Patient-history (EHR-simulation) index -----------------------------
# Prebuilt per-subject prior-encounter index (admissions / ED visits / ICD /
# discharge-note PMH) produced offline by `uv run build_history_index`
# (pmh_features.build_pmh_index). PatientHistoryLookupTool loads this at
# inference and queries it per patient with the strict `< intime` leakage
# filter, simulating an EHR record lookup for returning patients without
# re-parsing the 3.3 GB discharge.csv. Gitignored (MIMIC DUA).
HISTORY_INDEX_DIR = ARTIFACTS_DIR / "history"
HISTORY_INDEX_PKL = HISTORY_INDEX_DIR / "pmh_index.joblib"
