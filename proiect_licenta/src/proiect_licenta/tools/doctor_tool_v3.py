"""
Doctor Prediction Tool v3 — 13-class label space with longitudinal vitals + rhythm

Same role as DoctorPredictionToolV2 but loads the v3 with-nurse model
artifacts (DOCTOR_V3_DIR) and exposes the new feature set:

  v3 base feature vector
    triage features (TF-IDF + structured) + cascading triage predictions
  + snapshot vitals + clinical flags + medications  (Phase A nurse data)
  + longitudinal vitals (min/max/last/delta per vital + abnormal-reading
    counts) + cardiac rhythm one-hot bucket + rhythm_irregular flag
                                              (Phase B nurse data)

The nurse only collects a single vital snapshot at inference time. We
synthesize the trajectory features as if min == max == last == snapshot
and delta == 0; the model was trained against this same fallback for
stays without vitalsign.csv coverage, so the inference behavior is
consistent with training.

The catch-all class ("Symptoms, Signs, Ill-Defined") is excluded from the
v3 label space, so this tool can never output it. If you need a model
that can output the catch-all, keep using DoctorPredictionToolV2.
"""

import json
from pathlib import Path
from typing import Type

import joblib
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths (canonical layout in proiect_licenta.paths)
# ---------------------------------------------------------------------------
# Doctor v3 reuses triage v1 base artifacts (for cascading-feature parity
# with how doctor v3 was trained in training/train_nurse_v3.py).
from proiect_licenta.paths import (
    TRIAGE_V1_DIR as MODELS_DIR,
    DOCTOR_V3_DIR as DOCTOR_MODELS_DIR,
)

from proiect_licenta.tools.triage_tool import normalize_complaint_text

# ---------------------------------------------------------------------------
# Medication classification — shared vocabulary with the training pipeline
# ---------------------------------------------------------------------------
from proiect_licenta.tools.med_vocab import (
    DRUG_NAME_MAP, MED_CLASS_KEYWORDS, MED_CATEGORIES,
    flags_from_name, flags_from_text,
)

# ---------------------------------------------------------------------------
# Reuse rhythm bucketing from the training pipeline (single source of truth)
# ---------------------------------------------------------------------------
# Importing the training-side normalizer guarantees that the same string
# (e.g., "afib", "Atrial Fibrillation", "AF") buckets to the same label
# at training and inference. Same approach as med_vocab.flags_from_row.
from proiect_licenta.training.train_nurse_v3 import (
    _normalize_rhythm,
    _RHYTHM_BUCKETS,
    _VITAL_COLS_LONG,
    LONG_VITAL_FEATURE_COLS,
)

# Vital sign medians (from training data, used when patient doesn't know)
VITAL_MEDIANS = {
    "temperature": 98.1, "heartrate": 84, "resprate": 18,
    "o2sat": 98, "sbp": 134, "dbp": 78,
}


def classify_medications(medications_raw: str) -> dict:
    """Classify patient-reported medications into category flags.

    Same logic as DoctorPredictionToolV2.classify_medications — kept here
    rather than imported so this tool has no runtime dependency on the v2
    tool (which targets a different model directory).
    """
    flags = {cat: 0 for cat in MED_CATEGORIES}

    if not medications_raw or medications_raw.lower() in (
        "unknown", "none", "no", "skip", "n/a", "na", "-", "",
        "i don't know", "i dont know", "idk",
    ):
        return {"n_medications": 0, "meds_unknown": 1, **flags}

    med_lower = medications_raw.lower()
    meds_list = [
        m.strip() for m in
        med_lower.replace(";", ",").replace(" and ", ",").split(",")
        if m.strip()
    ]
    n_meds = max(len(meds_list), 1)

    matched = flags_from_name(med_lower) | flags_from_text(med_lower)
    for cat in matched:
        if cat in flags:
            flags[cat] = 1

    return {"n_medications": n_meds, "meds_unknown": 0, **flags}


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_doctor_v3_cache = None


def get_doctor_v3_models():
    global _doctor_v3_cache
    if _doctor_v3_cache is None:
        tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
        acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
        disposition_model = joblib.load(MODELS_DIR / "disposition_model.joblib")

        diagnosis_model = joblib.load(DOCTOR_MODELS_DIR / "diagnosis_model.joblib")
        department_model = joblib.load(DOCTOR_MODELS_DIR / "department_model.joblib")

        with open(DOCTOR_MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        _doctor_v3_cache = {
            "tfidf": tfidf,
            "severity_map": severity_map,
            "acuity_model": acuity_model,
            "disposition_model": disposition_model,
            "diagnosis_model": diagnosis_model,
            "department_model": department_model,
            "metadata": metadata,
        }
    return _doctor_v3_cache


# ---------------------------------------------------------------------------
# Department names
# ---------------------------------------------------------------------------
DEPARTMENT_NAMES = {
    "MED": "General Medicine",
    "CMED": "Cardiac Medicine",
    "NMED": "Neuro Medicine",
    "SURG": "General Surgery",
    "OMED": "Oncology Medicine",
    "ORTHO": "Orthopedics",
    "NSURG": "Neurosurgery",
    "TRAUM": "Trauma",
    "OTHER_SURG": "Other Surgery (Vascular/Thoracic/Cardiac/Plastic)",
    "OB_GYN": "Obstetrics / Gynecology",
    "OTHER": "Other Specialty (Urology/Psychiatry/ENT/Eye/Dental)",
}


# ---------------------------------------------------------------------------
# Tool Input Schema
# ---------------------------------------------------------------------------
class DoctorV3Input(BaseModel):
    """Input schema for the Doctor v3 Prediction Tool (with full nurse data)."""
    chief_complaints: str = Field(
        ..., description="Comma-separated chief complaints"
    )
    pain_score: int = Field(
        ..., description="Pain score 0-10, or -1 if unknown"
    )
    age: int = Field(default=50, description="Patient age in years")
    gender: str = Field(
        default="unknown", description="'male', 'female', or 'unknown'"
    )
    arrival_transport: str = Field(
        default="unknown",
        description="'ambulance', 'walk_in', 'helicopter', or 'unknown'"
    )
    predicted_acuity: int = Field(
        ..., description="ESI acuity level 1-5 from triage"
    )
    is_admitted: bool = Field(
        ..., description="Whether patient is admitted (from triage)"
    )
    temperature: float = Field(
        default=-1.0,
        description="Body temperature in Fahrenheit (e.g., 101.2), or -1 if unknown"
    )
    heartrate: float = Field(
        default=-1.0,
        description="Heart rate in bpm (e.g., 92), or -1 if unknown"
    )
    resprate: float = Field(
        default=-1.0,
        description="Respiratory rate breaths/min (e.g., 18), or -1 if unknown"
    )
    o2sat: float = Field(
        default=-1.0,
        description="Oxygen saturation % (e.g., 97), or -1 if unknown"
    )
    sbp: float = Field(
        default=-1.0,
        description="Systolic blood pressure mmHg (e.g., 130), or -1 if unknown"
    )
    dbp: float = Field(
        default=-1.0,
        description="Diastolic blood pressure mmHg (e.g., 85), or -1 if unknown"
    )
    medications_raw: str = Field(
        default="unknown",
        description="Comma-separated medication list from patient, or 'none'/'unknown'"
    )
    rhythm: str = Field(
        default="",
        description=(
            "Cardiac rhythm reading from the monitor, free-text. "
            "Examples: 'sinus', 'sinus rhythm', 'NSR', 'atrial fibrillation', "
            "'afib', 'paced', 'sinus tachycardia'. Empty string if unknown."
        ),
    )


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------
class DoctorPredictionToolV3(BaseTool):
    name: str = "doctor_prediction_tool_v3"
    description: str = (
        "Doctor v3 prediction tool. Uses XGBoost models trained on the "
        "13-class label space (catch-all 'Symptoms/Signs/Ill-Defined' excluded), "
        "the full admitted-patient dataset, snapshot vitals, longitudinal vital "
        "trajectory features, and cardiac rhythm. "
        "Requires: chief_complaints, pain_score, predicted_acuity, is_admitted. "
        "Nurse data (all optional, use -1 / empty string for unknown): "
        "temperature, heartrate, resprate, o2sat, sbp, dbp, medications_raw, rhythm."
    )
    args_schema: Type[BaseModel] = DoctorV3Input

    def _run(
        self,
        chief_complaints: str,
        pain_score: int,
        predicted_acuity: int,
        is_admitted: bool,
        age: int = 50,
        gender: str = "unknown",
        arrival_transport: str = "unknown",
        temperature: float = -1.0,
        heartrate: float = -1.0,
        resprate: float = -1.0,
        o2sat: float = -1.0,
        sbp: float = -1.0,
        dbp: float = -1.0,
        medications_raw: str = "unknown",
        rhythm: str = "",
    ) -> str:
        """Run v3 doctor prediction with full nurse data."""

        if not is_admitted:
            return json.dumps({
                "status": "NOT_ADMITTED",
                "message": "Patient is predicted to be discharged. "
                           "Diagnosis and department prediction is only for admitted patients.",
            }, indent=2)

        models = get_doctor_v3_models()
        tfidf = models["tfidf"]
        severity_map = models["severity_map"]
        diagnosis_model = models["diagnosis_model"]
        department_model = models["department_model"]
        metadata = models["metadata"]
        diagnosis_labels = metadata["diagnosis_labels"]
        department_labels = metadata["department_labels"]

        # ── 1. Normalize complaint text ──
        complaint_text = normalize_complaint_text(chief_complaints)
        raw_complaints = [c.strip() for c in chief_complaints.split(",") if c.strip()]
        n_complaints = len(raw_complaints)
        complaint_length = len(complaint_text)

        # ── 2. TF-IDF ──
        tfidf_matrix = tfidf.transform([complaint_text])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        )

        # ── 3. Severity priors ──
        words = complaint_text.split()
        severities = [severity_map[w] for w in words if w in severity_map]
        min_sev = min(severities) if severities else 3.0
        mean_sev = float(np.mean(severities)) if severities else 3.0
        max_sev = max(severities) if severities else 3.0
        std_sev = float(np.std(severities)) if len(severities) > 1 else 0.0

        # ── 4. Pain ──
        pain_val = max(-1, min(10, pain_score))
        pain_missing = 1 if pain_val < 0 else 0
        pain_low = 1 if 0 <= pain_val <= 3 else 0
        pain_mid = 1 if 4 <= pain_val <= 6 else 0
        pain_high = 1 if 7 <= pain_val <= 10 else 0

        # ── 5. Demographics ──
        age_val = max(0, min(120, age))
        age_bins = [0, 18, 35, 50, 65, 80, 120]
        age_bin = 2
        for i, (low, high) in enumerate(zip(age_bins[:-1], age_bins[1:])):
            if low < age_val <= high:
                age_bin = i
                break

        gender_male = 1 if gender.lower() in ("male", "m") else 0

        at = arrival_transport.lower().replace(" ", "_")
        arrival_ambulance = 1 if at == "ambulance" else 0
        arrival_helicopter = 1 if at == "helicopter" else 0
        arrival_walk_in = 1 if at in ("walk_in", "walkin", "walk") else 0

        # ── 6. Interaction features ──
        pain_clipped = max(0, pain_val) if pain_val >= 0 else 0
        age_ambulance = age_val * arrival_ambulance
        pain_x_min_severity = pain_clipped * (5 - min_sev)
        age_severity = age_val * (5 - min_sev)
        high_pain_ambulance = pain_high * arrival_ambulance
        elderly = 1 if age_val >= 65 else 0
        elderly_ambulance = elderly * arrival_ambulance

        # ── 7. Assemble triage feature vector ──
        structured = pd.DataFrame({
            "pain": [pain_val],
            "pain_missing": [pain_missing],
            "pain_low": [pain_low],
            "pain_mid": [pain_mid],
            "pain_high": [pain_high],
            "n_complaints": [n_complaints],
            "complaint_length": [complaint_length],
            "min_severity_prior": [min_sev],
            "mean_severity_prior": [mean_sev],
            "max_severity_prior": [max_sev],
            "std_severity_prior": [std_sev],
            "age": [age_val],
            "age_bin": [float(age_bin)],
            "gender_male": [gender_male],
            "arrival_ambulance": [arrival_ambulance],
            "arrival_helicopter": [arrival_helicopter],
            "arrival_walk_in": [arrival_walk_in],
            "age_ambulance": [age_ambulance],
            "pain_x_min_severity": [pain_x_min_severity],
            "age_severity": [age_severity],
            "high_pain_ambulance": [high_pain_ambulance],
            "elderly": [elderly],
            "elderly_ambulance": [elderly_ambulance],
        })

        triage_features = pd.concat([structured, tfidf_df], axis=1)
        triage_features["predicted_acuity"] = predicted_acuity
        triage_features["predicted_disposition"] = 1  # always admitted

        # ── 8. Snapshot vital sign features ──
        vitals_raw = {
            "temperature": temperature,
            "heartrate": heartrate,
            "resprate": resprate,
            "o2sat": o2sat,
            "sbp": sbp,
            "dbp": dbp,
        }

        vital_values = {}
        vital_missing = {}
        for vname, vval in vitals_raw.items():
            if vval is None or vval < 0:
                vital_missing[f"{vname}_missing"] = 1
                vital_values[vname] = VITAL_MEDIANS[vname]
            else:
                vital_missing[f"{vname}_missing"] = 0
                vital_values[vname] = vval

        # Clip to plausible ranges (must mirror _clean_vitals at training time)
        vital_values["temperature"] = float(np.clip(vital_values["temperature"], 90, 110))
        vital_values["heartrate"] = float(np.clip(vital_values["heartrate"], 20, 250))
        vital_values["resprate"] = float(np.clip(vital_values["resprate"], 4, 60))
        vital_values["o2sat"] = float(np.clip(vital_values["o2sat"], 50, 100))
        vital_values["sbp"] = float(np.clip(vital_values["sbp"], 50, 300))
        vital_values["dbp"] = float(np.clip(vital_values["dbp"], 20, 200))

        # Clinical flags
        fever = 1 if vital_values["temperature"] > 100.4 else 0
        tachycardia = 1 if vital_values["heartrate"] > 100 else 0
        bradycardia = 1 if vital_values["heartrate"] < 60 else 0
        tachypnea = 1 if vital_values["resprate"] > 20 else 0
        hypoxia = 1 if vital_values["o2sat"] < 94 else 0
        hypertension = 1 if vital_values["sbp"] > 140 else 0
        hypotension = 1 if vital_values["sbp"] < 90 else 0
        map_val = (vital_values["sbp"] + 2 * vital_values["dbp"]) / 3

        vitals_df = pd.DataFrame({
            "temperature": [vital_values["temperature"]],
            "heartrate": [vital_values["heartrate"]],
            "resprate": [vital_values["resprate"]],
            "o2sat": [vital_values["o2sat"]],
            "sbp": [vital_values["sbp"]],
            "dbp": [vital_values["dbp"]],
            "temperature_missing": [vital_missing["temperature_missing"]],
            "heartrate_missing": [vital_missing["heartrate_missing"]],
            "resprate_missing": [vital_missing["resprate_missing"]],
            "o2sat_missing": [vital_missing["o2sat_missing"]],
            "sbp_missing": [vital_missing["sbp_missing"]],
            "dbp_missing": [vital_missing["dbp_missing"]],
            "fever": [fever],
            "tachycardia": [tachycardia],
            "bradycardia": [bradycardia],
            "tachypnea": [tachypnea],
            "hypoxia": [hypoxia],
            "hypertension": [hypertension],
            "hypotension": [hypotension],
            "map": [map_val],
        })

        # ── 9. Medication features ──
        med_info = classify_medications(medications_raw)
        meds_df = pd.DataFrame({
            "n_medications": [med_info["n_medications"]],
            "meds_unknown": [med_info["meds_unknown"]],
            "has_cardiac_meds": [med_info["has_cardiac_meds"]],
            "has_diabetes_meds": [med_info["has_diabetes_meds"]],
            "has_psych_meds": [med_info["has_psych_meds"]],
            "has_respiratory_meds": [med_info["has_respiratory_meds"]],
            "has_opioid_meds": [med_info["has_opioid_meds"]],
            "has_anticoagulant_meds": [med_info["has_anticoagulant_meds"]],
            "has_gi_meds": [med_info["has_gi_meds"]],
            "has_thyroid_meds": [med_info["has_thyroid_meds"]],
            "has_anticonvulsant_meds": [med_info["has_anticonvulsant_meds"]],
        })

        # ── 10. Longitudinal vitals (degraded trajectory from snapshot) ──
        # The nurse only collected one reading; we synthesize the trajectory
        # features as if min == max == last == snapshot, delta == 0, and
        # the abnormal-reading counts == the corresponding clinical flag.
        # This matches the fall-back path in train_nurse_v3._fill_longitudinal_vitals.
        clinical_flags = {
            "n_fever_readings": fever,
            "n_tachycardia_readings": tachycardia,
            "n_bradycardia_readings": bradycardia,
            "n_tachypnea_readings": tachypnea,
            "n_hypoxia_readings": hypoxia,
            "n_hypertension_readings": hypertension,
            "n_hypotension_readings": hypotension,
        }
        long_data = {}
        for vital in _VITAL_COLS_LONG:
            v = vital_values[vital]
            long_data[f"{vital}_min"] = v
            long_data[f"{vital}_max"] = v
            long_data[f"{vital}_last"] = v
            long_data[f"{vital}_delta"] = 0.0
        long_data.update(clinical_flags)

        # Rhythm one-hot (nurse-provided string -> bucket)
        bucket = _normalize_rhythm(rhythm) if rhythm else ""
        for b in _RHYTHM_BUCKETS:
            long_data[f"rhythm_{b}"] = 1 if bucket == b else 0
        long_data["rhythm_irregular"] = 1 if bucket and bucket != "sinus" else 0
        long_data["has_longitudinal_vitals"] = 0  # nurse provides snapshot only

        long_df = pd.DataFrame(
            {col: [long_data[col]] for col in LONG_VITAL_FEATURE_COLS}
        )

        # ── 11. Assemble full feature vector ──
        features = pd.concat(
            [triage_features, vitals_df, meds_df, long_df],
            axis=1,
        )

        # ── 12. Predict diagnosis ──
        diag_pred_idx = int(diagnosis_model.predict(features)[0])
        diag_proba = diagnosis_model.predict_proba(features)[0]
        diag_label = diagnosis_labels[diag_pred_idx]
        diag_confidence = float(diag_proba[diag_pred_idx])

        top3_diag_idx = np.argsort(diag_proba)[::-1][:3]
        top3_diagnoses = [
            {"category": diagnosis_labels[i], "probability": f"{diag_proba[i]:.1%}"}
            for i in top3_diag_idx
        ]

        # ── 13. Predict department (cascading) ──
        features_dept = features.copy()
        features_dept["predicted_diagnosis"] = diag_pred_idx

        dept_pred_idx = int(department_model.predict(features_dept)[0])
        dept_proba = department_model.predict_proba(features_dept)[0]
        dept_label = department_labels[dept_pred_idx]
        dept_confidence = float(dept_proba[dept_pred_idx])
        dept_full_name = DEPARTMENT_NAMES.get(dept_label, dept_label)

        top3_dept_idx = np.argsort(dept_proba)[::-1][:3]
        top3_departments = [
            {
                "code": department_labels[i],
                "name": DEPARTMENT_NAMES.get(department_labels[i], department_labels[i]),
                "probability": f"{dept_proba[i]:.1%}",
            }
            for i in top3_dept_idx
        ]

        # ── Summarize nurse data used ──
        vitals_provided = {
            k: v for k, v in vitals_raw.items()
            if v is not None and v >= 0
        }
        clinical_flags_active = []
        if fever: clinical_flags_active.append("Fever")
        if tachycardia: clinical_flags_active.append("Tachycardia")
        if bradycardia: clinical_flags_active.append("Bradycardia")
        if tachypnea: clinical_flags_active.append("Tachypnea")
        if hypoxia: clinical_flags_active.append("Hypoxia")
        if hypertension: clinical_flags_active.append("Hypertension")
        if hypotension: clinical_flags_active.append("Hypotension")

        med_flags_active = [
            cat.replace("has_", "").replace("_meds", "").replace("_", " ").title()
            for cat in MED_CATEGORIES if med_info[cat] == 1
        ]

        # ── Build result ──
        result = {
            "patient_summary": {
                "chief_complaints": raw_complaints,
                "normalized_text": complaint_text,
                "age": age_val,
                "gender": gender,
                "pain_score": pain_val,
                "arrival_transport": arrival_transport,
                "triage_acuity": predicted_acuity,
                "admission_status": "ADMITTED",
            },
            "nurse_data_used": {
                "vitals_provided": vitals_provided,
                "vitals_missing": [
                    k.replace("_missing", "")
                    for k, v in vital_missing.items() if v == 1
                ],
                "clinical_flags": clinical_flags_active if clinical_flags_active else ["None"],
                "rhythm_raw": rhythm if rhythm else None,
                "rhythm_bucket": bucket if bucket else None,
                "rhythm_irregular": bool(long_data["rhythm_irregular"]),
                "medications_count": med_info["n_medications"],
                "medication_categories_detected": med_flags_active if med_flags_active else ["None"],
                "medications_unknown": bool(med_info["meds_unknown"]),
            },
            "diagnosis_prediction": {
                "predicted_category": diag_label,
                "confidence": f"{diag_confidence:.1%}",
                "top_3_categories": top3_diagnoses,
                "label_space": f"{len(diagnosis_labels)} classes (catch-all excluded)",
            },
            "department_prediction": {
                "predicted_department": dept_label,
                "department_full_name": dept_full_name,
                "confidence": f"{dept_confidence:.1%}",
                "top_3_departments": top3_departments,
            },
        }

        return json.dumps(result, indent=2, ensure_ascii=False)
