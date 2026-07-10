"""Doctor prediction tool (v3 base): initial pre-nurse assessment.

Diagnosis category (13 classes, catch-all excluded) and hospital department
(11 classes). Functionally identical to v1 but loads the v3_base artifacts from
artifacts/doctor/v3_base/, trained on the full filtered admitted-patient set.

The cascade source is triage v1, matching what train_doctor_v3.py used at
training time. The runtime triage agent uses triage v3, but the doctor
diagnosis/department models were trained against the v1 cascade and must keep
that input distribution at inference for train/inference symmetry. The doctor
disposition model uses triage v3 directly because it was trained that way.
"""

import json
from pathlib import Path
from typing import Type

import joblib
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Paths
from proiect_licenta.paths import (
    TRIAGE_V1_DIR as MODELS_DIR,
    DOCTOR_V3_BASE_DIR as DOCTOR_MODELS_DIR,
)
from proiect_licenta.tools.triage_tool import normalize_complaint_text


# Lazy model loading (per-process cache)
_doctor_cache = None


def get_doctor_models():
    global _doctor_cache
    if _doctor_cache is None:
        tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
        acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
        disposition_model = joblib.load(MODELS_DIR / "disposition_model.joblib")

        diagnosis_model = joblib.load(DOCTOR_MODELS_DIR / "diagnosis_model.joblib")
        department_model = joblib.load(DOCTOR_MODELS_DIR / "department_model.joblib")

        with open(DOCTOR_MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        _doctor_cache = {
            "tfidf": tfidf,
            "severity_map": severity_map,
            "acuity_model": acuity_model,
            "disposition_model": disposition_model,
            "diagnosis_model": diagnosis_model,
            "department_model": department_model,
            "metadata": metadata,
        }
    return _doctor_cache


# Display name lookup (kept in sync with v1/v2/v3 nurse tools)
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


# Tool Input Schema
class DoctorV3BaseInput(BaseModel):
    """Input schema for the Doctor v3 base prediction tool."""
    chief_complaints: str = Field(
        ...,
        description="Comma-separated chief complaints, e.g. 'chest pain, dyspnea'",
    )
    pain_score: int = Field(..., description="Pain score 0-10, or -1 if unknown.")
    age: int = Field(default=50, description="Patient age in years.")
    gender: str = Field(
        default="unknown",
        description="Patient gender: 'male', 'female', or 'unknown'.",
    )
    arrival_transport: str = Field(
        default="unknown",
        description="'ambulance', 'walk_in', 'helicopter', or 'unknown'.",
    )
    predicted_acuity: int = Field(
        ..., description="ESI acuity level (1-5) from the triage agent.",
    )
    is_admitted: bool = Field(
        ..., description="Whether the triage agent predicted the patient should be admitted.",
    )


# CrewAI Tool
class DoctorPredictionToolV3Base(BaseTool):
    name: str = "doctor_prediction_tool_v3_base"
    description: str = (
        "Predicts the diagnosis category (13 classes, catch-all excluded) and "
        "hospital department (11 classes) for admitted patients using the v3_base "
        "XGBoost models. Pre-nurse: triage data only, no vitals/medications/PMH. "
        "Requires: chief_complaints (comma-separated), pain_score (0-10 or -1), "
        "predicted_acuity (1-5 from triage), is_admitted (bool from triage). "
        "Optional: age, gender, arrival_transport. "
        "Returns JSON with diagnosis + department predictions, both top-3 lists "
        "with probabilities. Only call this tool if the patient is predicted to "
        "be ADMITTED. For the post-nurse refined assessment use "
        "doctor_prediction_tool_v3 (with nurse data)."
    )
    args_schema: Type[BaseModel] = DoctorV3BaseInput

    def _run(
        self,
        chief_complaints: str,
        pain_score: int,
        predicted_acuity: int,
        is_admitted: bool,
        age: int = 50,
        gender: str = "unknown",
        arrival_transport: str = "unknown",
    ) -> str:
        if not is_admitted:
            return json.dumps({
                "status": "NOT_ADMITTED",
                "message": (
                    "Patient is predicted to be discharged. Diagnosis and "
                    "department prediction is only for admitted patients."
                ),
            }, indent=2)

        models = get_doctor_models()
        tfidf = models["tfidf"]
        severity_map = models["severity_map"]
        diagnosis_model = models["diagnosis_model"]
        department_model = models["department_model"]
        metadata = models["metadata"]

        diagnosis_labels = metadata["diagnosis_labels"]
        department_labels = metadata["department_labels"]

        # 1. Normalize complaint text
        complaint_text = normalize_complaint_text(chief_complaints)
        raw_complaints = [c.strip() for c in chief_complaints.split(",") if c.strip()]
        n_complaints = len(raw_complaints)
        complaint_length = len(complaint_text)

        # 2. TF-IDF
        tfidf_matrix = tfidf.transform([complaint_text])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        )

        # 3. Severity priors
        words = complaint_text.split()
        severities = [severity_map[w] for w in words if w in severity_map]
        min_sev = min(severities) if severities else 3.0
        mean_sev = float(np.mean(severities)) if severities else 3.0
        max_sev = max(severities) if severities else 3.0
        std_sev = float(np.std(severities)) if len(severities) > 1 else 0.0

        # 4. Pain
        pain_val = max(-1, min(10, pain_score))
        pain_missing = 1 if pain_val < 0 else 0
        pain_low = 1 if 0 <= pain_val <= 3 else 0
        pain_mid = 1 if 4 <= pain_val <= 6 else 0
        pain_high = 1 if 7 <= pain_val <= 10 else 0

        # 5. Demographics
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

        # 6. Interaction features
        pain_clipped = max(0, pain_val) if pain_val >= 0 else 0
        age_ambulance = age_val * arrival_ambulance
        pain_x_min_severity = pain_clipped * (5 - min_sev)
        age_severity = age_val * (5 - min_sev)
        high_pain_ambulance = pain_high * arrival_ambulance
        elderly = 1 if age_val >= 65 else 0
        elderly_ambulance = elderly * arrival_ambulance

        # 7. Assemble feature vector
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

        features = pd.concat([structured, tfidf_df], axis=1)
        features["predicted_acuity"] = predicted_acuity
        features["predicted_disposition"] = 1  # always 1 (gate is is_admitted)

        # 8. Diagnosis prediction (13 classes)
        diag_proba = diagnosis_model.predict_proba(features)[0]
        diag_pred_idx = int(np.argmax(diag_proba))
        diag_label = diagnosis_labels[diag_pred_idx]
        diag_confidence = float(diag_proba[diag_pred_idx])
        top3_diag_idx = np.argsort(diag_proba)[::-1][:3]
        top3_diagnoses = [
            {
                "category": diagnosis_labels[i],
                "probability_raw": float(diag_proba[i]),
                "probability": f"{diag_proba[i]:.1%}",
            }
            for i in top3_diag_idx
        ]

        # 9. Department prediction (cascading from diagnosis)
        features_dept = features.copy()
        features_dept["predicted_diagnosis"] = diag_pred_idx
        dept_proba = department_model.predict_proba(features_dept)[0]
        dept_pred_idx = int(np.argmax(dept_proba))
        dept_label = department_labels[dept_pred_idx]
        dept_confidence = float(dept_proba[dept_pred_idx])
        dept_full_name = DEPARTMENT_NAMES.get(dept_label, dept_label)
        top3_dept_idx = np.argsort(dept_proba)[::-1][:3]
        top3_departments = [
            {
                "code": department_labels[i],
                "name": DEPARTMENT_NAMES.get(department_labels[i], department_labels[i]),
                "probability_raw": float(dept_proba[i]),
                "probability": f"{dept_proba[i]:.1%}",
            }
            for i in top3_dept_idx
        ]

        # 10. Result
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
            "diagnosis_prediction": {
                "predicted_category": diag_label,
                "confidence": f"{diag_confidence:.1%}",
                "top_3_categories": top3_diagnoses,
            },
            "department_prediction": {
                "predicted_department": dept_label,
                "department_full_name": dept_full_name,
                "confidence": f"{dept_confidence:.1%}",
                "top_3_departments": top3_departments,
            },
            "label_space": {
                "diagnosis_n_classes": len(diagnosis_labels),
                "note": (
                    "v3_base label space excludes the 'Symptoms, Signs, "
                    "Ill-Defined' catch-all category that v1/v2 retained. "
                    "Predictions therefore route an otherwise-catch-all "
                    "patient to the closest non-catch-all category."
                ),
            },
            "model_version": "v3_base (pre-nurse, 13-class)",
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
