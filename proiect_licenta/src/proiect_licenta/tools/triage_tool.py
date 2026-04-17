"""
Triage Prediction Tool v2 — CrewAI Tool

Uses TF-IDF + XGBoost models with demographics and vital signs for triage predictions.
Takes chief complaints, pain score, age, gender, arrival transport, and optional vitals.
Returns ESI acuity level (1-5), admission/discharge, confidence scores.

Loads models from models_v2/ (triage pipeline v2 with vital signs).
"""

import json
from pathlib import Path
from typing import Optional, Type

import joblib
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models_v2"


# ---------------------------------------------------------------------------
# Complaint text normalization (must match training pipeline)
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    "abd": "abdominal",
    "n/v": "nausea vomiting",
    "n v": "nausea vomiting",
    "s/p": "status post",
    "s p": "status post",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "ha": "headache",
    "ams": "altered mental status",
    "loc": "loss of consciousness",
    "etoh": "alcohol intoxication",
    "uti": "urinary tract infection",
    "uri": "upper respiratory infection",
    "mv": "motor vehicle",
    "mva": "motor vehicle accident",
    "mvc": "motor vehicle collision",
    "htn": "hypertension",
    "dm": "diabetes",
    "chf": "congestive heart failure",
    "gi": "gastrointestinal",
    "r/o": "rule out",
    "w/": "with",
    "w/o": "without",
    "fx": "fracture",
    "lac": "laceration",
    "inj": "injury",
    "sx": "symptoms",
    "dx": "diagnosis",
    "tx": "treatment",
    "hx": "history",
    "bld": "blood",
    "diff": "difficulty",
    "eval": "evaluation",
    "sz": "seizure",
    "ped": "pediatric",
    "psych": "psychiatric",
    "resp": "respiratory",
    "bilat": "bilateral",
    "lt": "left",
    "rt": "right",
    "pos": "positive",
    "neg": "negative",
    "hiv": "hiv",
    "copd": "chronic obstructive pulmonary disease",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident stroke",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "ble": "bleeding",
}


def normalize_complaint_text(text: str) -> str:
    """Normalize complaint text to match training pipeline."""
    if not text or not text.strip():
        return ""
    text = text.lower().strip()
    text = text.replace(",", " ").replace(";", " ").replace("/", " ").replace("-", " ")
    text = text.replace("(", " ").replace(")", " ").replace(".", " ")

    words = text.split()
    expanded = []
    for word in words:
        word = word.strip()
        if word in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[word])
        elif len(word) > 1:
            expanded.append(word)
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# Vital sign constants (must match triage_pipeline_v2.py)
# ---------------------------------------------------------------------------
VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

VITAL_CLIP_RANGES = {
    "temperature": (90.0, 110.0),
    "heartrate":   (20.0, 250.0),
    "resprate":    (4.0, 60.0),
    "o2sat":       (50.0, 100.0),
    "sbp":         (50.0, 300.0),
    "dbp":         (20.0, 200.0),
}

ABNORMALITY_THRESHOLDS = {
    "fever":        ("temperature", ">",  100.4),
    "hypothermic":  ("temperature", "<",  96.8),
    "tachycardic":  ("heartrate",   ">",  100),
    "bradycardic":  ("heartrate",   "<",  60),
    "tachypneic":   ("resprate",    ">",  20),
    "hypoxic":      ("o2sat",       "<",  94),
    "hypertensive": ("sbp",         ">",  140),
    "hypotensive":  ("sbp",         "<",  90),
}


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_models_cache = None


def get_models():
    global _models_cache
    if _models_cache is None:
        acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
        disposition_model = joblib.load(MODELS_DIR / "disposition_model.joblib")
        tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
        vital_medians = joblib.load(MODELS_DIR / "vital_medians.joblib")

        with open(MODELS_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        _models_cache = (acuity_model, disposition_model, tfidf, severity_map,
                         vital_medians, metadata)
    return _models_cache


# ---------------------------------------------------------------------------
# Acuity level descriptions
# ---------------------------------------------------------------------------
ACUITY_DESCRIPTIONS = {
    1: "Resuscitation -- Immediate life-saving intervention required",
    2: "Emergent -- High risk, confused/lethargic, severe pain/distress",
    3: "Urgent -- Multiple resources needed but stable vital signs",
    4: "Less Urgent -- One resource expected (e.g., sutures, X-ray)",
    5: "Non-Urgent -- No resources expected, may be referred to clinic",
}


# ---------------------------------------------------------------------------
# Tool Input Schema
# ---------------------------------------------------------------------------
class TriageInput(BaseModel):
    """Input schema for the Triage Prediction Tool v2."""
    chief_complaints: str = Field(
        ...,
        description="Comma-separated list of chief complaints, "
                    "e.g., 'abd pain, headache, nausea'"
    )
    pain_score: int = Field(
        ...,
        description="Pain score from 0 (no pain) to 10 (worst pain). "
                    "Use -1 if unknown."
    )
    age: int = Field(
        default=50,
        description="Patient age in years. Use 50 if unknown."
    )
    gender: str = Field(
        default="unknown",
        description="Patient gender: 'male', 'female', or 'unknown'."
    )
    arrival_transport: str = Field(
        default="unknown",
        description="How patient arrived: 'ambulance', 'walk_in', "
                    "'helicopter', or 'unknown'."
    )
    temperature: float = Field(
        default=-1.0,
        description="Body temperature in Fahrenheit. Use -1 if unknown."
    )
    heartrate: float = Field(
        default=-1.0,
        description="Heart rate in bpm. Use -1 if unknown."
    )
    resprate: float = Field(
        default=-1.0,
        description="Respiratory rate in breaths/min. Use -1 if unknown."
    )
    o2sat: float = Field(
        default=-1.0,
        description="Oxygen saturation percentage (0-100). Use -1 if unknown."
    )
    sbp: float = Field(
        default=-1.0,
        description="Systolic blood pressure in mmHg. Use -1 if unknown."
    )
    dbp: float = Field(
        default=-1.0,
        description="Diastolic blood pressure in mmHg. Use -1 if unknown."
    )


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------
class TriagePredictionTool(BaseTool):
    name: str = "triage_prediction_tool"
    description: str = (
        "Predicts the Emergency Severity Index (ESI) acuity level (1-5) and "
        "whether the patient should be admitted or discharged. "
        "Required inputs: chief_complaints (comma-separated string), "
        "pain_score (0-10 or -1 if unknown). "
        "Optional inputs: age, gender, arrival_transport, "
        "temperature (°F), heartrate (bpm), resprate, o2sat (%), sbp, dbp (mmHg). "
        "Vital signs default to -1 (unknown) — provide them when available "
        "(e.g., ambulance/helicopter patients with EMS vitals). "
        "Uses XGBoost models trained on 100K+ MIMIC-IV ED visits with vital signs."
    )
    args_schema: Type[BaseModel] = TriageInput

    def _run(
        self,
        chief_complaints: str,
        pain_score: int,
        age: int = 50,
        gender: str = "unknown",
        arrival_transport: str = "unknown",
        temperature: float = -1.0,
        heartrate: float = -1.0,
        resprate: float = -1.0,
        o2sat: float = -1.0,
        sbp: float = -1.0,
        dbp: float = -1.0,
    ) -> str:
        """Run triage prediction with optional vital signs."""
        (acuity_model, disposition_model, tfidf, severity_map,
         vital_medians, metadata) = get_models()

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

        # 6. Arrival transport
        at = arrival_transport.lower().replace(" ", "_")
        arrival_ambulance = 1 if at == "ambulance" else 0
        arrival_helicopter = 1 if at == "helicopter" else 0
        arrival_walk_in = 1 if at in ("walk_in", "walkin", "walk") else 0

        # 7. v1 interaction features
        pain_clipped = max(0, pain_val) if pain_val >= 0 else 0
        age_ambulance = age_val * arrival_ambulance
        pain_x_min_severity = pain_clipped * (5 - min_sev)
        age_severity = age_val * (5 - min_sev)
        high_pain_ambulance = pain_high * arrival_ambulance
        elderly = 1 if age_val >= 65 else 0
        elderly_ambulance = elderly * arrival_ambulance

        # 8. Vital signs — resolve missing values
        raw_vitals = {
            "temperature": temperature,
            "heartrate": heartrate,
            "resprate": resprate,
            "o2sat": o2sat,
            "sbp": sbp,
            "dbp": dbp,
        }

        vital_values = {}
        vital_missing_flags = {}
        for col in VITAL_COLS:
            val = raw_vitals[col]
            lo, hi = VITAL_CLIP_RANGES[col]
            # Treat -1 or out-of-range as missing
            if val < 0 or val < lo or val > hi:
                vital_missing_flags[f"{col}_missing"] = 1
                vital_values[col] = vital_medians[col]
            else:
                vital_missing_flags[f"{col}_missing"] = 0
                vital_values[col] = val

        # 9. Abnormality flags (computed on imputed values)
        abnormality_flags = {}
        for flag_name, (col, op, threshold) in ABNORMALITY_THRESHOLDS.items():
            v = vital_values[col]
            if op == ">":
                abnormality_flags[flag_name] = 1 if v > threshold else 0
            else:
                abnormality_flags[flag_name] = 1 if v < threshold else 0

        abnormal_vital_count = sum(abnormality_flags.values())

        # 10. Vital interaction features
        tachycardic = abnormality_flags["tachycardic"]
        hypoxic = abnormality_flags["hypoxic"]
        hypotensive = abnormality_flags["hypotensive"]
        fever = abnormality_flags["fever"]

        tachycardic_ambulance = tachycardic * arrival_ambulance
        hypoxic_ambulance = hypoxic * arrival_ambulance
        hypotensive_ambulance = hypotensive * arrival_ambulance
        fever_ambulance = fever * arrival_ambulance
        tachycardic_elderly = tachycardic * elderly
        hypoxic_elderly = hypoxic * elderly
        hypotensive_elderly = hypotensive * elderly

        # 11. Assemble feature vector (must match training order!)
        structured = pd.DataFrame({
            # v1 features
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
            # v2 — raw vitals
            "temperature": [vital_values["temperature"]],
            "heartrate": [vital_values["heartrate"]],
            "resprate": [vital_values["resprate"]],
            "o2sat": [vital_values["o2sat"]],
            "sbp": [vital_values["sbp"]],
            "dbp": [vital_values["dbp"]],
            # v2 — missing flags
            "temperature_missing": [vital_missing_flags["temperature_missing"]],
            "heartrate_missing": [vital_missing_flags["heartrate_missing"]],
            "resprate_missing": [vital_missing_flags["resprate_missing"]],
            "o2sat_missing": [vital_missing_flags["o2sat_missing"]],
            "sbp_missing": [vital_missing_flags["sbp_missing"]],
            "dbp_missing": [vital_missing_flags["dbp_missing"]],
            # v2 — abnormality flags
            "fever": [abnormality_flags["fever"]],
            "hypothermic": [abnormality_flags["hypothermic"]],
            "tachycardic": [tachycardic],
            "bradycardic": [abnormality_flags["bradycardic"]],
            "tachypneic": [abnormality_flags["tachypneic"]],
            "hypoxic": [hypoxic],
            "hypertensive": [abnormality_flags["hypertensive"]],
            "hypotensive": [hypotensive],
            # v2 — abnormal count
            "abnormal_vital_count": [abnormal_vital_count],
            # v2 — vital-transport interactions
            "tachycardic_ambulance": [tachycardic_ambulance],
            "hypoxic_ambulance": [hypoxic_ambulance],
            "hypotensive_ambulance": [hypotensive_ambulance],
            "fever_ambulance": [fever_ambulance],
            # v2 — vital-age interactions
            "tachycardic_elderly": [tachycardic_elderly],
            "hypoxic_elderly": [hypoxic_elderly],
            "hypotensive_elderly": [hypotensive_elderly],
        })
        features = pd.concat([structured, tfidf_df], axis=1)

        # 12. Predict acuity
        acuity_pred_shifted = int(acuity_model.predict(features)[0])
        acuity_pred = acuity_pred_shifted + 1
        acuity_proba = acuity_model.predict_proba(features)[0]
        acuity_confidence = float(acuity_proba[acuity_pred_shifted])

        # 13. Predict disposition (cascading)
        features_disp = features.copy()
        features_disp["predicted_acuity"] = acuity_pred

        disp_pred = int(disposition_model.predict(features_disp)[0])
        disp_proba = disposition_model.predict_proba(features_disp)[0]
        disp_confidence = float(disp_proba[disp_pred])
        disposition_text = "ADMITTED" if disp_pred == 1 else "NOT ADMITTED (DISCHARGE)"

        # Build result
        acuity_breakdown = {
            f"ESI {i + 1}": f"{prob:.1%}"
            for i, prob in enumerate(acuity_proba)
        }

        # Summarize which vitals were provided vs imputed
        vitals_provided = {
            col: vital_values[col]
            for col in VITAL_COLS
            if vital_missing_flags[f"{col}_missing"] == 0
        }
        vitals_imputed = [
            col for col in VITAL_COLS
            if vital_missing_flags[f"{col}_missing"] == 1
        ]

        # List abnormalities detected
        abnormalities_detected = [
            flag for flag, val in abnormality_flags.items() if val == 1
        ]

        result = {
            "complaint_analysis": {
                "input_complaints": raw_complaints,
                "normalized_text": complaint_text,
                "n_complaints": n_complaints,
                "min_severity_prior": round(min_sev, 2),
                "mean_severity_prior": round(mean_sev, 2),
            },
            "patient_info": {
                "age": age_val,
                "gender": gender,
                "arrival_transport": arrival_transport,
            },
            "vital_signs": {
                "provided": vitals_provided,
                "imputed_as_missing": vitals_imputed,
                "abnormalities_detected": abnormalities_detected,
                "abnormal_vital_count": abnormal_vital_count,
            },
            "acuity_prediction": {
                "predicted_esi_level": acuity_pred,
                "description": ACUITY_DESCRIPTIONS.get(acuity_pred, "Unknown"),
                "confidence": f"{acuity_confidence:.1%}",
                "probability_breakdown": acuity_breakdown,
            },
            "disposition_prediction": {
                "prediction": disposition_text,
                "confidence": f"{disp_confidence:.1%}",
            },
            "pain_score_used": pain_val,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)
