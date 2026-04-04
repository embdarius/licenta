"""
Triage Prediction Tool v3 — CrewAI Tool

Uses TF-IDF + XGBoost models with demographics for triage predictions.
Takes chief complaints, pain score, age, gender, arrival transport.
Returns ESI acuity level (1-5), admission/discharge, confidence scores.
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
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


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

        with open(MODELS_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        _models_cache = (acuity_model, disposition_model, tfidf, severity_map, metadata)
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
    """Input schema for the Triage Prediction Tool."""
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
        "Optional inputs: age (int, default 50), gender ('male'/'female'/'unknown'), "
        "arrival_transport ('ambulance'/'walk_in'/'helicopter'/'unknown'). "
        "Uses an XGBoost model trained on 400K+ MIMIC-IV ED visits."
    )
    args_schema: Type[BaseModel] = TriageInput

    def _run(
        self,
        chief_complaints: str,
        pain_score: int,
        age: int = 50,
        gender: str = "unknown",
        arrival_transport: str = "unknown",
    ) -> str:
        """Run triage prediction."""
        acuity_model, disposition_model, tfidf, severity_map, metadata = get_models()

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
        age_bin = 2  # default
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

        # 7. Assemble feature vector (must match training order!)
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
        })
        features = pd.concat([structured, tfidf_df], axis=1)

        # 8. Predict acuity
        acuity_pred_shifted = int(acuity_model.predict(features)[0])
        acuity_pred = acuity_pred_shifted + 1
        acuity_proba = acuity_model.predict_proba(features)[0]
        acuity_confidence = float(acuity_proba[acuity_pred_shifted])

        # 9. Predict disposition
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
