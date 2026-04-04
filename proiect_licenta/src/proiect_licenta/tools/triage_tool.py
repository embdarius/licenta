"""
Triage Prediction Tool v2 — CrewAI Tool

Uses TF-IDF + XGBoost models for triage predictions.
Takes structured chief complaints + pain score, returns:
  - ESI acuity level (1-5)
  - Admission/discharge prediction
  - Confidence scores
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
    "etoh": "alcohol",
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
}


def normalize_complaint_text(text: str) -> str:
    """Normalize complaint text to match training pipeline preprocessing."""
    if not text or not text.strip():
        return ""

    text = text.lower().strip()
    text = text.replace(",", " ").replace(";", " ").replace("/", " ").replace("-", " ")

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


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------
class TriagePredictionTool(BaseTool):
    name: str = "triage_prediction_tool"
    description: str = (
        "Predicts the Emergency Severity Index (ESI) acuity level (1-5) and "
        "whether the patient should be admitted or discharged based on their "
        "chief complaints and pain score. Input: comma-separated chief "
        "complaints and a pain score (0-10, or -1 if unknown). This tool uses "
        "an XGBoost model trained on 400,000+ real emergency department visits "
        "from the MIMIC-IV dataset."
    )
    args_schema: Type[BaseModel] = TriageInput

    def _run(self, chief_complaints: str, pain_score: int) -> str:
        """Run triage prediction."""
        acuity_model, disposition_model, tfidf, severity_map, metadata = get_models()

        # 1. Normalize complaint text (same as training)
        complaint_text = normalize_complaint_text(chief_complaints)

        # 2. Count complaints
        raw_complaints = [c.strip() for c in chief_complaints.split(",") if c.strip()]
        n_complaints = len(raw_complaints)

        # 3. TF-IDF transform
        tfidf_matrix = tfidf.transform([complaint_text])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
        )

        # 4. Severity priors
        words = complaint_text.split()
        severities = [severity_map[w] for w in words if w in severity_map]
        max_severity = min(severities) if severities else 3.0
        mean_severity = float(np.mean(severities)) if severities else 3.0

        # 5. Pain score
        pain_val = max(-1, min(10, pain_score))

        # 6. Assemble feature vector
        engineered = pd.DataFrame({
            "pain": [pain_val],
            "n_complaints": [n_complaints],
            "max_severity_prior": [max_severity],
            "mean_severity_prior": [mean_severity],
        })
        features = pd.concat([engineered, tfidf_df], axis=1)

        # 7. Predict acuity (model outputs 0-4, add 1 for ESI 1-5)
        acuity_pred_shifted = int(acuity_model.predict(features)[0])
        acuity_pred = acuity_pred_shifted + 1
        acuity_proba = acuity_model.predict_proba(features)[0]

        acuity_confidence = float(acuity_proba[acuity_pred_shifted])

        # 8. Predict disposition (add predicted acuity as feature)
        features_disp = features.copy()
        features_disp["predicted_acuity"] = acuity_pred

        disp_pred = int(disposition_model.predict(features_disp)[0])
        disp_proba = disposition_model.predict_proba(features_disp)[0]
        disp_confidence = float(disp_proba[disp_pred])

        disposition_text = "ADMITTED" if disp_pred == 1 else "NOT ADMITTED (DISCHARGE)"

        # Build probability breakdown
        acuity_breakdown = {}
        for i, prob in enumerate(acuity_proba):
            esi = i + 1
            acuity_breakdown[f"ESI {esi}"] = f"{prob:.1%}"

        result = {
            "complaint_analysis": {
                "input_complaints": raw_complaints,
                "normalized_text": complaint_text,
                "n_complaints": n_complaints,
                "max_severity_prior": round(max_severity, 2),
                "mean_severity_prior": round(mean_severity, 2),
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
