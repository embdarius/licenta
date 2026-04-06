import json
from pathlib import Path
from typing import Type
import joblib
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import torch
from sentence_transformers import SentenceTransformer

from proiect_licenta.data_pipeline import build_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_models_cache = None

def get_doctor_models():
    global _models_cache
    if _models_cache is None:
        diag_model = joblib.load(MODELS_DIR / "diagnosis_model.joblib")
        diag_le = joblib.load(MODELS_DIR / "diagnosis_le.joblib")
        srv_model = joblib.load(MODELS_DIR / "service_model.joblib")
        srv_le = joblib.load(MODELS_DIR / "service_le.joblib")
        bert_name = joblib.load(MODELS_DIR / "doctor_bert_name.joblib")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_model = SentenceTransformer(bert_name, device=device)
        
        _models_cache = (diag_model, diag_le, srv_model, srv_le, bert_model)
    return _models_cache

class DoctorInput(BaseModel):
    """Input for DoctorPredictionTool."""
    chief_complaints: str = Field(..., description="Comma-separated list of chief complaints.")
    pain_score: int = Field(..., description="Pain score from 0 to 10. (-1 if unknown)")
    age: int = Field(default=50, description="Patient age.")
    gender: str = Field(default="unknown", description="Patient gender ('male', 'female', 'unknown')")
    arrival_transport: str = Field(default="unknown", description="Arrival transport ('ambulance', 'walk_in', 'helicopter')")
    predicted_acuity: int = Field(..., description="Predicted ESI acuity (1-5) from previous triage step.")
    predicted_disposition: str = Field(..., description="Predicted disposition ('ADMITTED' or 'NOT ADMITTED') from triage.")

class DoctorPredictionTool(BaseTool):
    name: str = "doctor_prediction_tool"
    description: str = (
        "Predicts the broad Diagnosis Category for a patient, and if admitted, the Hospital Service Department. "
        "Requires outputs from the Triage Agent (predicted_acuity and predicted_disposition). "
        "Uses ClinicalBERT embeddings and XGBoost on MIMIC-IV."
    )
    args_schema: Type[BaseModel] = DoctorInput

    def _run(self, chief_complaints: str, pain_score: int, predicted_acuity: int,
             predicted_disposition: str, age: int = 50, gender: str = "unknown",
             arrival_transport: str = "unknown") -> str:
        
        diag_model, diag_le, srv_model, srv_le, bert_model = get_doctor_models()
        
        gender_male = 1 if gender.lower() in ("male", "m") else 0
        at = arrival_transport.lower().replace(" ", "_")
        arrival_ambulance = 1 if at == "ambulance" else 0
        arrival_helicopter = 1 if at == "helicopter" else 0
        arrival_walk_in = 1 if at in ("walk_in", "walkin", "walk") else 0
        
        # We construct structured features ONLY (no TF-IDF for Doctor Agent)
        # We will parse via build_features to guarantee identical demographic normalization
        df_temp = pd.DataFrame([{
            "chiefcomplaint": chief_complaints,
            "pain": pain_score,
            "age": age,
            "gender_male": gender_male,
            "arrival_ambulance": arrival_ambulance,
            "arrival_helicopter": arrival_helicopter,
            "arrival_walk_in": arrival_walk_in,
            "acuity": 3
        }])
        
        # Minimal extraction of purely structured demographics
        import io, contextlib
        # We need mock tfidf and severity map to pass into build_features to extract demographics correctly
        tfidf_mock = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        severity_mock = joblib.load(MODELS_DIR / "severity_map.joblib")
        
        with contextlib.redirect_stdout(io.StringIO()):
            X_raw_baseline, _, _ = build_features(df_temp, tfidf=tfidf_mock, severity_map=severity_mock, fit=False)
            
        # Strip TFIDF and retain structural cols
        structured_cols = [c for c in X_raw_baseline.columns if not c.startswith('tfidf_')]
        X_structured = X_raw_baseline[structured_cols].copy()
        
        # Append Triage outputs
        X_structured['predicted_acuity'] = predicted_acuity
        is_admitted = 1 if "ADMITTED" in predicted_disposition.upper() and "NOT" not in predicted_disposition.upper() else 0
        X_structured['predicted_admit'] = is_admitted
        
        # Generate ClinicalBERT Embeddings natively
        embeddings = bert_model.encode([chief_complaints], show_progress_bar=False)
        emb_df = pd.DataFrame(embeddings, columns=[f"bert_{i}" for i in range(embeddings.shape[1])])
        
        X_final = pd.concat([X_structured, emb_df], axis=1)
        
        # 1. Predict Diagnosis
        diag_probs = diag_model.predict_proba(X_final)[0]
        diag_pred_idx = int(np.argmax(diag_probs))
        diag_cat = str(diag_le.inverse_transform([diag_pred_idx])[0])
        diag_conf = float(diag_probs[diag_pred_idx])
        
        top3_idx = np.argsort(diag_probs)[::-1][:3]
        top3_diags = diag_le.inverse_transform(top3_idx)
        top3_probs = diag_probs[top3_idx]
        diag_breakdown = {str(d): f"{p:.1%}" for d, p in zip(top3_diags, top3_probs)}
        
        result = {
            "diagnosis_prediction": {
                "primary_category": diag_cat,
                "confidence": f"{diag_conf:.1%}",
                "top_3_categories": diag_breakdown
            }
        }
        
        # 2. Predict Service (only if admitted)
        if is_admitted:
            X_srv = X_final.copy()
            X_srv['predicted_diagnosis_enc'] = diag_pred_idx
            
            srv_probs = srv_model.predict_proba(X_srv)[0]
            srv_pred_idx = int(np.argmax(srv_probs))
            srv_pred = str(srv_le.inverse_transform([srv_pred_idx])[0])
            srv_conf = float(srv_probs[srv_pred_idx])
            
            top3_s_idx = np.argsort(srv_probs)[::-1][:3]
            top3_s_diags = srv_le.inverse_transform(top3_s_idx)
            top3_s_probs = srv_probs[top3_s_idx]
            srv_breakdown = {str(d): f"{p:.1%}" for d, p in zip(top3_s_diags, top3_s_probs)}
            
            result["service_prediction"] = {
                "hospital_department": srv_pred,
                "confidence": f"{srv_conf:.1%}",
                "top_alternative_departments": srv_breakdown
            }
        else:
            result["service_prediction"] = {
                "hospital_department": "N/A (Discharged)",
                "confidence": "100.0%"
            }
            
        return json.dumps(result, indent=2, ensure_ascii=False)
