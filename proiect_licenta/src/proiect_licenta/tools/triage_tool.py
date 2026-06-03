"""
Triage Prediction Tool v3 — CrewAI Tool

Uses TF-IDF + XGBoost models with demographics, vital signs, and PMH
(Past Medical History) features for triage predictions. Takes chief
complaints, pain score, age, gender, arrival transport, optional vitals,
and optional patient-reported prior history. Returns ESI acuity level
(1-5), admission/discharge, confidence scores.

Loads models from artifacts/triage/v3/ (triage pipeline v3 iter 2: v2 features
+ 19-column PMH block + longer training + ordinal-aware acuity weighting).

PMH inputs (`prior_history`, `n_prior_admissions`) are collected by the NLP
Parser during patient intake and passed through here. If the patient
skips them, the tool zero-fills with `no_history=1` — the exact pattern
used by ~39% of v3 training rows (first-time MIMIC patients), so the model
learned to handle missing PMH gracefully.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Type

import joblib
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths (canonical layout in proiect_licenta.paths)
# ---------------------------------------------------------------------------
from proiect_licenta.paths import TRIAGE_V3_DIR as MODELS_DIR


# ---------------------------------------------------------------------------
# Complaint text normalization — shared with training pipelines.
# Re-exported here so doctor_tool / doctor_tool_v2 can keep importing
# `normalize_complaint_text` from triage_tool.
# ---------------------------------------------------------------------------
from proiect_licenta.preprocessing import (  # noqa: F401
    ABBREVIATIONS,
    normalize_complaint_text,
)


# ---------------------------------------------------------------------------
# PMH (Past Medical History) — shared with training pipeline.
# Same imports and same vocabulary the doctor v3 tool uses, so inference-side
# parsing matches training-side feature construction.
# ---------------------------------------------------------------------------
from proiect_licenta.pmh_features import (
    PMH_FEATURE_COLS, PMH_NO_PRIOR_DAYS,
    parse_pmh_lookup as _parse_pmh_lookup,
    pmh_self_report_discrepancy,
)
from proiect_licenta.pmh_vocab import (
    PMH_CATEGORIES,
    flags_from_text as pmh_flags_from_text,
)


# Patient inputs that should be treated as "no history" / "skip" — matches
# the doctor_tool_v3.py vocabulary to keep the two tools' behavior consistent.
_PMH_SKIP_TOKENS = {
    "", "skip", "unknown", "no", "none", "n/a", "na", "-",
    "i don't know", "i dont know", "idk",
    "no significant", "no significant pmh", "no pmh",
}


# ---------------------------------------------------------------------------
# Vital sign constants (must match training/train_triage_v2.py)
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
# Pickle-resolution shim — neg_quadratic_kappa
# ---------------------------------------------------------------------------
# The v3 acuity model was trained with `eval_metric=neg_quadratic_kappa`
# as a custom callable (section 1.2). When training runs via
# `runpy.run_path(..., run_name='__main__')` (the Colab flow), pickle
# records the callable's qualified name as `__main__.neg_quadratic_kappa`.
# At load time pickle needs that name to exist on sys.modules['__main__'].
# At runtime, sys.modules['__main__'] is the host script (e.g.
# `run_crew.exe.__main__`), so the lookup fails with:
#
#   AttributeError: Can't get attribute 'neg_quadratic_kappa' on <module '__main__' ...>
#
# `save_models()` in train_triage_v3.py was fixed to strip eval_metric
# before joblib.dump (commit on 2026-05-29), but that fix only protects
# artifacts trained AFTER the fix landed. To keep the runtime resilient
# to pre-fix artifacts already on disk / on Drive, we inject a shim
# function into __main__ before any model load.
#
# The shim is never actually called at inference — eval_metric only
# matters during .fit(). It just needs to exist with the right name so
# pickle's by-name lookup succeeds.

def neg_quadratic_kappa(y_true, y_pred):
    """Inference-side shim of the training-time eval_metric callable.

    Same signature as `train_triage_v3.neg_quadratic_kappa`. Returns
    -κ (XGBoost convention: lower is better). Never invoked at inference
    — exists so `joblib.load` can resolve `__main__.neg_quadratic_kappa`
    on pre-fix v3 artifacts.
    """
    from sklearn.metrics import cohen_kappa_score
    y_pred_class = np.argmax(y_pred, axis=1)
    return -cohen_kappa_score(y_true, y_pred_class, weights="quadratic")


def _ensure_pickle_compat_in_main():
    """Inject `neg_quadratic_kappa` into sys.modules['__main__'] so that
    pre-fix v3 acuity artifacts (which pickled the callable with
    __module__='__main__') can be loaded via joblib.load without raising
    AttributeError.

    Idempotent. No-op if __main__ already has the attribute (e.g. when
    running under the training script itself).
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "neg_quadratic_kappa"):
        main_mod.neg_quadratic_kappa = neg_quadratic_kappa


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_models_cache = None


def get_models():
    global _models_cache
    if _models_cache is None:
        # Must run BEFORE joblib.load on the acuity model — see the
        # _ensure_pickle_compat_in_main docstring above.
        _ensure_pickle_compat_in_main()

        acuity_model = joblib.load(MODELS_DIR / "acuity_model.joblib")
        disposition_model = joblib.load(MODELS_DIR / "disposition_model.joblib")
        tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        severity_map = joblib.load(MODELS_DIR / "severity_map.joblib")
        vital_medians = joblib.load(MODELS_DIR / "vital_medians.joblib")

        # Defensive: also strip eval_metric on the loaded model in case
        # downstream code (e.g. doctor_tool_v2's cascade) pickles a copy.
        # No-op when set_params doesn't accept eval_metric on this estimator.
        try:
            acuity_model.set_params(eval_metric=None)
        except Exception:
            pass

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
    prior_history: str = Field(
        default="",
        description="Patient-reported chronic conditions / past medical history "
                    "as free text (e.g., 'CHF, diabetes, prior stroke', or 'CKD on "
                    "dialysis'). Use empty string if the patient skipped or has "
                    "no significant prior history. Parsed at inference against the "
                    "same pmh_vocab the training pipeline uses, so the 13 binary "
                    "diagnosis-group flags can fire even from short patient input."
    )
    n_prior_admissions: int = Field(
        default=-1,
        description="Approximate number of prior hospital admissions the patient "
                    "reports. Use -1 if not collected / unknown. 0 is a valid value "
                    "meaning 'never been admitted before'."
    )
    pmh_lookup_json: str = Field(
        default="",
        description="Optional JSON `pmh_block` from the patient_history_lookup_tool "
                    "(EHR lookup for a RETURNING patient with an MRN). When provided "
                    "for a known patient, it supplies the REAL prior-encounter record "
                    "— PMH flags PLUS the days-since-last-visit / same-complaint "
                    "numerics a patient can't report at the bedside — and OVERRIDES "
                    "the self-reported prior_history / n_prior_admissions. Copy the "
                    "lookup tool's `pmh_block` (or its full output) here verbatim. "
                    "Leave empty for first-time/unknown patients (subject_id -1) or "
                    "when the lookup returned known_patient=false; the tool then "
                    "falls back to the patient's self-report."
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
        "temperature (°F), heartrate (bpm), resprate, o2sat (%), sbp, dbp (mmHg), "
        "prior_history (free-text chronic conditions, '' if unknown), "
        "n_prior_admissions (int, -1 if unknown). "
        "Vital signs default to -1 (unknown) — provide them when available "
        "(e.g., ambulance/helicopter patients with EMS vitals). "
        "PMH inputs (prior_history, n_prior_admissions) default to skip-equivalents "
        "and zero-fill to first-time-patient pattern when the patient doesn't report. "
        "Uses XGBoost v3 models (acuity 67.55% / disposition 77.98% on the 83K MIMIC-IV "
        "test set; under-triage rate 15.56%, ESI 5 recall 26.82% — kept iteration 2 "
        "stack: PMH features + longer training + ordinal-aware ESI weighting)."
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
        prior_history: str = "",
        n_prior_admissions: int = -1,
        pmh_lookup_json: str = "",
    ) -> str:
        """Run triage prediction with optional vital signs and PMH inputs."""
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

        # 10b. PMH features (v3) — EHR lookup (preferred) OR ask-the-patient
        # fallback, mirroring doctor_tool_v3.py / doctor_disposition_tool.py so
        # triage and doctor share the same inference-time PMH logic. If the
        # patient gave an MRN and patient_history_lookup_tool found a returning
        # patient, its real prior-encounter block — including the
        # days_since_last_* / same_complaint_as_prior numerics a patient can't
        # report at the bedside — OVERRIDES the self-report. Otherwise we parse
        # the free-text history and zero-fill the unrecoverable numerics to the
        # first-time-patient sentinels (the no_history=1 pattern the model saw
        # on ~39% of training rows).
        lookup_block = _parse_pmh_lookup(pmh_lookup_json)
        used_history_lookup = lookup_block is not None
        if used_history_lookup:
            pmh_data = {col: lookup_block[col] for col in PMH_FEATURE_COLS}
            pmh_flags_active = {
                c for c in PMH_CATEGORIES if pmh_data.get(f"pmh_{c}") == 1
            }
            pmh_no_history = int(pmh_data["no_history"])
            n_prior_adm_val = int(pmh_data["n_prior_admissions"])
        else:
            pmh_text = (prior_history or "").strip()
            pmh_unknown = pmh_text == "" or pmh_text.lower() in _PMH_SKIP_TOKENS
            if pmh_unknown:
                pmh_flags_active: set = set()
                pmh_no_history = 1
            else:
                pmh_flags_active = pmh_flags_from_text(pmh_text)
                pmh_no_history = 0

            # n_prior_admissions: if not collected (-1) → 0 + no_history=1.
            # days_since_last_* are not collectable from the patient at the
            # bedside, so we always zero-fill (PMH_NO_PRIOR_DAYS sentinel) — the
            # model saw the same pattern on first-time-patient training rows.
            if n_prior_admissions is None or n_prior_admissions < 0:
                n_prior_adm_val = 0
                n_prior_ed_val = 0
                pmh_no_history = 1 if not pmh_flags_active else pmh_no_history
            else:
                n_prior_adm_val = int(n_prior_admissions)
                n_prior_ed_val = 0  # not collected separately at inference

            # If the patient reported conditions but skipped the count, still
            # treat them as having history.
            if pmh_flags_active and pmh_no_history == 1:
                pmh_no_history = 0

            pmh_data = {f"pmh_{c}": (1 if c in pmh_flags_active else 0)
                        for c in PMH_CATEGORIES}
            pmh_data["n_prior_admissions"] = n_prior_adm_val
            pmh_data["n_prior_ed_visits"] = n_prior_ed_val
            pmh_data["days_since_last_admission"] = float(PMH_NO_PRIOR_DAYS)
            pmh_data["days_since_last_ed"] = float(PMH_NO_PRIOR_DAYS)
            pmh_data["same_complaint_as_prior"] = 0.0
            pmh_data["no_history"] = pmh_no_history

        # Read-only reconciliation: conditions the patient self-reported that
        # the real record didn't have (the record still drives the feature
        # vector). Empty unless the EHR lookup overrode a non-empty self-report.
        self_report_not_in_record = (
            pmh_self_report_discrepancy(prior_history, lookup_block)
            if used_history_lookup else []
        )

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

        # v3 — PMH block. Built from the patient-reported prior_history /
        # n_prior_admissions; column order = PMH_FEATURE_COLS so it matches
        # the position used by train_triage_v3.build_features (right after
        # v2 vital cols, before TF-IDF).
        pmh_df = pd.DataFrame({col: [pmh_data[col]] for col in PMH_FEATURE_COLS})

        features = pd.concat([structured, pmh_df, tfidf_df], axis=1)

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

        # Summarize what PMH info was actually applied at inference, so the
        # caller (and the doctor downstream) can see which flags fired and
        # whether the patient was treated as first-time.
        prior_history_summary = {
            "raw_input": prior_history if prior_history else None,
            "n_prior_admissions_used": n_prior_adm_val,
            "no_history_flag": pmh_no_history,
            "pmh_categories_fired": sorted(pmh_flags_active),
            "pmh_category_count": len(pmh_flags_active),
            # EHR lookup: True when the PMH block came from the real prior
            # record (patient_history_lookup_tool) rather than self-report —
            # including the days-since / same-complaint numerics below.
            "used_history_lookup": bool(used_history_lookup),
            "history_lookup_numerics": ({
                "n_prior_admissions": int(pmh_data["n_prior_admissions"]),
                "n_prior_ed_visits": int(pmh_data["n_prior_ed_visits"]),
                "days_since_last_admission": float(pmh_data["days_since_last_admission"]),
                "days_since_last_ed": float(pmh_data["days_since_last_ed"]),
                "same_complaint_as_prior": float(pmh_data["same_complaint_as_prior"]),
            } if used_history_lookup else None),
            # Read-only: self-reported conditions absent from the prior record
            # (the record still drives the prediction). Empty unless the lookup
            # overrode a non-empty self-report.
            "self_report_not_in_record": self_report_not_in_record,
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
            "vital_signs": {
                "provided": vitals_provided,
                "imputed_as_missing": vitals_imputed,
                "abnormalities_detected": abnormalities_detected,
                "abnormal_vital_count": abnormal_vital_count,
            },
            "prior_history_used": prior_history_summary,
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
