"""Doctor disposition tool: calibrated binary admit/discharge.

Wraps the calibrated model from train_doctor_disposition.py
(artifacts/doctor/v3/disposition_model.joblib, with a _raw sibling for audit).
The cascade source is triage v3, unlike the v3 diagnosis/department models which
use the triage v1 cascade. At inference it builds the 2069-col v3 input vector,
appends the 6 soft-cascade columns (triage v3 acuity softmax + admit probability),
the longitudinal-vital fallback, and the 11 medication columns to reach the
2128-col training layout, then calls predict_proba (the artifact is a calibrated
isotonic model) and thresholds at DECISION_THRESHOLD to decide. Output includes
the probabilities, the decision, the triage v3 baseline it refined, and a
nurse_data_used block.
"""

import json
import sys
from pathlib import Path
from typing import Type

import joblib
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score


# Pickle-compat shim - neg_quadratic_kappa
# The triage v3 acuity model was pickled with a custom `eval_metric` callable
# whose __module__ is '__main__' (because it was trained via runpy in the
# Colab notebook). Without this shim, joblib.load fails at unpickle time when
# this tool is imported from the crew runtime.  Same fix as triage_tool.py.
def neg_quadratic_kappa(y_true, y_pred):
    """Inference-side shim of the training-time eval_metric callable."""
    y_pred_class = np.argmax(y_pred, axis=1)
    return -cohen_kappa_score(y_true, y_pred_class, weights="quadratic")


def _ensure_pickle_compat_in_main():
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "neg_quadratic_kappa"):
        main_mod.neg_quadratic_kappa = neg_quadratic_kappa


# Paths + shared helpers (triage v3 cascade + v3-nurse longitudinal layout)
from proiect_licenta.paths import (
    TRIAGE_V3_DIR,
    DOCTOR_V3_DIR,
)
from proiect_licenta.tools.triage_tool import normalize_complaint_text
from proiect_licenta.tools.med_vocab import (
    MED_CATEGORIES, MED_FEATURE_COLS, flags_from_name, flags_from_text,
    parse_med_lookup, med_self_report_discrepancy,
)
from proiect_licenta.training.train_nurse_v3 import (
    _normalize_rhythm,
    _RHYTHM_BUCKETS,
    _VITAL_COLS_LONG,
    LONG_VITAL_FEATURE_COLS,
)
from proiect_licenta.vital_trajectory import build_longitudinal_block
from proiect_licenta.tools.vital_trajectory_io import (
    parse_vital_trajectory, parse_rhythm_readings,
)
from proiect_licenta.training import train_triage_v3 as _triage_v3
from proiect_licenta.training.train_triage_v3 import (
    build_features as _build_v3_features,
    VITAL_COLS as _V3_VITAL_COLS,
)
from proiect_licenta.pmh_vocab import (
    PMH_CATEGORIES, flags_from_text as pmh_flags_from_text,
)
from proiect_licenta.pmh_features import (
    PMH_FEATURE_COLS, PMH_NO_PRIOR_DAYS,
    parse_pmh_lookup as _parse_pmh_lookup,
    pmh_self_report_discrepancy,
)


# Operating point: the cutoff where calibrated P(admit) becomes an "admit"
# decision. Changing it only slides along the fixed ROC curve (no retraining).
# Tuned to 0.40 from 0.50 via sweep_disposition_threshold.py on the 83,617-row
# test split: 0.40 maximizes F1 and Youden's J and cuts under-triage from 23.5%
# to 18.1% for a 0.6pp accuracy cost. Accuracy favors the discharge majority, so
# for a clinical disposition under-triage is the costlier error and F1/Youden are
# the right selectors. Going below 0.40 should be backed by an explicit cost
# model, not the ED over-triage literature (which is about acuity triage, not
# admit/discharge where over-triage means an unnecessary admission).
DECISION_THRESHOLD: float = 0.40


# Vital sign medians (population averages from training, used when the patient
# can't supply a value). Same as v3-nurse tool to keep semantics aligned.
VITAL_MEDIANS = {
    "temperature": 98.1, "heartrate": 84, "resprate": 18,
    "o2sat": 98, "sbp": 134, "dbp": 78,
}


def _classify_medications(medications_raw: str) -> dict:
    """Same medication classification used by the v3-nurse tool, duplicated
    here so this tool has no runtime dependency on the nurse tool module."""
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


# Lazy model loading (per-process cache)
_dispo_cache = None


def get_disposition_models():
    global _dispo_cache
    if _dispo_cache is None:
        _ensure_pickle_compat_in_main()  # MUST come before joblib.load

        # Triage v3 cascade artifacts
        tfidf = joblib.load(TRIAGE_V3_DIR / "tfidf_vectorizer.joblib")
        severity_map = joblib.load(TRIAGE_V3_DIR / "severity_map.joblib")
        vital_medians_v3 = joblib.load(TRIAGE_V3_DIR / "vital_medians.joblib")
        acuity_model = joblib.load(TRIAGE_V3_DIR / "acuity_model.joblib")
        disposition_model_triage = joblib.load(
            TRIAGE_V3_DIR / "disposition_model.joblib",
        )
        # Strip the custom eval_metric callable so subsequent loads / saves
        # don't keep dragging in the __main__ shim.
        try:
            acuity_model.set_params(eval_metric=None)
        except Exception:
            pass

        # Doctor disposition v3 - deployment + raw audit copies
        disposition_model = joblib.load(DOCTOR_V3_DIR / "disposition_model.joblib")

        with open(DOCTOR_V3_DIR / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        _dispo_cache = {
            "tfidf": tfidf,
            "severity_map": severity_map,
            "vital_medians_v3": vital_medians_v3,
            "acuity_model": acuity_model,
            "triage_dispo_model": disposition_model_triage,
            "dispo_model": disposition_model,
            "metadata": metadata,
        }
    return _dispo_cache


# Tool Input Schema
class DoctorDispositionInput(BaseModel):
    """Input schema for the Doctor Disposition Tool.

    Same fields as the v3-nurse reassessment tool - admit/discharge
    refinement uses the same signals (triage + vitals + meds + rhythm + PMH).
    """
    chief_complaints: str = Field(
        ...,
        description="Comma-separated chief complaints, e.g. 'chest pain, dyspnea'.",
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
        ...,
        description="ESI acuity level (1-5) from the triage agent - reported in the output but the tool re-runs triage v3 internally for the soft cascade.",
    )
    triage_is_admitted: bool = Field(
        ...,
        description="Whether the TRIAGE agent predicted admit. The doctor disposition tool may override this verdict.",
    )
    temperature: float = Field(default=-1.0, description="Temperature in Fahrenheit, or -1.0 if unknown.")
    heartrate: float = Field(default=-1.0, description="Heart rate in bpm, or -1.0 if unknown.")
    resprate: float = Field(default=-1.0, description="Respiratory rate, or -1.0 if unknown.")
    o2sat: float = Field(default=-1.0, description="Oxygen saturation %, or -1.0 if unknown.")
    sbp: float = Field(default=-1.0, description="Systolic BP, or -1.0 if unknown.")
    dbp: float = Field(default=-1.0, description="Diastolic BP, or -1.0 if unknown.")
    rhythm: str = Field(
        default="",
        description="Cardiac rhythm string from the nurse (e.g. 'sinus', 'atrial fibrillation', 'paced'). Empty string if not collected.",
    )
    medications_raw: str = Field(
        default="unknown",
        description="Free-text medication list from the nurse (or 'unknown'/'none').",
    )
    prior_history: str = Field(
        default="",
        description="Patient's free-text chronic conditions / past medical history. Empty / skip-equivalent -> no_history pattern.",
    )
    n_prior_admissions: int = Field(
        default=-1,
        description="Patient-reported approximate number of prior hospital admissions, or -1 if not collected.",
    )
    vital_trajectory_json: str = Field(
        default="",
        description=(
            "Optional JSON string of MULTIPLE chronological vital readings "
            "collected during the stay, e.g. "
            '{\"heartrate\": [88, 105, 112], \"o2sat\": [98, 95, 93]}. '
            "Copy the nurse tool's `vital_trajectory` block here verbatim. "
            "When provided, the model uses real vital trends (min/max/last/"
            "delta, abnormal-reading counts) instead of a single-snapshot "
            "fallback - this materially improves the admit/discharge "
            "prediction. Empty string if only one reading is available."
        ),
    )
    pmh_lookup_json: str = Field(
        default="",
        description=(
            "Optional JSON `pmh_block` returned by patient_history_lookup_tool "
            "for a RETURNING patient (paste it verbatim). When provided, the "
            "real prior-encounter record (PMH flags AND the days-since-last-"
            "admission / same-complaint-as-prior numerics a patient can't report "
            "at the bedside) OVERRIDES the self-reported prior_history / "
            "n_prior_admissions fields. Empty string for first-time / unknown "
            "patients - then the self-report fields are used."
        ),
    )
    med_lookup_json: str = Field(
        default="",
        description=(
            "Optional JSON `med_block` returned by patient_history_lookup_tool "
            "for a RETURNING patient (paste it verbatim). When provided, the "
            "patient's reconciled home-med list from their most recent prior visit "
            "OVERRIDES the self-reported medications_raw. Empty string for "
            "first-time / unknown patients or when no prior med record exists - "
            "then the self-reported medications are parsed instead."
        ),
    )


# CrewAI Tool
class DoctorDispositionTool(BaseTool):
    name: str = "doctor_disposition_tool"
    description: str = (
        "Refines the triage admit/discharge prediction using ALL data collected "
        "so far: triage features (via the soft cascade - full 5-class acuity "
        "softmax + triage disposition probability), snapshot vitals, longitudinal "
        "vital fallback, cardiac rhythm, medications, and past medical history. "
        "The underlying model is a calibrated binary classifier trained on the "
        "FULL 425K ED stays (admit + discharge) - see "
        "docs/agents/doctor-agent.md#doctor-disposition-v3 for the per-subgroup "
        "lift and ECE 0.0036 calibration story. Returns calibrated admit/discharge "
        "probabilities and the binary decision at threshold 0.40 (tuned from 0.50). ALWAYS call this "
        "tool between the nurse step and the v3 diagnosis/department reassessment; "
        "the reassessment task should gate on THIS tool's `is_admitted` flag, not "
        "the triage one."
    )
    args_schema: Type[BaseModel] = DoctorDispositionInput

    def _run(
        self,
        chief_complaints: str,
        pain_score: int,
        predicted_acuity: int,
        triage_is_admitted: bool,
        age: int = 50,
        gender: str = "unknown",
        arrival_transport: str = "unknown",
        temperature: float = -1.0,
        heartrate: float = -1.0,
        resprate: float = -1.0,
        o2sat: float = -1.0,
        sbp: float = -1.0,
        dbp: float = -1.0,
        rhythm: str = "",
        medications_raw: str = "unknown",
        prior_history: str = "",
        n_prior_admissions: int = -1,
        vital_trajectory_json: str = "",
        pmh_lookup_json: str = "",
        med_lookup_json: str = "",
    ) -> str:
        models = get_disposition_models()
        tfidf = models["tfidf"]
        severity_map = models["severity_map"]
        vital_medians_v3 = models["vital_medians_v3"]
        acuity_model = models["acuity_model"]
        triage_dispo_model = models["triage_dispo_model"]
        dispo_model = models["dispo_model"]

        # 1. Demographics + arrival transport flags
        age_val = max(0, min(120, age))
        gender_male = 1 if gender.lower() in ("male", "m") else 0
        at = arrival_transport.lower().replace(" ", "_")
        arrival_ambulance = 1 if at == "ambulance" else 0
        arrival_helicopter = 1 if at == "helicopter" else 0
        arrival_walk_in = 1 if at in ("walk_in", "walkin", "walk") else 0
        is_walkin_mode = not (arrival_ambulance or arrival_helicopter)

        # 2. Pain
        pain_val = max(-1, min(10, pain_score))
        pain_missing = 1 if pain_val < 0 else 0

        # ── 3. Vital values - keep two copies:
        #    (a) `vital_values` - imputed for the disposition model's own use
        #        + downstream longitudinal fallback
        #    (b) `vital_cascade` - NaN if missing OR if walk-in (so v3's
        #        `_missing` flags compute correctly inside build_features)
        vitals_raw_in = {
            "temperature": temperature, "heartrate": heartrate,
            "resprate": resprate, "o2sat": o2sat,
            "sbp": sbp, "dbp": dbp,
        }
        vital_values = {}
        vital_cascade = {}
        for vname, vval in vitals_raw_in.items():
            missing = (vval is None) or (vval == -1.0) or (
                isinstance(vval, float) and np.isnan(vval)
            )
            if missing:
                vital_values[vname] = VITAL_MEDIANS[vname]
                vital_cascade[vname] = np.nan
            else:
                vital_values[vname] = float(vval)
                vital_cascade[vname] = (
                    np.nan if is_walkin_mode else float(vval)
                )

        # Clip imputed values to plausible ranges (must match _clean_vitals)
        vital_values["temperature"] = float(np.clip(vital_values["temperature"], 90, 110))
        vital_values["heartrate"] = float(np.clip(vital_values["heartrate"], 20, 250))
        vital_values["resprate"] = float(np.clip(vital_values["resprate"], 4, 60))
        vital_values["o2sat"] = float(np.clip(vital_values["o2sat"], 50, 100))
        vital_values["sbp"] = float(np.clip(vital_values["sbp"], 50, 300))
        vital_values["dbp"] = float(np.clip(vital_values["dbp"], 20, 200))

        # Same nurse-style clinical flags used by _fill_longitudinal_vitals
        fever = 1 if vital_values["temperature"] > 100.4 else 0
        tachycardia = 1 if vital_values["heartrate"] > 100 else 0
        bradycardia = 1 if vital_values["heartrate"] < 60 else 0
        tachypnea = 1 if vital_values["resprate"] > 20 else 0
        hypoxia = 1 if vital_values["o2sat"] < 94 else 0
        hypertension = 1 if vital_values["sbp"] > 140 else 0
        hypotension = 1 if vital_values["sbp"] < 90 else 0

        # 4. PMH - EHR lookup (preferred) OR ask-the-patient fallback
        # If patient_history_lookup_tool found a returning patient, its real
        # prior-encounter block (PMH flags + the days-since / same-complaint
        # numerics the bedside interview can't recover) overrides the
        # self-report. Otherwise parse the patient-reported prior_history text
        # and zero-fill the unrecoverable numerics to the first-time sentinels.
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
            pmh_unknown = pmh_text == "" or pmh_text.lower() in (
                "skip", "unknown", "no", "none", "n/a", "na", "-",
                "i don't know", "i dont know", "idk",
                "no significant", "no significant pmh", "no pmh",
            )
            if pmh_unknown:
                pmh_flags_active: set = set()
                pmh_no_history = 1
            else:
                pmh_flags_active = pmh_flags_from_text(pmh_text)
                pmh_no_history = 0

            if n_prior_admissions is None or n_prior_admissions < 0:
                n_prior_adm_val = 0
                n_prior_ed_val = 0
                pmh_no_history = 1 if not pmh_flags_active else pmh_no_history
            else:
                n_prior_adm_val = int(n_prior_admissions)
                n_prior_ed_val = 0
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

        # Read-only reconciliation: conditions the patient mentioned that the
        # real record didn't have (the record still wins for the feature vector;
        # this is only surfaced to the clinician). Empty unless the EHR lookup
        # was used AND the patient also gave free-text history.
        self_report_not_in_record = (
            pmh_self_report_discrepancy(prior_history, lookup_block)
            if used_history_lookup else []
        )

        # 5. Single-row df for the cascade input (v3 build_features)
        df_cascade = pd.DataFrame({
            "chiefcomplaint": [chief_complaints],
            "age": [age_val],
            "pain": [pain_val],
            "pain_missing": [pain_missing],
            "gender_male": [gender_male],
            "arrival_ambulance": [arrival_ambulance],
            "arrival_helicopter": [arrival_helicopter],
            "arrival_walk_in": [arrival_walk_in],
            "acuity": [predicted_acuity if 1 <= predicted_acuity <= 5 else 3],
            # vital_cascade carries NaN for walk-ins / missing
            "temperature": [vital_cascade["temperature"]],
            "heartrate": [vital_cascade["heartrate"]],
            "resprate": [vital_cascade["resprate"]],
            "o2sat": [vital_cascade["o2sat"]],
            "sbp": [vital_cascade["sbp"]],
            "dbp": [vital_cascade["dbp"]],
            **{col: [pmh_data[col]] for col in PMH_FEATURE_COLS},
        })

        triage_features, _, _, _ = _build_v3_features(
            df_cascade, tfidf=tfidf, severity_map=severity_map,
            vital_medians=vital_medians_v3, fit=False,
        )
        triage_features = triage_features.reset_index(drop=True)

        # 6. Soft cascade - full triage v3 acuity softmax + dispo prob
        acuity_proba = acuity_model.predict_proba(triage_features)[0]
        predicted_acuity_int = int(np.argmax(acuity_proba)) + 1
        triage_features_disp = triage_features.copy()
        triage_features_disp["predicted_acuity"] = predicted_acuity_int
        triage_dispo_proba_admit = float(
            triage_dispo_model.predict_proba(triage_features_disp)[0, 1]
        )

        for k in range(5):
            triage_features[f"triage_acuity_proba_{k + 1}"] = float(acuity_proba[k])
        triage_features["triage_disposition_proba_admit"] = triage_dispo_proba_admit

        # 7. Medications - EHR lookup (preferred) OR ask-the-patient
        # If patient_history_lookup_tool returned a med_block (returning patient
        # with a prior reconciled home-med list), it OVERRIDES the self-reported
        # medications; otherwise parse the patient-reported med text. Mirrors the
        # pmh_lookup_json override above.
        med_lookup_block = parse_med_lookup(med_lookup_json)
        used_med_lookup = med_lookup_block is not None
        if used_med_lookup:
            med_info = med_lookup_block
        else:
            med_info = _classify_medications(medications_raw)
        self_report_meds_not_in_record = (
            med_self_report_discrepancy(medications_raw, med_lookup_block)
            if used_med_lookup else []
        )
        meds_df = pd.DataFrame({col: [med_info[col]] for col in MED_FEATURE_COLS})

        # 8. Longitudinal vital block
        # If the nurse supplied a multi-reading trajectory, build REAL
        # min/max/last/delta + abnormal-reading counts from it (and set
        # has_longitudinal_vitals=1). Otherwise fall back to the single
        # snapshot (min==max==last, delta=0, has_longitudinal_vitals=0) - the
        # historical behavior. Snapshot-only systematically under-states the
        # admit probability vs the training distribution, so the trajectory
        # path matters for disposition accuracy.
        trajectory = parse_vital_trajectory(vital_trajectory_json)
        rhythm_readings = parse_rhythm_readings(vital_trajectory_json)
        long_data = build_longitudinal_block(
            snapshot=vital_values, readings=trajectory, rhythm=rhythm,
            rhythm_readings=rhythm_readings,
        )
        # Display bucket: primary reported rhythm, else the first trajectory
        # reading (the rhythm_irregular flag below comes from the aggregate).
        _display_rhythm = rhythm or (rhythm_readings[0] if rhythm_readings else "")
        bucket = _normalize_rhythm(_display_rhythm) if _display_rhythm else ""
        long_df = pd.DataFrame({col: [long_data[col]] for col in LONG_VITAL_FEATURE_COLS})
        used_trajectory = int(long_data["has_longitudinal_vitals"]) == 1

        # 9. Assemble final feature matrix (matches train_doctor_disposition)
        features = pd.concat([triage_features, meds_df, long_df], axis=1)

        # 10. Predict (calibrated)
        proba = dispo_model.predict_proba(features)[0]
        p_discharge = float(proba[0])
        p_admit = float(proba[1])
        is_admitted = p_admit >= DECISION_THRESHOLD
        confidence = max(p_admit, p_discharge)

        # Comparison with triage
        triage_admit_at_threshold = triage_dispo_proba_admit >= DECISION_THRESHOLD
        flipped = (triage_admit_at_threshold != is_admitted)

        # Concise reasoning bullets from top-signal features
        # Bands are anchored to DECISION_THRESHOLD so the prose always agrees
        # with the binary decision (a case just above the threshold is described
        # as a borderline admit, not a "modest discharge").
        reasoning = []
        _margin = p_admit - DECISION_THRESHOLD
        if p_admit > 0.85:
            reasoning.append("Very high admit probability.")
        elif p_admit > 0.70:
            reasoning.append("High admit probability.")
        elif _margin >= 0 and _margin <= 0.10:
            reasoning.append(
                f"Borderline admit - just above the {DECISION_THRESHOLD:.2f} threshold."
            )
        elif _margin > 0.10:
            reasoning.append("Modest admit probability.")
        elif _margin < 0 and _margin >= -0.10:
            reasoning.append(
                f"Borderline discharge - just below the {DECISION_THRESHOLD:.2f} threshold."
            )
        else:
            reasoning.append("High discharge probability.")
        if age_val >= 65:
            reasoning.append(
                "Elderly patient (age >= 65) - model gains substantial signal in this subgroup."
            )
        if n_prior_adm_val >= 1:
            reasoning.append(
                f"Patient reports {n_prior_adm_val} prior hospital admission(s) - strong prior for repeat admission."
            )
        if any((fever, tachycardia, hypoxia, hypotension, hypertension)):
            abn = [name for name, flag in [
                ("fever", fever), ("tachycardia", tachycardia),
                ("hypoxia", hypoxia), ("hypotension", hypotension),
                ("hypertension", hypertension),
            ] if flag]
            reasoning.append(
                f"Abnormal vitals detected: {', '.join(abn)}."
            )
        if pmh_flags_active:
            reasoning.append(
                f"PMH categories recognized: {', '.join(sorted(pmh_flags_active))}."
            )
        if bucket and bucket != "sinus":
            reasoning.append(
                f"Non-sinus cardiac rhythm reported ('{bucket}') - irregular-rhythm flag set."
            )

        # Build result
        result = {
            "patient_summary": {
                "chief_complaints": [c.strip() for c in chief_complaints.split(",") if c.strip()],
                "age": age_val,
                "gender": gender,
                "pain_score": pain_val,
                "arrival_transport": arrival_transport,
                "triage_acuity_input": predicted_acuity,
                "triage_admit_input": bool(triage_is_admitted),
            },
            "disposition_prediction": {
                "is_admitted": bool(is_admitted),
                "decision": "ADMIT" if is_admitted else "DISCHARGE",
                "p_admit": round(p_admit, 4),
                "p_discharge": round(p_discharge, 4),
                "confidence": f"{confidence:.1%}",
                "decision_threshold": DECISION_THRESHOLD,
                "calibration_note": (
                    "Probabilities are isotonic-calibrated (ECE 0.0036 on the "
                    "held-out test set, see "
                    "docs/agents/doctor-agent.md#doctor-disposition-v3). A "
                    "reading of P(admit)=0.82 means ~82% of patients with "
                    "that score were historically admitted."
                ),
            },
            "triage_v3_baseline": {
                "triage_admit_input": bool(triage_is_admitted),
                "triage_dispo_proba_admit": round(triage_dispo_proba_admit, 4),
                "triage_argmax_acuity": predicted_acuity_int,
                "triage_acuity_softmax": [
                    round(float(p), 4) for p in acuity_proba
                ],
                "doctor_overrode_triage": bool(flipped),
                "override_direction": (
                    None if not flipped
                    else ("admit -> discharge" if not is_admitted
                          else "discharge -> admit")
                ),
            },
            "reasoning": reasoning,
            "nurse_data_used": {
                "vitals_provided": {
                    k: v for k, v in vitals_raw_in.items()
                    if v is not None and v != -1.0
                },
                "used_vital_trajectory": used_trajectory,
                "rhythm_raw": rhythm if rhythm else None,
                "rhythm_bucket": bucket if bucket else None,
                "rhythm_irregular": int(long_data["rhythm_irregular"]),
                "medications_raw": medications_raw if medications_raw and medications_raw.lower() not in (
                    "unknown", "none", "no", "skip",
                ) else None,
                "n_medications": int(med_info["n_medications"]),
                "med_categories_flagged": [
                    cat for cat in MED_CATEGORIES if med_info.get(cat, 0) == 1
                ],
                # EHR med lookup: when True, the med flags came from the patient's
                # prior reconciled home-med list rather than self-report.
                "used_med_lookup": bool(used_med_lookup),
                "self_report_meds_not_in_record": self_report_meds_not_in_record,
                "prior_history_raw": prior_history if prior_history else None,
                "pmh_categories_detected": sorted(pmh_flags_active) if pmh_flags_active else [],
                "n_prior_admissions_reported": (
                    n_prior_adm_val if n_prior_adm_val >= 0
                    and n_prior_admissions is not None and n_prior_admissions >= 0
                    else None
                ),
                "no_history_fallback": bool(pmh_no_history),
                # EHR lookup: when True, the PMH block came from the real prior
                # record (patient_history_lookup_tool) rather than self-report -
                # including the days-since / same-complaint numerics below.
                "used_history_lookup": bool(used_history_lookup),
                "history_lookup_numerics": ({
                    "n_prior_admissions": int(pmh_data["n_prior_admissions"]),
                    "n_prior_ed_visits": int(pmh_data["n_prior_ed_visits"]),
                    "days_since_last_admission": float(pmh_data["days_since_last_admission"]),
                    "days_since_last_ed": float(pmh_data["days_since_last_ed"]),
                    "same_complaint_as_prior": float(pmh_data["same_complaint_as_prior"]),
                } if used_history_lookup else None),
                # Read-only: self-reported conditions absent from the prior
                # record (the record still drives the prediction). Possible
                # outside-system or newly-identified history worth a clinician's
                # attention. Empty unless the lookup overrode a non-empty report.
                "self_report_not_in_record": self_report_not_in_record,
            },
            "model_version": "doctor_disposition_v3 (Option B, plan section 3)",
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
