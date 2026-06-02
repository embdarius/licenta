"""
Patient History Lookup Tool — CrewAI Tool (EHR simulation)
==========================================================

Simulates an EHR record lookup for a *returning* patient. Keyed on a
``subject_id`` / simulated MRN (deliberately NOT fuzzy name/age matching —
record linkage is its own error-prone subsystem and out of scope), it returns
the real 19-column ``PMH_FEATURE_COLS`` block derived from the patient's prior
MIMIC encounters, with the strict ``prior_*time < current_intime`` leakage
filter applied at query time.

Motivation
----------
The runtime handles patient history one way today: the NLP parser / nurse
*asks* the patient (``prior_history`` free text, ``n_prior_admissions``). That
recovers the coarse ``pmh_<group>`` flags but NOT the fine recent-visit
numerics — ``days_since_last_admission``, ``days_since_last_ed``,
``n_prior_ed_visits``, ``same_complaint_as_prior`` — which a patient cannot
report at the bedside. The disposition + v3 doctor tools therefore zero-fill
those to the first-time-patient sentinels (``days_since = 9999``,
``same_complaint = 0``). A per-case diagnostic on the Phase-4 benchmark showed
the *entire* residual runtime↔feature-vector-gated gap was one returning
patient whose admit decision flipped on exactly these fields. This tool is the
complementary half: for a patient already in the system, *fetch* the prior data
instead of asking for it.

Known vs unknown patients split exactly the way the models were trained:
  * known patient (prior encounters before ``current_intime``) -> real PMH block;
  * first-time / unknown subject_id -> ``known_patient=false`` and the runtime
    keeps its existing ask-the-patient zero-fill path (the all-zero
    ``no_history=1`` pattern the models saw on ~39% of training rows).

The doctor tools consume the returned ``pmh_block`` via their
``pmh_lookup_json`` argument (mirroring the ``vital_trajectory_json`` pattern):
when present it overrides the text-derived PMH block, when absent they fall
back to ask-the-patient. Fully backward-compatible.

Leakage discipline
------------------
The persisted index contains ALL of a subject's encounters; the
``< current_intime`` filter is applied per query inside
``assemble_pmh_for_stay`` (the same function training uses), and that function
asserts no surviving prior encounter is dated at/after ``current_intime``. The
current encounter's own discharge summary can never leak because its
admit/intime is not strictly before ``current_intime``.
"""

import json
from typing import Type

import joblib
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from proiect_licenta.paths import HISTORY_INDEX_PKL
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.pmh_features import (
    PMH_FEATURE_COLS, assemble_pmh_for_stay,
)


# ---------------------------------------------------------------------------
# Lazy index loading (per-process cache, same pattern as the doctor tools)
# ---------------------------------------------------------------------------
_index_cache = None


def get_history_index():
    """Load and cache the persisted patient-history index. Returns the inner
    index dict (the four per-subject structures) or None if it hasn't been
    built yet (so the tool degrades gracefully to 'unknown patient')."""
    global _index_cache
    if _index_cache is None:
        if not HISTORY_INDEX_PKL.exists():
            _index_cache = {"index": None}
        else:
            payload = joblib.load(HISTORY_INDEX_PKL)
            _index_cache = payload if isinstance(payload, dict) and "index" in payload \
                else {"index": payload}
    return _index_cache.get("index")


def _subject_known(index: dict, subject_id: int) -> bool:
    """True if the index has ANY encounter for this subject (the intime filter
    still decides whether there's PRIOR history)."""
    return (
        subject_id in index.get("adm_by_subject", {})
        or subject_id in index.get("ed_by_subject", {})
    )


# ---------------------------------------------------------------------------
# Tool Input Schema
# ---------------------------------------------------------------------------
class PatientHistoryLookupInput(BaseModel):
    """Input schema for the Patient History Lookup Tool."""
    subject_id: int = Field(
        ...,
        description=(
            "The patient's MRN / subject_id. Use -1 (or 0) for a first-time or "
            "anonymous patient with no record in the system — the tool then "
            "returns known_patient=false and the pipeline keeps asking the "
            "patient about their history."
        ),
    )
    current_intime: str = Field(
        ...,
        description=(
            "The CURRENT ED arrival timestamp (ISO 8601, e.g. "
            "'2150-03-14 18:22:00'). Only encounters strictly BEFORE this time "
            "are used — this is the leakage guard."
        ),
    )
    chief_complaints: str = Field(
        default="",
        description=(
            "Comma-separated chief complaints for the CURRENT visit. Used only "
            "to compute same_complaint_as_prior (Jaccard vs the patient's last "
            "ED complaint). Empty is fine."
        ),
    )


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------
class PatientHistoryLookupTool(BaseTool):
    name: str = "patient_history_lookup_tool"
    description: str = (
        "Looks up a RETURNING patient's prior medical history from the hospital "
        "record system, keyed on their MRN / subject_id. Returns the real past "
        "medical history flags AND the prior-encounter numerics (days since last "
        "admission / ED visit, number of prior ED visits, whether the current "
        "complaint matches the last visit) that a patient cannot reliably report "
        "at the bedside. Call this right after intake if the patient provides an "
        "MRN; copy the returned `pmh_block` JSON into the doctor disposition and "
        "v3 reassessment tools' `pmh_lookup_json` argument so they use the real "
        "record instead of the patient's self-report. For a first-time / unknown "
        "patient (subject_id -1) it returns known_patient=false and the pipeline "
        "falls back to asking the patient. All lookups are leakage-safe: only "
        "encounters strictly before the current arrival time are used."
    )
    args_schema: Type[BaseModel] = PatientHistoryLookupInput

    def _run(
        self,
        subject_id: int,
        current_intime: str,
        chief_complaints: str = "",
    ) -> str:
        index = get_history_index()

        # A missing / unparseable arrival time means the leakage filter can't be
        # applied safely -> refuse the lookup rather than risk leaking or
        # silently overriding self-report with an empty block.
        intime_valid = bool(str(current_intime).strip()) and not pd.isna(
            pd.to_datetime(current_intime, errors="coerce")
        )

        # No index built, no/unknown subject, or no valid intime -> ask-the-patient.
        if index is None or subject_id is None or int(subject_id) <= 0 \
                or not intime_valid \
                or not _subject_known(index, int(subject_id)):
            if index is None:
                note = ("Patient-history index not built (run `uv run "
                        "build_history_index`); ask-the-patient fallback used.")
            elif not intime_valid:
                note = ("No valid arrival time supplied — cannot apply the "
                        "leakage filter; ask-the-patient fallback used.")
            else:
                note = ("No prior record found for this MRN — the pipeline will "
                        "ask the patient about their history instead "
                        "(first-time-patient fallback).")
            return json.dumps({
                "known_patient": False,
                "subject_id": int(subject_id) if subject_id is not None else -1,
                "pmh_block": None,
                "note": note,
            }, indent=2)

        complaint_norm = normalize_complaint_text(chief_complaints or "")
        pmh_block = assemble_pmh_for_stay(
            subject_id=int(subject_id),
            intime=current_intime,
            complaint_norm=complaint_norm,
            index=index,
        )

        has_prior = pmh_block["no_history"] == 0
        fired = sorted(
            c.replace("pmh_", "", 1)
            for c in PMH_FEATURE_COLS
            if c.startswith("pmh_") and pmh_block.get(c) == 1
        )

        return json.dumps({
            "known_patient": True,
            "subject_id": int(subject_id),
            "has_prior_history": bool(has_prior),
            # The 19-column block, ready to paste into the doctor tools'
            # pmh_lookup_json argument verbatim.
            "pmh_block": pmh_block,
            "summary": {
                "n_prior_admissions": pmh_block["n_prior_admissions"],
                "n_prior_ed_visits": pmh_block["n_prior_ed_visits"],
                "days_since_last_admission": pmh_block["days_since_last_admission"],
                "days_since_last_ed": pmh_block["days_since_last_ed"],
                "same_complaint_as_prior": pmh_block["same_complaint_as_prior"],
                "pmh_categories": fired,
            },
            "note": (
                "Real prior-encounter record found (leakage-safe: only visits "
                "before the current arrival). Copy `pmh_block` into the "
                "disposition + v3 tools' `pmh_lookup_json` argument."
                if has_prior else
                "Subject is in the system but has no encounters before the "
                "current arrival — treated as first-time (no_history=1)."
            ),
        }, indent=2)
