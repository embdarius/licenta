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

The tool ALSO returns a ``med_block`` — the patient's reconciled home-med list
(``MED_FEATURE_COLS``) from their most recent PRIOR ED stay (leakage-safe,
``< intime``; meds are sourced from prior visits because a live patient's
current list is exactly what the nurse is collecting). The doctor tools consume
it via the parallel ``med_lookup_json`` argument; triage has no medication
features so it ignores meds. ``med_block`` is ``None`` for first-time/unknown
patients or when no prior stay carried a medrecon record.

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
    PMH_FEATURE_COLS, assemble_pmh_for_stay, assemble_meds_for_stay,
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


def _latest_encounter_time(index: dict, subject_id: int):
    """Most recent admission/ED-visit timestamp for a subject (lists are sorted,
    so the last element is latest). Returns None if the subject is unknown.

    Used to anchor the 'now' sentinel: MIMIC timestamps are de-identified into
    the future, so a real wall-clock 'now' would precede every encounter and
    report "no prior history". Treating the subject's most recent recorded visit
    as "today's visit" makes the < filter return everything strictly before it —
    the same scenario the offline benchmark exercises with a real stay intime."""
    times = []
    adm = index.get("adm_by_subject", {}).get(subject_id)
    if adm:
        times.append(adm[-1][0])
    ed = index.get("ed_by_subject", {}).get(subject_id)
    if ed:
        times.append(ed[-1][0])
    return max(times) if times else None


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
        default="now",
        description=(
            "When the patient is arriving for care RIGHT NOW (the live runtime), "
            "pass 'now' (the default) — the tool then counts all of a known "
            "patient's recorded encounters as prior history. Only pass an "
            "explicit ISO-8601 timestamp (e.g. '2150-03-14 18:22:00') when you "
            "are reconstructing a HISTORICAL visit and need that exact moment as "
            "the leakage cutoff (the offline benchmark does this); only encounters "
            "strictly before it are used."
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
        "at the bedside. It ALSO returns a `med_block` — the patient's reconciled "
        "home-medication list from their most recent PRIOR visit. Call this right "
        "after intake if the patient provides an MRN; copy the returned `pmh_block` "
        "into the doctor disposition and v3 reassessment tools' `pmh_lookup_json` "
        "argument, and the `med_block` (when non-null) into their `med_lookup_json` "
        "argument, so they use the real record instead of the patient's self-report. "
        "For a first-time / unknown patient (subject_id -1) it returns "
        "known_patient=false (and null blocks) and the pipeline falls back to asking "
        "the patient. All lookups are leakage-safe: only encounters strictly before "
        "the current arrival time are used."
    )
    args_schema: Type[BaseModel] = PatientHistoryLookupInput

    def _run(
        self,
        subject_id: int,
        current_intime: str,
        chief_complaints: str = "",
    ) -> str:
        index = get_history_index()
        sid = int(subject_id) if subject_id is not None else -1

        # Resolve the arrival time. For a LIVE patient ('now', the default — also
        # any empty / skip-equivalent value, since the LLM may omit or fumble the
        # literal string) we anchor to the subject's most recent recorded
        # encounter: MIMIC timestamps are de-identified into the future, so the
        # real wall-clock now would precede every encounter and yield "no prior
        # history". Anchoring to their latest visit (treated as today's) makes the
        # < filter return everything strictly before it. An explicit, parseable
        # ISO timestamp (the offline benchmark path) is used as-is. Only a
        # non-empty value that ALSO fails to parse as a date is treated as
        # invalid -> refuse rather than guess.
        raw_intime = str(current_intime).strip()
        now_sentinel = raw_intime.lower() in (
            "now", "current", "today", "", "unknown", "none", "skip",
            "n/a", "na", "-", "null",
        )
        if now_sentinel:
            resolved_intime = (
                _latest_encounter_time(index, sid)
                if (index is not None and sid > 0) else None
            )
        else:
            resolved_intime = pd.to_datetime(raw_intime, errors="coerce")
        intime_valid = resolved_intime is not None and not pd.isna(resolved_intime)

        # No index built, no/unknown subject, or no valid intime -> ask-the-patient.
        # Order the diagnostics so the note names the ACTUAL reason (an unknown
        # subject must be reported as such, not as an intime problem).
        if index is None or sid <= 0 or not _subject_known(index, sid) \
                or not intime_valid:
            if index is None:
                note = ("Patient-history index not built (run `uv run "
                        "build_history_index`); ask-the-patient fallback used.")
            elif sid <= 0:
                note = ("No MRN provided — treating as a first-time patient "
                        "(ask-the-patient fallback).")
            elif not _subject_known(index, sid):
                note = (f"No record found for ID {sid}. Confirm this is the "
                        f"patient's subject_id (MRN) — NOT a visit/stay number — "
                        f"and that they exist in the built index. Ask-the-patient "
                        f"fallback used.")
            else:
                note = ("Could not parse the supplied arrival time; pass 'now' "
                        "for a live visit. Ask-the-patient fallback used.")
            return json.dumps({
                "known_patient": False,
                "subject_id": sid,
                "pmh_block": None,
                "med_block": None,
                "note": note,
            }, indent=2)

        complaint_norm = normalize_complaint_text(chief_complaints or "")
        pmh_block = assemble_pmh_for_stay(
            subject_id=sid,
            intime=resolved_intime,
            complaint_norm=complaint_norm,
            index=index,
        )

        # Medications: the patient's most recent PRIOR stay's home-med list
        # (leakage-safe, < intime). None when no prior stay carried a medrecon
        # record or the index predates the med build → ask-the-patient fallback.
        med_block = assemble_meds_for_stay(
            subject_id=sid, intime=resolved_intime, index=index,
        )
        med_categories = (
            sorted(c.replace("has_", "", 1).replace("_meds", "").replace("_", " ")
                   for c in med_block
                   if c.startswith("has_") and med_block.get(c) == 1)
            if med_block else []
        )

        has_prior = pmh_block["no_history"] == 0
        fired = sorted(
            c.replace("pmh_", "", 1)
            for c in PMH_FEATURE_COLS
            if c.startswith("pmh_") and pmh_block.get(c) == 1
        )

        return json.dumps({
            "known_patient": True,
            "subject_id": sid,
            "has_prior_history": bool(has_prior),
            # The 19-column block, ready to paste into the doctor tools'
            # pmh_lookup_json argument verbatim.
            "pmh_block": pmh_block,
            # The 11-column medication block from the patient's most recent
            # PRIOR stay's reconciled home-med list (None if none on record).
            # Paste into the doctor tools' `med_lookup_json` argument.
            "med_block": med_block,
            "summary": {
                "n_prior_admissions": pmh_block["n_prior_admissions"],
                "n_prior_ed_visits": pmh_block["n_prior_ed_visits"],
                "days_since_last_admission": pmh_block["days_since_last_admission"],
                "days_since_last_ed": pmh_block["days_since_last_ed"],
                "same_complaint_as_prior": pmh_block["same_complaint_as_prior"],
                "pmh_categories": fired,
                "med_record_found": med_block is not None,
                "n_medications": (med_block["n_medications"] if med_block else 0),
                "medication_categories": med_categories,
            },
            "note": (
                "Real prior-encounter record found (leakage-safe: only visits "
                "before the current arrival). Copy `pmh_block` into the "
                "disposition + v3 tools' `pmh_lookup_json` argument, and "
                "`med_block` (if non-null) into their `med_lookup_json` argument."
                if has_prior else
                "Subject is in the system but has no encounters before the "
                "current arrival — treated as first-time (no_history=1)."
            ),
        }, indent=2)
