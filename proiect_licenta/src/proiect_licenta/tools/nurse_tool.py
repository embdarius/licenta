"""
Nurse Data Collection Tool — CrewAI Tool

Interactively collects vital signs, cardiac rhythm, and medication history
from the patient. Each field can be skipped if the patient doesn't know.
Returns structured JSON consumed by the Doctor prediction tools (v2 and v3).

Output schema is additive: the v2 tool ignores the `rhythm` field added
for v3, so this tool can drive both pipelines without breakage.
"""

import json
import re
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

SKIP_WORDS = {
    "skip", "idk", "i don't know", "i dont know", "unknown",
    "no", "n/a", "na", "-", "not sure", "don't know", "dont know", "",
}


def _parse_numeric(text: str) -> float | None:
    """Try to parse a numeric value from patient input."""
    text = text.strip().lower()
    if text in SKIP_WORDS:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r'[\d.]+', text)
        return float(match.group()) if match else None


def _parse_bp(text: str) -> tuple[float | None, float | None]:
    """Parse blood pressure input like '120/80' or '120 over 80'."""
    text = text.strip().lower()
    if text in SKIP_WORDS:
        return None, None
    text = text.replace("over", "/")
    match = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r'(\d+)', text)
    if match:
        return float(match.group(1)), None
    return None, None


def _parse_timestamp(text: str) -> float | None:
    """Parse an optional reading timestamp into a sortable float key, else None.

    Accepts a 24h clock time ('14:30' -> minutes since midnight) or a relative
    number of minutes ('15', '+15', '15 min', '15 minutes'). Blank / 'now' /
    skip-words -> None (the reading then falls back to collection order). Mixing
    clock-time and relative formats within one session is the user's
    responsibility; if any reading lacks a timestamp the caller uses entry order
    for the whole session anyway, so a single consistent format is all that
    matters."""
    text = text.strip().lower()
    if text in SKIP_WORDS or text in ("now", "current"):
        return None
    m = re.match(r'^(\d{1,2}):(\d{2})$', text)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)))
    m = re.search(r'-?\d+(?:\.\d+)?', text)
    return float(m.group()) if m else None


_VITAL_FIELDS = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp")


def _collect_reading_round(idx: int) -> dict:
    """Interactively collect ONE chronological set of vital readings + cardiac
    rhythm + an optional timestamp. Returns a dict with the 6 vitals, `rhythm`,
    and `ts` (float sort-key or None). `idx` is 1-based, for display only."""
    tag = "first set" if idx == 1 else f"set #{idx}"
    print(f"\n  --- Reading {tag} ---")
    print("  [Nurse]: Temperature? (Fahrenheit, e.g., 98.6)")
    temp = _parse_numeric(input("  [You]:   "))
    print("\n  [Nurse]: Heart rate? (bpm, e.g., 80)")
    hr = _parse_numeric(input("  [You]:   "))
    print("\n  [Nurse]: Respiratory rate? (breaths/min, e.g., 16)")
    rr = _parse_numeric(input("  [You]:   "))
    print("\n  [Nurse]: Oxygen saturation? (%, e.g., 98)")
    o2 = _parse_numeric(input("  [You]:   "))
    print("\n  [Nurse]: Blood pressure? (e.g., 120/80)")
    sbp, dbp = _parse_bp(input("  [You]:   "))
    print("\n  [Nurse]: Cardiac rhythm? (e.g., 'sinus', 'atrial fibrillation',")
    print("           'paced'; skip if no monitor or unknown)")
    rhythm = input("  [You]:   ").strip()
    if rhythm.lower() in SKIP_WORDS:
        rhythm = None
    print("\n  [Nurse]: Time of this reading? (clock time like 14:30, OR minutes")
    print("           since arrival like 15; press Enter / 'skip' to use the")
    print("           order you enter readings in)")
    ts = _parse_timestamp(input("  [You]:   "))
    return {
        "temperature": temp, "heartrate": hr, "resprate": rr,
        "o2sat": o2, "sbp": sbp, "dbp": dbp, "rhythm": rhythm, "ts": ts,
    }


class NurseCollectionInput(BaseModel):
    """Input schema for the Nurse Data Collection Tool."""
    patient_context: str = Field(
        ...,
        description="Brief patient context from prior assessment "
                    "(chief complaints, acuity level, admission status)."
    )


class NurseDataCollectionTool(BaseTool):
    name: str = "nurse_data_collection"
    description: str = (
        "Collects vital signs, cardiac rhythm, medication history, and past "
        "medical history from the patient interactively. Asks for: "
        "temperature, heart rate, respiratory rate, oxygen saturation, blood "
        "pressure, cardiac rhythm (e.g. 'sinus' or 'atrial fibrillation'), "
        "current medications, chronic conditions / PMH (e.g. 'CHF, diabetes'), "
        "and approximate number of prior hospital admissions. It collects ONE "
        "OR MORE chronological reading sets (each with an optional timestamp) so "
        "the doctor models can use real vital trends; the full series rides in "
        "the returned `vital_trajectory` block. The patient can skip any "
        "question. Returns structured JSON consumed by the Doctor prediction "
        "tools (v2 and v3). Call this tool with a brief patient context string."
    )
    args_schema: Type[BaseModel] = NurseCollectionInput

    def _run(self, patient_context: str) -> str:
        """Interactively collect vital signs and medications."""
        print(f"\n{'='*55}")
        print("  NURSE: Vital Signs & Medication Assessment")
        print(f"{'='*55}")
        print("  (Type 'skip' or press Enter if you don't know)\n")

        # Collect ONE OR MORE chronological reading sets. The first set is the
        # primary/arrival snapshot; each additional set lets the doctor models
        # build real vital TRENDS (min/max/last/delta + abnormal-reading counts)
        # instead of a single snapshot, which materially improves the admit/
        # discharge prediction and brings the runtime closer to the feature-
        # vector benchmark (which aggregates every reading in the 4h window).
        rounds = [_collect_reading_round(1)]
        _MAX_ROUNDS = 24  # safety cap against a runaway interactive loop
        while len(rounds) < _MAX_ROUNDS:
            print("\n  [Nurse]: Add another set of readings taken later in the visit?")
            print("           ('y' to add one to capture the vital trend;")
            print("            anything else to finish)")
            if input("  [You]:   ").strip().lower() not in (
                "y", "yes", "another", "more", "add",
            ):
                break
            rounds.append(_collect_reading_round(len(rounds) + 1))

        # Order the rounds chronologically. If EVERY round carries a timestamp,
        # sort by it (so readings can be entered out of order); otherwise keep
        # entry order as the chronological proxy — the no-timestamp fallback.
        # All-or-nothing avoids interleaving timestamped + untimestamped sets.
        if len(rounds) > 1 and all(r["ts"] is not None for r in rounds):
            ordered = sorted(rounds, key=lambda r: r["ts"])
        else:
            ordered = rounds

        # The first ENTERED set is the primary/arrival snapshot — it populates
        # `vital_signs` (the v2 tool + the doctor tools' single-snapshot cascade
        # and longitudinal fallback). The trajectory below carries the full
        # ordered series, from which the doctor tools build min/max/last/delta.
        primary = rounds[0]
        temp = primary["temperature"]; hr = primary["heartrate"]
        rr = primary["resprate"]; o2 = primary["o2sat"]
        sbp = primary["sbp"]; dbp = primary["dbp"]
        rhythm_raw = primary["rhythm"]

        # Build the chronological trajectory per vital (dropping skipped values),
        # plus the rhythm reading sequence, from the ordered rounds. Same wire
        # format as before ({vital: [floats], "rhythm": [strs]}), so the doctor
        # tools / aggregator are unchanged.
        vital_trajectory = {}
        for _vname in _VITAL_FIELDS:
            _seq = [float(r[_vname]) for r in ordered if r[_vname] is not None]
            if _seq:
                vital_trajectory[_vname] = _seq
        _rhythm_readings = [r["rhythm"] for r in ordered if r["rhythm"]]
        if _rhythm_readings:
            vital_trajectory["rhythm"] = _rhythm_readings

        # Medications
        print("\n  [Nurse]: Are you currently taking any medications?")
        print("           (List them separated by commas, or 'none')")
        meds_raw = input("  [You]:   ").strip()
        if meds_raw.lower() in SKIP_WORDS:
            meds_raw = None

        # Prior history — Change 1 in v3 nurse. Free-text chronic conditions
        # / past medical history; parsed against pmh_vocab at inference. The
        # v2 tool ignores `prior_history` and `n_prior_admissions`, so the
        # same nurse output still drives the v2 pipeline.
        print("\n  [Nurse]: Do you have any chronic conditions or past medical")
        print("           history? (e.g., 'CHF, diabetes, prior stroke'; skip")
        print("           if none or unknown)")
        prior_history_raw = input("  [You]:   ").strip()
        if prior_history_raw.lower() in SKIP_WORDS:
            prior_history_raw = None

        # Approximate count of prior hospital admissions, optional. -1 means
        # not collected; 0 means the patient said zero. Both map to the
        # no_history=1 fallback at the doctor tool unless conditions were
        # listed above.
        print("\n  [Nurse]: Roughly how many times have you been admitted to")
        print("           a hospital before? (a number, or skip)")
        prior_adm_raw = input("  [You]:   ").strip()
        if prior_adm_raw.lower() in SKIP_WORDS:
            n_prior_admissions = -1
        else:
            n_prior_admissions = int(_parse_numeric(prior_adm_raw) or -1)

        # Build result. The `rhythm`, `prior_history`, and
        # `n_prior_admissions` fields are new in v3; the v2 tool ignores
        # them, so the same nurse output drives both pipelines.
        result = {
            "vital_signs": {
                "temperature": temp,
                "heartrate": hr,
                "resprate": rr,
                "o2sat": o2,
                "sbp": sbp,
                "dbp": dbp,
            },
            # Multi-reading trajectory (first + optional second readings). The
            # doctor disposition + v3 tools accept this as `vital_trajectory_json`
            # and build real min/max/last/delta features from it; empty -> the
            # tools fall back to the single snapshot. The v2 tool ignores it.
            "vital_trajectory": vital_trajectory,
            "rhythm": rhythm_raw,
            "medications_raw": meds_raw,
            "prior_history": prior_history_raw,
            "n_prior_admissions": n_prior_admissions,
        }

        available = sum(1 for v in result["vital_signs"].values() if v is not None)
        _n_sets = len(rounds)
        _sorted_note = " (timestamp-ordered)" if (
            _n_sets > 1 and all(r["ts"] is not None for r in rounds)
        ) else ""
        print(f"\n  [Nurse]: Thank you! Collected {available}/6 vital signs "
              f"across {_n_sets} reading set(s){_sorted_note}.")
        if rhythm_raw:
            print(f"           Cardiac rhythm: {rhythm_raw}"
                  + (f" (+{len(_rhythm_readings) - 1} more)"
                     if len(_rhythm_readings) > 1 else ""))
        if meds_raw:
            print(f"           Medications recorded: {meds_raw}")
        else:
            print("           No medication data provided.")
        if prior_history_raw:
            print(f"           Prior history: {prior_history_raw}")
        if n_prior_admissions >= 0:
            print(f"           Prior admissions reported: {n_prior_admissions}")
        print(f"{'='*55}")

        return json.dumps(result, indent=2)
