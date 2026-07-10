"""Nurse data collection tool.

Interactively collects vital signs, cardiac rhythm, and medication history; each
field can be skipped. Returns structured JSON consumed by the doctor tools. The
output schema is additive: the v2 tool ignores the rhythm field added for v3, so
this tool drives both pipelines.
"""

import json
import re
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from proiect_licenta.interaction import make_ask

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


def _collect_reading_round(idx: int, ask) -> dict:
    """Interactively collect ONE chronological set of vital readings + cardiac
    rhythm + an optional timestamp. Returns a dict with the 6 vitals, `rhythm`,
    and `ts` (float sort-key or None). `idx` is 1-based, for display only.

    `ask(prompt, kind, meta=...)` does the asking - stdin at the terminal, or the
    web channel behind the UI. The `kind`/`meta` hints let the web client render
    the right input widget; the terminal ignores them."""
    setno = {"meta": {"reading": idx}}
    temp = _parse_numeric(ask("Temperature? (Fahrenheit, e.g., 98.6)", "number", **setno))
    hr = _parse_numeric(ask("Heart rate? (bpm, e.g., 80)", "number", **setno))
    rr = _parse_numeric(ask("Respiratory rate? (breaths/min, e.g., 16)", "number", **setno))
    o2 = _parse_numeric(ask("Oxygen saturation? (%, e.g., 98)", "number", **setno))
    sbp, dbp = _parse_bp(ask("Blood pressure? (e.g., 120/80)", "bp", **setno))
    rhythm = ask(
        "Cardiac rhythm? (e.g., sinus, atrial fibrillation, paced; "
        "skip if no monitor or unknown)", "rhythm", **setno).strip()
    if rhythm.lower() in SKIP_WORDS:
        rhythm = None
    ts = _parse_timestamp(ask(
        "Time of this reading? (clock time like 14:30, OR minutes since arrival "
        "like 15; skip to use entry order)", "text", **setno))
    return {
        "temperature": temp, "heartrate": hr, "resprate": rr,
        "o2sat": o2, "sbp": sbp, "dbp": dbp, "rhythm": rhythm, "ts": ts,
    }


def build_nurse_payload(
    rounds: list[dict],
    meds_raw: str | None,
    prior_history_raw: str | None,
    n_prior_admissions: int,
) -> dict:
    """Assemble the canonical nurse-output dict from one or more reading rounds.

    Single source of truth for the nurse payload shape: the stdin tool and the
    web backend both call it so they produce an identical structure (vital_signs,
    vital_trajectory, rhythm, medications_raw, prior_history, n_prior_admissions).

    `rounds` is a list of dicts each holding the 6 vitals + `rhythm` + `ts`
    (float sort-key or None), exactly as ``_collect_reading_round`` builds. The
    FIRST entered round is the primary/arrival snapshot; if every round carries a
    timestamp the trajectory is ordered by it, otherwise entry order is the
    chronological proxy. Skip-equivalent meds/PMH should already be normalized to
    None by the caller; ``n_prior_admissions`` is -1 when not collected.
    """
    if not rounds:
        rounds = [{
            "temperature": None, "heartrate": None, "resprate": None,
            "o2sat": None, "sbp": None, "dbp": None, "rhythm": None, "ts": None,
        }]

    # Order the rounds chronologically. If EVERY round carries a timestamp,
    # sort by it (so readings can be entered out of order); otherwise keep entry
    # order as the chronological proxy. All-or-nothing avoids interleaving.
    if len(rounds) > 1 and all(r.get("ts") is not None for r in rounds):
        ordered = sorted(rounds, key=lambda r: r["ts"])
    else:
        ordered = rounds

    # The first ENTERED set is the primary/arrival snapshot.
    primary = rounds[0]
    temp = primary["temperature"]; hr = primary["heartrate"]
    rr = primary["resprate"]; o2 = primary["o2sat"]
    sbp = primary["sbp"]; dbp = primary["dbp"]
    rhythm_raw = primary["rhythm"]

    # Chronological trajectory per vital (dropping skipped values) + rhythm seq.
    vital_trajectory = {}
    for _vname in _VITAL_FIELDS:
        _seq = [float(r[_vname]) for r in ordered if r.get(_vname) is not None]
        if _seq:
            vital_trajectory[_vname] = _seq
    _rhythm_readings = [r["rhythm"] for r in ordered if r.get("rhythm")]
    if _rhythm_readings:
        vital_trajectory["rhythm"] = _rhythm_readings

    return {
        "vital_signs": {
            "temperature": temp,
            "heartrate": hr,
            "resprate": rr,
            "o2sat": o2,
            "sbp": sbp,
            "dbp": dbp,
        },
        "vital_trajectory": vital_trajectory,
        "rhythm": rhythm_raw,
        "medications_raw": meds_raw,
        "prior_history": prior_history_raw,
        "n_prior_admissions": n_prior_admissions,
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
    # Web session channel (set by the live backend). None -> stdin (terminal).
    channel: Any = None

    def _run(self, patient_context: str) -> str:
        """Interactively collect vital signs and medications."""
        ask = make_ask(self.channel, default_role="Nurse")
        if self.channel is None:
            print("  NURSE: Vital Signs & Medication Assessment")
            print("  (Type 'skip' or press Enter if you don't know)\n")

        # Collect ONE OR MORE chronological reading sets. The first set is the
        # primary/arrival snapshot; each additional set lets the doctor models
        # build real vital TRENDS (min/max/last/delta + abnormal-reading counts)
        # instead of a single snapshot, which materially improves the admit/
        # discharge prediction and brings the runtime closer to the feature-
        # vector benchmark (which aggregates every reading in the 4h window).
        rounds = [_collect_reading_round(1, ask)]
        _MAX_ROUNDS = 24  # safety cap against a runaway interactive loop
        while len(rounds) < _MAX_ROUNDS:
            again = ask(
                "Add another set of readings taken later in the visit? "
                "(yes to capture the vital trend; no to finish)", "yesno")
            if again.strip().lower() not in (
                "y", "yes", "another", "more", "add",
            ):
                break
            rounds.append(_collect_reading_round(len(rounds) + 1, ask))

        # Medications
        meds_raw = ask(
            "Are you currently taking any medications? "
            "(List them separated by commas, or 'none')", "text").strip()
        if meds_raw.lower() in SKIP_WORDS:
            meds_raw = None

        # Prior history - Change 1 in v3 nurse. Free-text chronic conditions
        # / past medical history; parsed against pmh_vocab at inference. The
        # v2 tool ignores `prior_history` and `n_prior_admissions`, so the
        # same nurse output still drives the v2 pipeline.
        prior_history_raw = ask(
            "Do you have any chronic conditions or past medical history? "
            "(e.g., CHF, diabetes, prior stroke; skip if none or unknown)",
            "text").strip()
        if prior_history_raw.lower() in SKIP_WORDS:
            prior_history_raw = None

        # Approximate count of prior hospital admissions, optional. -1 means
        # not collected; 0 means the patient said zero. Both map to the
        # no_history=1 fallback at the doctor tool unless conditions were
        # listed above.
        prior_adm_raw = ask(
            "Roughly how many times have you been admitted to a hospital "
            "before? (a number, or skip)", "integer").strip()
        if prior_adm_raw.lower() in SKIP_WORDS:
            n_prior_admissions = -1
        else:
            n_prior_admissions = int(_parse_numeric(prior_adm_raw) or -1)

        # Build the result via the shared assembler so the stdin tool and the web
        # backend produce an identical payload. rhythm, prior_history, and
        # n_prior_admissions are v3-only; the v2 tool ignores them, so the same
        # nurse output drives both pipelines.
        result = build_nurse_payload(
            rounds, meds_raw, prior_history_raw, n_prior_admissions,
        )
        rhythm_raw = result["rhythm"]
        _rhythm_readings = result["vital_trajectory"].get("rhythm", [])

        if self.channel is None:
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

        return json.dumps(result, indent=2)
