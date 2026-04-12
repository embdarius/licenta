"""
Nurse Data Collection Tool — CrewAI Tool

Interactively collects vital signs and medication history from the patient.
Each field can be skipped if the patient doesn't know.
Returns structured JSON for the Doctor v2 prediction tool.
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
        "Collects vital signs and medication history from the patient interactively. "
        "Asks the patient for: temperature, heart rate, respiratory rate, "
        "oxygen saturation, blood pressure, and current medications. "
        "The patient can skip any question they don't know the answer to. "
        "Returns structured JSON with the collected data. "
        "Call this tool with a brief patient context string."
    )
    args_schema: Type[BaseModel] = NurseCollectionInput

    def _run(self, patient_context: str) -> str:
        """Interactively collect vital signs and medications."""
        print(f"\n{'='*55}")
        print("  NURSE: Vital Signs & Medication Assessment")
        print(f"{'='*55}")
        print("  (Type 'skip' or press Enter if you don't know)\n")

        # Temperature
        print("  [Nurse]: What is your temperature? (Fahrenheit, e.g., 98.6)")
        temp = _parse_numeric(input("  [You]:   "))

        # Heart rate
        print("\n  [Nurse]: What is your heart rate? (beats per minute, e.g., 80)")
        hr = _parse_numeric(input("  [You]:   "))

        # Respiratory rate
        print("\n  [Nurse]: How many breaths per minute? (e.g., 16)")
        rr = _parse_numeric(input("  [You]:   "))

        # O2 saturation
        print("\n  [Nurse]: What is your oxygen saturation? (percentage, e.g., 98)")
        o2 = _parse_numeric(input("  [You]:   "))

        # Blood pressure
        print("\n  [Nurse]: What is your blood pressure? (e.g., 120/80)")
        sbp, dbp = _parse_bp(input("  [You]:   "))

        # Medications
        print("\n  [Nurse]: Are you currently taking any medications?")
        print("           (List them separated by commas, or 'none')")
        meds_raw = input("  [You]:   ").strip()
        if meds_raw.lower() in SKIP_WORDS:
            meds_raw = None

        # Build result
        result = {
            "vital_signs": {
                "temperature": temp,
                "heartrate": hr,
                "resprate": rr,
                "o2sat": o2,
                "sbp": sbp,
                "dbp": dbp,
            },
            "medications_raw": meds_raw,
        }

        available = sum(1 for v in result["vital_signs"].values() if v is not None)
        print(f"\n  [Nurse]: Thank you! Collected {available}/6 vital signs.")
        if meds_raw:
            print(f"           Medications recorded: {meds_raw}")
        else:
            print("           No medication data provided.")
        print(f"{'='*55}")

        return json.dumps(result, indent=2)
