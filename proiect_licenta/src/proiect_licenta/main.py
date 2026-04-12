#!/usr/bin/env python
"""
Multi-Agent Medical Decision Support System — Entry Point

Flow:
  1. Patient describes symptoms in natural language
  2. NLP Parser Agent extracts structured chief complaints + pain score
  3. Triage Agent predicts ESI acuity (1-5) and admission/discharge
  4. Doctor Agent predicts diagnosis category + department (if admitted)
"""

import sys
import warnings

from proiect_licenta.crew import ProiectLicenta

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """Run the medical decision support crew interactively."""
    print("\n" + "=" * 60)
    print("  [+] Multi-Agent Medical Decision Support System")
    print("  Powered by MIMIC-IV + CrewAI")
    print("=" * 60)

    user_message = input("\nPlease describe your symptoms: ")

    if not user_message.strip():
        print("No symptoms provided. Exiting.")
        return

    inputs = {
        'user_message': user_message,
    }

    try:
        result = ProiectLicenta().crew().kickoff(inputs=inputs)
        print("\n" + "=" * 60)
        print("  MEDICAL ASSESSMENT RESULT")
        print("=" * 60)
        print(result)
        print("=" * 60 + "\n")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the triage ML models (run data_pipeline.py)."""
    from proiect_licenta.data_pipeline import main as train_pipeline
    train_pipeline()


def train_doctor():
    """Train the doctor ML models (run doctor_data_pipeline.py)."""
    from proiect_licenta.doctor_data_pipeline import main as train_doctor_pipeline
    train_doctor_pipeline()


if __name__ == "__main__":
    run()
