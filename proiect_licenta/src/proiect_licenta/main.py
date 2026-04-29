#!/usr/bin/env python
"""
Multi-Agent Medical Decision Support System — Entry Point

Flow:
  1. Patient describes symptoms in natural language
  2. NLP Parser Agent extracts structured chief complaints + pain score
  3. Triage Agent predicts ESI acuity (1-5) and admission/discharge
  4. Doctor Agent predicts diagnosis category + department (v1, if admitted)
  5. Nurse Agent collects vital signs + medication history
  6. Doctor Agent enhanced reassessment (v2, with nurse data)
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
    """Train the triage ML models v1 (training/train_triage_v1.py)."""
    from proiect_licenta.training.train_triage_v1 import main as train_pipeline
    train_pipeline()


def train_triage_v2():
    """Train the triage ML models v2 with vital signs (training/train_triage_v2.py)."""
    from proiect_licenta.training.train_triage_v2 import main as train_pipeline_v2
    train_pipeline_v2()


def train_doctor():
    """Train the doctor v1 ML models (training/train_doctor.py)."""
    from proiect_licenta.training.train_doctor import main as train_doctor_pipeline
    train_doctor_pipeline()


def train_nurse():
    """Train Doctor v2 models with vital signs + medication features (training/train_nurse.py)."""
    from proiect_licenta.training.train_nurse import main as train_nurse_pipeline
    train_nurse_pipeline()


def train_doctor_v3():
    """Train Doctor v3 base models (catch-all excluded, full dataset, no nurse data)."""
    from proiect_licenta.training.train_doctor_v3 import main as train_pipeline
    train_pipeline()


def train_nurse_v3():
    """Train Doctor v3 with nurse data (catch-all excluded, full dataset, snapshot vitals + meds; Phase B adds longitudinal vitals + rhythm)."""
    from proiect_licenta.training.train_nurse_v3 import main as train_pipeline
    train_pipeline()


if __name__ == "__main__":
    run()
