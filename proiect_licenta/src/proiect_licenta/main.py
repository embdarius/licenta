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

# Force UTF-8 stdout/stderr BEFORE crewai is imported (it wraps the streams).
# MIMIC loaders + the LLM narratives emit non-cp1252 characters (e.g. "≥",
# em-dashes), which crash the default Windows cp1252 console. errors="replace"
# guarantees a print never aborts a long run. See docs/future-work.md known issues.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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


def train_triage_v3():
    """Train the triage ML models v3 (v2 features + PMH; training/train_triage_v3.py)."""
    from proiect_licenta.training.train_triage_v3 import main as train_pipeline_v3
    train_pipeline_v3()


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


def train_doctor_disposition():
    """Train Doctor disposition v3 — peer binary admit/discharge model trained on the FULL 425K dataset (plan section 3, Option B)."""
    from proiect_licenta.training.train_doctor_disposition import main as train_pipeline
    train_pipeline()


def generate_cases():
    """Generate synthetic natural-language test cases from MIMIC-IV tabular rows
    (Phase 4 — Case Generation Agent). Samples a stratified admit/discharge set
    from the held-out test splits, voices each row as a realistic patient
    narrative via the LLM (grounded, validated), and persists them under
    data/derived/synthetic_cases/. Optional `--limit N` for a smoke run.

    Drives benchmarks/benchmark_pipeline_e2e.py.
    """
    import argparse
    from proiect_licenta.case_generation import (
        generate_and_save_cases, DEFAULT_N_ADMITTED, DEFAULT_N_DISCHARGED, DEFAULT_SEED,
    )

    parser = argparse.ArgumentParser(description="Generate synthetic ED test cases.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only generate the first N sampled cases (smoke test).")
    parser.add_argument("--n-admitted", type=int, default=DEFAULT_N_ADMITTED)
    parser.add_argument("--n-discharged", type=int, default=DEFAULT_N_DISCHARGED)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    generate_and_save_cases(
        n_admitted=args.n_admitted,
        n_discharged=args.n_discharged,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    run()
