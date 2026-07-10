#!/usr/bin/env python
"""Entry points for the crew and the training pipelines."""

import sys
import warnings

# Force UTF-8 stdout before crewai wraps the streams. MIMIC loaders and LLM
# output emit non-cp1252 characters that otherwise crash the Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from proiect_licenta.crew import ProiectLicenta

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """Run the medical decision support crew interactively."""
    print("\nMulti-Agent Medical Decision Support System")

    user_message = input("\nPlease describe your symptoms: ")

    if not user_message.strip():
        print("No symptoms provided. Exiting.")
        return

    inputs = {
        'user_message': user_message,
    }

    try:
        result = ProiectLicenta().crew().kickoff(inputs=inputs)
        print("\nMedical assessment result:\n")
        print(result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the triage v1 models."""
    from proiect_licenta.training.train_triage_v1 import main as train_pipeline
    train_pipeline()


def train_triage_v2():
    """Train the triage v2 models (adds vital signs)."""
    from proiect_licenta.training.train_triage_v2 import main as train_pipeline_v2
    train_pipeline_v2()


def train_triage_v3():
    """Train the triage v3 models (v2 features plus PMH)."""
    from proiect_licenta.training.train_triage_v3 import main as train_pipeline_v3
    train_pipeline_v3()


def train_doctor():
    """Train the doctor v1 models."""
    from proiect_licenta.training.train_doctor import main as train_doctor_pipeline
    train_doctor_pipeline()


def train_nurse():
    """Train the doctor v2 models (adds vitals and medications)."""
    from proiect_licenta.training.train_nurse import main as train_nurse_pipeline
    train_nurse_pipeline()


def train_doctor_v3():
    """Train the doctor v3 base models (no nurse data)."""
    from proiect_licenta.training.train_doctor_v3 import main as train_pipeline
    train_pipeline()


def train_nurse_v3():
    """Train the doctor v3 models with nurse data (snapshot and longitudinal vitals, rhythm)."""
    from proiect_licenta.training.train_nurse_v3 import main as train_pipeline
    train_pipeline()


def train_doctor_disposition():
    """Train the doctor disposition model (binary admit/discharge on the full dataset)."""
    from proiect_licenta.training.train_doctor_disposition import main as train_pipeline
    train_pipeline()


def train_icd_resolver():
    """Build the Stage-2 exact-ICD resolver from the doctor v3 training split."""
    from proiect_licenta.training.train_icd_resolver import main as build_resolver
    build_resolver()


def build_history_index():
    """Build the patient-history lookup index. Pass --all-subjects to index the
    full population instead of the generated-cases cohort."""
    import runpy
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_history_index.py"
    runpy.run_path(str(script), run_name="__main__")


def generate_cases():
    """Generate synthetic natural-language test cases from MIMIC-IV tabular rows."""
    import argparse
    import os
    # Importing the module builds no LLM (get_llm runs at kickoff), so it is
    # safe to import before resolving --backend.
    from proiect_licenta.case_generation import (
        generate_and_save_cases, SYNTH_DIR,
        DEFAULT_N_ADMITTED, DEFAULT_N_DISCHARGED, DEFAULT_SEED,
    )

    parser = argparse.ArgumentParser(description="Generate synthetic ED test cases.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only generate the first N sampled cases (smoke test).")
    parser.add_argument("--backend", choices=["flash", "medgemma"], default=None,
                        help="LLM backend used to voice the narratives. "
                             "flash writes synthetic_cases/; medgemma writes "
                             "synthetic_cases_medgemma/ so the two sets stay separate.")
    parser.add_argument("--n-admitted", type=int, default=DEFAULT_N_ADMITTED)
    parser.add_argument("--n-discharged", type=int, default=DEFAULT_N_DISCHARGED)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # Set the backend before generate_and_save_cases reads it at kickoff.
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    backend = os.getenv("LLM_BACKEND", "flash").lower()
    out_dir = (SYNTH_DIR if backend in ("", "flash")
               else SYNTH_DIR.parent / f"synthetic_cases_{backend}")

    generate_and_save_cases(
        n_admitted=args.n_admitted,
        n_discharged=args.n_discharged,
        seed=args.seed,
        limit=args.limit,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    run()
