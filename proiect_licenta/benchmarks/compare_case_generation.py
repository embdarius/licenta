"""
Case-Generation comparison — Experiment B (Flash vs MedGemma narrative quality)
===============================================================================

Compares the synthetic narratives produced by each LLM backend, kept as two
SEPARATE experiments from the parser/pipeline comparison (Experiment A in
benchmark_pipeline_e2e.py). This scores the Case-Generation agent itself, NOT
the downstream pipeline.

It reads the per-backend case sets written by:
    uv run generate_cases                      -> data/derived/synthetic_cases/
    uv run generate_cases --backend medgemma   -> data/derived/synthetic_cases_medgemma/

and reports, per backend, the deterministic grounding-validator pass rate
(stated age/pain match the row, no fabricated vitals in the opening narrative)
plus retry counts, then prints a few side-by-side narratives for the same
stay_ids so the difference in voice is visible.

Usage:
    uv run python benchmarks/compare_case_generation.py [--show N]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.case_generation import SYNTH_DIR

BACKEND_DIRS = {
    "flash": SYNTH_DIR,
    "medgemma": SYNTH_DIR.parent / "synthetic_cases_medgemma",
}


def _load(backend: str):
    p = BACKEND_DIRS[backend] / "cases.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _grounding_stats(payload: dict) -> dict:
    cases = payload["cases"]
    n = len(cases)
    clean = sum(1 for c in cases if (c.get("grounding") or {}).get("ok"))
    attempts = [(c.get("grounding") or {}).get("attempts", 1) for c in cases]
    return {
        "n": n,
        "clean": clean,
        "flagged": n - clean,
        "pass_rate": (clean / n) if n else float("nan"),
        "mean_attempts": (sum(attempts) / len(attempts)) if attempts else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=3,
                    help="How many side-by-side narratives to print.")
    args = ap.parse_args()

    print("\n" + "#" * 78)
    print("  CASE-GENERATION COMPARISON  (grounding quality per LLM backend)")
    print("#" * 78)

    loaded = {b: _load(b) for b in BACKEND_DIRS}
    available = {b: p for b, p in loaded.items() if p is not None}
    for b, p in BACKEND_DIRS.items():
        status = "ok" if loaded[b] else f"MISSING ({p / 'cases.json'})"
        print(f"  {b:9s}: {status}")
    if not available:
        print("\n  No case sets found. Run `uv run generate_cases` and "
              "`uv run generate_cases --backend medgemma` first.")
        return

    print(f"\n  {'backend':10s}{'cases':>8s}{'clean':>8s}{'flagged':>9s}"
          f"{'pass%':>9s}{'mean_tries':>12s}")
    print("  " + "-" * 56)
    for b, payload in available.items():
        s = _grounding_stats(payload)
        pr = "n/a" if s["pass_rate"] != s["pass_rate"] else f"{s['pass_rate']*100:5.1f}"
        print(f"  {b:10s}{s['n']:>8d}{s['clean']:>8d}{s['flagged']:>9d}"
              f"{pr:>9s}{s['mean_attempts']:>12.2f}")

    # Side-by-side narratives on the intersection of stay_ids (if both present).
    if len(available) == 2:
        fa = {c["stay_id"]: c for c in available["flash"]["cases"]}
        mg = {c["stay_id"]: c for c in available["medgemma"]["cases"]}
        common = [s for s in fa if s in mg][: args.show]
        if common:
            print("\n  Side-by-side narratives (same stay_id):")
            for sid in common:
                cc = fa[sid]["triage_inputs"]["chief_complaints"]
                print(f"\n  stay {sid}  complaint(raw): {cc[:70]}")
                print(f"    flash    : {fa[sid].get('narrative', '')[:110]}")
                print(f"    medgemma : {mg[sid].get('narrative', '')[:110]}")

    print("\n" + "#" * 78 + "\n")


if __name__ == "__main__":
    main()
