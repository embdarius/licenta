# Archive

One-off scripts and outputs kept for thesis-defense reference. **Not part of the active codebase.** Safe to delete once the thesis is submitted.

## Contents

- **`audit_med_vocab.py`** — Ad-hoc audit script run during nurse-agent development to inventory medication strings in `pyxis.csv` and validate the medication-vocabulary normalization. Its findings informed the current `src/proiect_licenta/tools/med_vocab.py`.
- **`audit_med_vocab_output.txt`** — Captured stdout from the run above.
- **`v3_benchmarks_pre_audit/`** — Old UTF-16 stdout captures from the per-agent v3 benchmark scripts (`v3_base_results.txt`, `v3_nurse_results.txt`), superseded by the detailed 2026-06-28 re-benchmark audit (`benchmarks/audit/2026-06-28/`, written up in `docs/results/rebenchmark_v3/`). Kept for reference; the audit reproduces these numbers exactly.

> Note: gitignored local stdout/log captures and superseded result JSONs from the same era live under `artifacts/benchmarks/_archive/` (not committed — MIMIC DUA).
