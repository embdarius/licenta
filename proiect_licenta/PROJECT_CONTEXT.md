# Multi-Agent Medical Decision Support System — Project Context

## Overview

This is a **Bachelor's Thesis** project implementing a **Multi-Agent Architecture for Medical Decision Support** using **CrewAI** and **supervised machine learning** trained on the **MIMIC-IV Emergency Department** dataset (~425K real patient encounters).

The system simulates a clinical emergency department workflow: a patient describes their symptoms in natural language, and a pipeline of specialized AI agents processes the input through triage, initial diagnosis, nurse data collection, and enhanced reassessment.

**Tech stack:** Python 3.13, CrewAI 1.9.3, Gemini 2.5 Flash (LLM), XGBoost, scikit-learn, pandas, thefuzz.

---

## Documentation Index

This file is the slim top-level overview. Detailed documentation is split per topic under `docs/`:

| File | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Full agent pipeline diagram, cascading prediction design, shared text preprocessing pipeline, project structure on disk |
| [`docs/agents/nlp-parser-agent.md`](docs/agents/nlp-parser-agent.md) | NLP Parser Agent (LLM): role, input/output contract, `AskPatientTool` |
| [`docs/agents/triage-agent.md`](docs/agents/triage-agent.md) | Triage Agent: acuity + disposition XGBoost models, 2023 features, 66.7% / 75.9% benchmarks, training evolution v1 -> v3b |
| [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md) | Doctor Agent v1 + v2 + v3: 7 XGBoost models (v1 diag/dept, v2 diag/dept, v3 base + nurse diag/dept, **and the new v3 disposition peer model**), diagnosis / department grouping tables, four-way comparison (v1 / v2 / v3 base / v3 with-nurse), medication classification, vital sign processing, longitudinal vitals + rhythm in v3, and the binary admit/discharge refinement model from plan section 3 / Option B |
| [`docs/agents/nurse-agent.md`](docs/agents/nurse-agent.md) | Nurse Agent: interactive collection flow, partial data handling, why a dedicated agent |
| [`docs/agents/case-generation-agent.md`](docs/agents/case-generation-agent.md) | Case Generation Agent (Phase 4, offline/benchmark-only): tabular row → grounded NL patient case, and the 4-way end-to-end vs tabular benchmark (incl. the gate-isolating column) + the runtime multi-reading-vitals fix |
| [`docs/datasets.md`](docs/datasets.md) | MIMIC-IV table reference — used tables + inspected-but-unused tables (`vitalsign.csv`, `pyxis.csv`, `admissions.csv`, clinical notes) with leakage considerations |
| [`docs/future-work.md`](docs/future-work.md) | Why the Doctor v2 gains were modest, prioritized next-step recommendations, Phase 4 Text Generation, Phase 5 Hospital Infrastructure, model-level improvements, known issues |

**If you're new:** read `docs/architecture.md` first, then the four agent files in order (NLP Parser -> Triage -> Doctor -> Nurse).

### Repo Conventions (must-knows for editing)

- **`src/proiect_licenta/paths.py`** is the single source of truth for filesystem paths. All dataset CSVs and artifact directories (`TRIAGE_V1_DIR`, `DOCTOR_V2_DIR`, `DOCTOR_V3_DIR`, `TRIAGE_CSV`, ...) are exported as constants. Never hard-code paths. `DOCTOR_V3_DIR` now holds three artifact bundles: `diagnosis_model.joblib`, `department_model.joblib`, and `disposition_model.joblib` (+ its `_raw` sibling for audit).
- **`src/proiect_licenta/preprocessing.py`** owns `ABBREVIATIONS` and `normalize_complaint_text`. Triage v1/v2, doctor, nurse, and runtime tools all import from here so training and inference can't drift.
- **Datasets and trained model weights are gitignored.** Datasets live in `data/`, artifacts in `artifacts/triage/{v1,v2}/` and `artifacts/doctor/{v1,v2,v3_base,v3}/`. Retrain with `uv run train_*` (see "How to Run").

---

## System at a Glance

```
Patient -> NLP Parser -> Triage -> Doctor v1 -> Nurse -> Doctor v2
           (LLM)        (ML)      (ML)         (interactive) (ML+vitals+meds)
```

- **4 agents**, **5 tasks** (the Doctor runs twice).
- **Live runtime uses Doctor v1 + v2** (14-class label space). The diagram above reflects that.
- **A separate Doctor v3 tier** (catch-all class excluded → 13-class label space, full filtered dataset, longitudinal vitals + rhythm) sits alongside v1/v2 in the repo. Used for training/benchmark comparisons; not currently wired into the live crew.
- **Cascading:** each model's output feeds the next.
- See [`docs/architecture.md`](docs/architecture.md) for the full diagram and pipeline design, and [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md) for the v3 details.

### Headline Benchmark Numbers

| Model | Top-1 | Top-3 | Notes |
|---|---|---|---|
| Triage v1 — Acuity (5 classes) | 66.7% | — | 97.7% within-1-level (no vitals) |
| Triage v2 — Acuity (5 classes) | 68.0% | — | 98.0% within-1-level; vitals added for ambulance/helicopter |
| Triage v3 iter 1 — Acuity (5 classes) | 68.6% | — | 98.2% within-1-level; + 19 PMH features (Doctor v3 Change 1 recipe). +0.60pp vs v2. **Not kept** — hidden ESI 5 recall regression (21.8% → 14.09%). |
| **Triage v3 iter 2 — Acuity (5 classes, kept)** | **67.55%** | — | **98.2% within-1-level; + sections 1.1 (longer training) + 1.2 (ordinal-aware weights + QWK early stopping). −0.46pp headline accuracy vs v2 but −2.11pp under-triage rate and ESI 5 recall recovered to 26.82% (+5pp vs v2). Clinical-safety trade kept over iter 1's higher headline accuracy.** |
| Triage v1 — Disposition (2 classes) | 75.9% | — | ROC AUC 0.84 (no vitals) |
| Triage v2 — Disposition (2 classes) | 76.2% | — | ROC AUC 0.84 |
| **Triage v3 iter 2 — Disposition (2 classes, kept)** | **78.0%** | — | **ROC AUC 0.86. +1.80pp vs v2; `pmh_Blood and Blood-Forming Organs` is rank 5 in the disposition model — PMH is materially driving the admit/discharge signal. Iter 1 and iter 2 are essentially identical on disposition (77.96% → 77.98%).** |
| Doctor v1 — Diagnosis (14 classes) | 50.2% | 83.6% | 3.05x over random |
| Doctor v1 — Department (11 classes) | 59.1% | 92.5% | majority baseline 59.9% |
| Doctor v2 — Diagnosis (14 classes) | 52.4% | 84.9% | +2.3pp over v1 |
| Doctor v2 — Department (11 classes) | 65.0% | 93.7% | +5.9pp over v1; beats majority baseline |
| Doctor v3 base — Diagnosis (13 classes) | 60.1% | 82.7% | catch-all excluded; full 102K filtered dataset |
| Doctor v3 base — Department (11 classes) | 60.7% | 92.3% | |
| Doctor v3 nurse — Diagnosis (13 classes) | **64.1%** | **85.7%** | +4.0pp over v3 base; PMH (Change 1) + Tier A vocab expansion (A2) |
| Doctor v3 nurse — Department (11 classes) | **70.8%** | **94.1%** | +10.1pp over v3 base; Tier A softmax cascade (A3) + isotonic calibration (A4) |
| Triage v3 cascade — Disposition (binary, baseline for next row) | 80.2% | — | ROC AUC 0.8894, ECE 0.0748. Computed on the disposition test split (83,617 rows). Triage v3's own headline 77.98% is from the *triage* pipeline's split — see doctor-agent.md for the two-reference-numbers explanation. |
| **Doctor disposition v3 (binary, calibrated, deployment-ready)** | **84.0%** | — | **ROC AUC 0.9138, ECE 0.0036, Brier 0.1128. +3.77pp accuracy / +0.0244 AUC over the triage v3 cascade baseline on the same 83,617 test rows. Specificity 88.3% (+8.3pp), sensitivity 76.5% (−4.05pp at threshold 0.5). Plan section 3 / Option B — peer binary admit/discharge model trained on the full 418K stays. **Live in the runtime crew as of 2026-05-30** (`doctor_disposition_tool` between nurse and reassessment; the reassessment gates on this task's `is_admitted` rather than triage's).** |
| **Doctor Stage-2 exact-ICD (3-char rollup, on v3 nurse)** | **—** | **—** | **Retrieval cascade (TF-IDF prototype cosine + prevalence, α=0.60). Within the top-5 predicted categories: 62.9% of admitted patients get their exact 3-char ICD in the top-5-cats×top-5-codes set, 55.5% in a flat top-10; oracle-category (true category given) 67.0%@5 / 79.9%@10. Blend beats prevalence-only (+11.8pp@5) and cosine-only (+16.7pp@5). Full-code secondary: 49.2%@5 oracle. Advisory only — does not change category/department. Shipped 2026-06-11. v3_base (pre-nurse) vs v3-nurse: nurse data adds +1.0pp E2E union / +1.3pp flat-10 (entirely via Stage-1 category recall 90.6%→92.8%); the Stage-2 oracle ceiling is model-agnostic and identical for both.** |

v1/v2 (14-class) and v3 (13-class) are not on identical test sets and shouldn't be compared as if they were. Full per-class metrics and confusion matrices live in the per-agent docs.

> **Note on v3 nurse history:** The current v3 nurse numbers above include **Change 1** (PMH features, 2026-05-21) **and Tier A** (A2 vocab expansion + A3 diagnosis-softmax cascade + A4 isotonic department calibration, 2026-05-22). The Tier A deltas vs the pre-Tier-A Change 1 baseline were **-0.09pp diagnosis (flat) and +2.18pp department**, dominated by A4 calibration. See [`docs/agents/doctor-agent.md#tier-a--vocab-expansion--softmax-cascade--isotonic-calibration`](docs/agents/doctor-agent.md#tier-a--vocab-expansion--softmax-cascade--isotonic-calibration) for the per-lever attribution and the MED-dominance caveat. A1 (Optuna macro-F1 sweep) still planned.

---

## Design Decisions

1. **Two-phase doctor assessment** — The Doctor Agent runs twice: v1 (triage data only) and v2 (with nurse data). This allows direct comparison of predictions before and after vital signs/medication data, demonstrating the clinical value of additional data collection.
2. **LLM for NLP parsing, ML for prediction** — The NLP Parser uses Gemini (LLM) for natural language understanding. All prediction models use supervised ML (XGBoost) trained on 400K+ real encounters for reliable, auditable predictions.
3. **Cascading prediction** — Each model's output feeds into the next: acuity -> disposition -> diagnosis -> department. This mirrors clinical flow and improves downstream predictions.
4. **Soft class weighting** — Uses sqrt(inverse_frequency) to balance minority class recall without destroying majority class accuracy.
5. **Category grouping** — Small diagnosis/department categories are merged to ensure sufficient training samples per class. The "Nervous System" ICD-9/10 split is an artifact that was corrected by merging.
6. **100K training cap for doctor models** — To keep training times reasonable while benchmarking. Can be increased to full 157K for production.
7. **Admitted-only for doctor models** — Diagnosis and department prediction only applies to admitted patients. Discharged patients get a discharge summary instead.
8. **No ICU data** — Project focuses entirely on the Emergency Department pathway.
9. **v3 tier (catch-all excluded)** — A separate Doctor v3 model tier excludes the "Symptoms, Signs, Ill-Defined" catch-all class (33% of admitted patients) which acts as a labeling artifact. v3 trains on the full filtered dataset (~102K rows, no 100K cap) and the v3 with-nurse variant adds longitudinal vitals + cardiac rhythm aggregated from `vitalsign.csv`. v1/v2 are kept as 14-class baselines for direct thesis comparison.
10. **Change 1 — PMH features (Doctor v3 nurse, 2026-05-21)** — Past Medical History features are derived from prior MIMIC encounters: 13 binary `pmh_<diagnosis_group>` flags parsed from the "Past Medical History" section of prior discharge summaries (`mimic-iv-notes/discharge.csv`, ~3.3 GB) and OR'd with ICD-derived flags from `hosp/diagnoses_icd.csv`, plus 6 repeat-visit numerics (`n_prior_admissions`, `days_since_last_admission`, `same_complaint_as_prior`, etc.). Leakage-free by construction (`prior_admittime < current_intime`). +1.08pp diagnosis / +1.50pp department over the pre-Change-1 v3 nurse baseline. The Nurse Agent now also asks the patient about chronic conditions at inference — zero-fill fallback identical to first-time-patient training rows.
11. **Stage-2 exact-ICD resolution (Doctor v3, 2026-06-11)** — A retrieval cascade on top of the v3 diagnosis model predicts the *exact* ICD diagnosis within each predicted category (3-char rollup headline + full-code secondary), ranking candidates by `α·cosine + (1−α)·prevalence` where cosine compares the complaint's TF-IDF vector to per-code prototype centroids built from the training split. α=0.60 tuned offline. It does **not** retrain or alter the category/department predictions — it's advisory, surfaced as `diagnosis_prediction.exact_diagnoses` and gracefully skipped if the artifact isn't built. No new dependencies (reuses the TF-IDF vectorizer). See [`docs/agents/doctor-agent.md#stage-2--exact-icd-resolution-within-categories-shipped-2026-06-11`](docs/agents/doctor-agent.md#stage-2--exact-icd-resolution-within-categories-shipped-2026-06-11).

---

## How to Run

```bash
# Install dependencies
uv sync

# Train/retrain triage v1 ML models (~90 minutes CPU; pre-vital baseline)
uv run train_models

# Train/retrain triage v2 ML models with vital signs (~90 minutes CPU)
uv run train_triage_v2

# Train/retrain triage v3 ML models with vital signs + PMH (~30-45 min Colab T4 GPU,
# ~120 min CPU; the discharge.csv PMH parse dominates). See notebooks/train_triage_v3.ipynb.
uv run train_triage_v3

# Train/retrain doctor v1 ML models (~30 minutes)
uv run train_doctor

# Train/retrain doctor v2 ML models with nurse data (~45 minutes)
uv run train_nurse

# Train/retrain doctor v3 base (catch-all excluded, full dataset, no nurse) (~40 minutes)
uv run train_doctor_v3

# Train/retrain doctor v3 with nurse (longitudinal vitals + rhythm + meds) (~70 minutes)
uv run train_nurse_v3

# Train/retrain doctor disposition v3 (Option B from plan section 3 — peer binary
# admit/discharge model on full 425K, soft cascade from triage v3, isotonic-calibrated)
# (~60 minutes on Colab T4 GPU; see notebooks/train_doctor_disposition.ipynb)
uv run train_doctor_disposition

# Build the Stage-2 exact-ICD resolver from the v3 training split (~1 minute; light
# loader, no GPU. Add --verify to assert the split matches the full v3 loader once.)
uv run train_icd_resolver

# Run the full 4-agent, 5-task system interactively
uv run crewai run

# Or equivalently
uv run run_crew

# Run benchmarks
uv run python benchmarks/benchmark_triage_v1.py             # Triage v1 models
uv run python benchmarks/benchmark_triage_v2.py             # Triage v2 models (with vitals)
uv run python benchmarks/benchmark_triage_v2_realistic.py   # Triage v2 under realistic missing-vitals scenario
uv run python benchmarks/benchmark_triage_v3.py             # Triage v3 (PMH) — head-to-head vs v2 on the same test rows
uv run python benchmarks/benchmark_doctor.py                # Doctor v1 models
uv run python benchmarks/benchmark_nurse.py                 # Doctor v1 vs v2 comparison
uv run python benchmarks/benchmark_doctor_v3.py             # Doctor v3 base (13 classes)
uv run python benchmarks/benchmark_nurse_v3.py              # Doctor v3 base vs v3 with-nurse
uv run python benchmarks/benchmark_doctor_disposition.py    # Doctor disposition v3 (Option B) — head-to-head vs triage v3 cascade dispo on the same test rows, with per-subgroup deltas + feature-importance audit
uv run python benchmarks/benchmark_icd_resolution.py        # Doctor Stage-2 exact-ICD (rollup + full; blend vs prevalence-only vs cosine-only; oracle + end-to-end)
uv run python benchmarks/compare_all_versions.py            # Four-way table v1/v2/v3-base/v3-nurse
```

### Phase 4 — Case Generation Agent (offline / benchmark-only)

```bash
# Generate synthetic NL test cases from MIMIC-IV tabular rows (LLM + grounding;
# samples a stratified admit/discharge set from the held-out test splits).
# Heavy: runs the loaders once. Use --limit N for a smoke run.
uv run generate_cases

# Run the 20 cases end-to-end through the full crew and compare against the
# tool-direct and feature-vector tabular baselines on the same stay_ids.
uv run python benchmarks/benchmark_pipeline_e2e.py   # [--limit N] [--skip-feature-vector] [--skip-e2e]
```

### Environment Variables (`.env`)
```
MODEL=gemini/gemini-2.5-flash
GEMINI_API_KEY=<key>
```

---

## Status Summary

- **Phase 1 — Triage System:** Complete (v1, v2, v3). See [`docs/agents/triage-agent.md`](docs/agents/triage-agent.md).
- **Triage v3 iteration 1 — PMH features (2026-05-26, superseded):** Initial v3 — reused Doctor v3 Change 1's 19-feature PMH block on the full triage dataset. +0.60pp acuity / +1.80pp disposition vs v2; walk-in patients gained ~2× the ambulance lift, validating the "PMH fills the signal gap where vitals are masked" hypothesis. **Not kept** — had a hidden ESI 5 recall regression (21.8% → 14.09%) that iteration 2 was scoped to fix.
- **Triage v3 iteration 2 — Sections 1.1 + 1.2 (2026-05-27, kept):** Stacked on top of iter 1: lr 0.02→0.01, n_estimators 3000→5000, plus an `ESI_EXTREME_BOOST` per-class sample-weight multiplier (ESI 1: 1.5×, 2: 1.3×, 5: 2.0×) and a custom `neg_quadratic_kappa` early-stopping metric on the acuity head. Headline acuity dropped 0.46pp vs v2 (67.55%), but **under-triage rate dropped 2.11pp** (17.67% → 15.56%) and **ESI 5 recall recovered to 26.82%** (+5pp vs v2). ESI 1, ESI 2, and ESI 5 all improved vs both v2 and iter 1. Disposition was essentially identical to iter 1 (77.96% → 77.98%, +1.81pp vs v2). Kept for clinical-safety reasons: ED triage literature is unambiguous that under-triage carries mortality risk. Full deltas live in [`docs/agents/triage-agent.md#triage-v3`](docs/agents/triage-agent.md#triage-v3) and the shipped-experiments section of [`docs/future-work.md`](docs/future-work.md#triage-v3-iteration-2--sections-11--12-longer-training--ordinal-aware-acuity-shipped-2026-05-27-kept).
- **Triage v3 iteration 3 — Section 1.5 (2026-05-28, reverted):** Hand-curated red-flag keyword features (44 binary columns across cardiac/neuro/respiratory/trauma/sepsis/etc.) added on top of iter 2. Headline metrics moved by noise (every Δ ≤ ±0.04pp; under-triage actually +0.02pp regressed); 31/44 acuity + 26/44 disposition red flags had non-zero gain (rf_palpitations rank 13 on disposition) but the signal was already captured by TF-IDF + iter 2's ESI extreme-class weights. Reverted in commit `cc348e6` — same structural diagnosis as the doctor v3 Bio_ClinicalBERT experiment. See ["entry 6" in tried-and-reverted](docs/future-work.md#6-triage-v3-section-15--hand-curated-red-flag-keyword-features--reverted-lift-within-noise-floor).
- **Triage v3 — inference rewiring (2026-05-29, shipped):** `triage_tool.py` now loads `artifacts/triage/v3/` and builds the 2070-feature vector including the 19-column PMH block. The NLP Parser collects `prior_history` (free-text chronic conditions) and `n_prior_admissions` (int) from every patient at intake; skip-equivalents zero-fill to the first-time-patient pattern (`no_history=1`) the model saw on 39% of training rows. Same `pmh_vocab` inference parsing as the doctor v3 tool, so triage and doctor share PMH semantics. v3 is now the live runtime model (replacing v2 in `triage_tool.py`). See [Inference status](docs/agents/triage-agent.md#inference-status--wired-2026-05-29).
- **Triage MRN/EHR lookup (2026-06-03, shipped):** Closed the last runtime↔feature-vector gap for triage on returning patients. `PatientHistoryLookupTool` is now registered on the **triage agent**, `triage_tool.py` gained a `pmh_lookup_json` override arg, and `triage_assessment_task` step 1b calls the lookup when the patient gives an MRN — so triage's acuity/disposition models receive the **real** prior-encounter numerics (`days_since_last_admission`, `days_since_last_ed`, `n_prior_ed_visits`, `same_complaint_as_prior`) instead of zero-filling them, matching what the triage feature-vector benchmark always fed. Self-report path unchanged for first-time/unknown patients. The shared `parse_pmh_lookup` + new read-only `pmh_self_report_discrepancy` helpers were hoisted into `pmh_features.py` (single source of truth); triage + both doctor tools import them and emit a `self_report_not_in_record` reconciliation note (record still drives the feature vector). The benchmark's `tool_direct` path now also forwards the block to the triage call, so `tool_direct_lookup` exercises the triage-side lookup (the E2E real-crew column already did via the crew). A fast re-run (`--skip-e2e`, 2026-06-03) shows the triage-PMH wiring is **net-neutral at n=20** — ESI acuity held at 70.0 (no prediction flipped); the effect is below the small-sample noise floor (scaling the cohort, planned near-future, is the unblock).
- **Medication MRN/EHR lookup (2026-06-03, shipped):** Extended the same EHR lookup to medications for returning patients. `PatientHistoryLookupTool` now also returns a `med_block` — the patient's reconciled `medrecon` home-med list from their most recent **PRIOR** stay (leakage-safe `< intime`, via new `build_pmh_index` `med_by_stay` + `assemble_meds_for_stay`); `med_vocab.py` hosts the shared `MED_FEATURE_COLS` / `med_block_from_rows` / `parse_med_lookup` / `med_self_report_discrepancy`. The doctor disposition + v3 reassessment tools gained a parallel `med_lookup_json` override (triage has no medication features, so it's untouched); new patients keep the free-text self-report path. Chosen **prior-visit** (not current-visit) semantics for leakage-safety and live-deployment honesty — a strong proxy for stable chronic meds, but it won't perfectly match the current-visit `medrecon` the feature-vector column uses if meds changed. Tasks + benchmark (`med_lookup_json` through `tool_direct_lookup` + E2E) wired; the index was rebuilt (`uv run build_history_index`) to populate `med_by_stay`. The 2026-06-03 fast re-run (`--skip-e2e`) is **net-neutral at n=20 with no regression** — only 5/20 patients had a prior medrecon to fetch (rest are first-in-window visits), so the override had almost no cases to act on; measuring its real effect needs the larger cohort (**scaling — planned, near-future**). See [MRN-based EHR lookup at triage](docs/agents/triage-agent.md#mrn-based-ehr-lookup-at-triage--wired-2026-06-03).
- **Phase 2 — Doctor v1 (initial assessment):** Complete. See [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md).
- **Phase 3 — Nurse Agent + Doctor v2:** Complete. See [`docs/agents/nurse-agent.md`](docs/agents/nurse-agent.md) and [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md).
- **Doctor v3 tier (catch-all excluded, longitudinal vitals + rhythm):** Complete. See the Phase 3 section of [`docs/agents/doctor-agent.md`](docs/agents/doctor-agent.md). A separate round of v3 improvements (`is_surgical` flag, Bio_ClinicalBERT embeddings, pairwise refiner) was tried and reverted — see "Empirical findings" in [`docs/future-work.md`](docs/future-work.md).
- **Doctor v3 nurse — Change 1 (PMH features):** Complete (2026-05-21). +1.08pp diagnosis / +1.50pp department on the same held-out test split. All 19 new PMH features show non-zero feature-importance gain (verification gate passed); none crack the top 50, matching the existing nurse features' "many small interactions" contribution pattern. See [Change 1 section](docs/agents/doctor-agent.md#change-1--pmh-features) in the doctor agent doc.
- **Doctor v3 nurse — Change 2 (multilabel sibling head):** Tried and reverted (2026-05-21). Best blend lift was +0.04pp top-3 — well below the predicted +1.5-3pp band. The multilabel head's signal turned out to be correlated with the softmax (same features, related labels), not orthogonal, so linear blending couldn't extract additional accuracy. See "Empirical findings — experiments tried and reverted" entry 4 in [`docs/future-work.md`](docs/future-work.md).
- **Doctor v3 nurse — Change 4 (two-stage department routing):** Tried and reverted (2026-05-21). Net delta vs the legacy single department head was **−1.22pp top-1 / −0.63pp top-3** on the same 20,420-row test split. Gate binary accuracy landed at 82.0% (31.7% of surgical patients misrouted into the medical head), bounding the end-to-end ceiling — soft-blending cannot recover the 18% of misrouted patients. TRAUM did recover +6.7pp under the surgical-only head (partial hypothesis validation), but volume-weighted MED regression of −1.5pp on 12,045 stays eats the surgical gains many times over. Implementation reverted to commit `c40f15b`; this writeup retained. See "Empirical findings — experiments tried and reverted" entry 5 in [`docs/future-work.md`](docs/future-work.md).
- **Doctor v3 nurse — Tier A (A2 + A3 + A4):** Shipped 2026-05-22. Three stacked changes — PMH vocabulary expansion (397 → 596 keywords), diagnosis-softmax cascade (13 `diag_proba_*` columns into the department model instead of a single argmax int), and isotonic calibration on the department model (FrozenEstimator + CalibratedClassifierCV, 10% held-out calibration set). Net deltas vs pre-Tier-A Change-1 baseline: **diagnosis −0.09pp top-1 (essentially flat), department +2.18pp top-1 (68.61% → 70.79%)**. A4 calibration delivered +1.46pp on its own — far above the predicted +0.1-0.3pp band — but the gain is concentrated on MED (+8.7pp recall, 12,045 stays), with most non-MED departments regressing. Cohen's κ on department actually dropped from 0.5009 to 0.4948 (less uniform agreement). See [Tier A section](docs/agents/doctor-agent.md#tier-a--vocab-expansion--softmax-cascade--isotonic-calibration) in the doctor-agent doc for per-lever attribution and per-class table.
- **Doctor disposition v3 — Option B (peer admit/discharge model, plan section 3):** Shipped 2026-05-28 (training + benchmark). New `artifacts/doctor/v3/disposition_model.joblib` is a calibrated binary classifier trained on the FULL 418K stays (not the admitted-only ~102K slice used by diagnosis/department). Soft cascade from triage v3 (5 acuity softmax + 1 dispo probability cols) instead of the hard cascade used elsewhere. Isotonic calibration on a 10% held-out fit slice (A4 pattern). Net deltas vs the triage v3 cascade dispo baseline on the same 83,617-row test split: **+3.77pp accuracy (80.2% → 84.0%)**, **+0.0244 ROC AUC (0.8894 → 0.9138)**, **−0.0713 ECE (0.0748 → 0.0036 — calibration crushed)**, +8.30pp specificity, −4.05pp sensitivity. The under-triage rate is higher at the default 0.5 threshold (operating-point trade-off, not a model-quality problem since AUC is strictly better). Per-subgroup lift concentrates as plan section 3 predicted: elderly +5.26pp, prior-admission +5.01pp, polypharmacy +4.79pp. PMH wiring audit passes (19/19 PMH cols with non-zero gain); soft cascade audit passes (6/6 cascade cols with non-zero gain); `triage_disposition_proba_admit` is the #1 non-TF-IDF feature by gain. See [Doctor disposition v3 section](docs/agents/doctor-agent.md#doctor-disposition-v3--peer-model-plan-section-3-option-b-shipped-2026-05-28) for the full numbers, calibration curve, per-subgroup table, and feature-importance audit.
- **Doctor v3 + disposition — inference rewiring (2026-05-30, shipped):** The live crew is now on the full v3 stack — `doctor_tool_v3_base.py` (NEW) for the pre-nurse initial assessment (13-class label space, top-3 diag/dept with probabilities), `doctor_disposition_tool.py` (NEW) between nurse and reassessment for the refined admit/discharge prediction, and `doctor_tool_v3.py` (already on disk, now wired) for the post-nurse enhanced diagnosis + department. v1/v2 tool files stay on disk for thesis benchmarks but are no longer registered with the live crew. New `doctor_disposition_task` slots between nurse and reassessment; the reassessment now gates on **its** `is_admitted` rather than triage's, so a triage-discharge that the disposition model flips to admit still triggers the full diagnosis/department workup (and vice versa). `tasks.yaml` was updated for the new task ordering + top-3 with probabilities output for diagnosis/department + hedged top-2 ESI output when triage's top-2 are within 0.10 + calibrated P(admit)/P(discharge) display on the disposition output. End-to-end smoke test passed; threshold tuning + asymmetric-cost retrain are documented as the next disposition-model levers.
- **Doctor v3 nurse — A1 (Optuna macro-F1 sweep):** IN PROGRESS, paused at **10/30 trials** (2026-05-22). Best trial #2 on the 16,336-row inner-val split: macro_f1 = **0.5536**, top-1 = 0.6306. vs the hand-picked baseline (macro_f1 = 0.5529 on outer test) the delta is **+0.0007 — within sampling noise**. TPE not yet converged after 10 cold-start trials; best `class_weight_exponent` ≈ 0.52 (≈sqrt default, matching the existing setup); low-lr trials are undertrained at the 3000-iter cap. `artifacts/doctor/v3/tuned_params.json` was written with trial #2's config but **no final retrain has been applied** — the production v3 nurse model still uses hand-picked defaults. Resume plan: run 20 more trials (~3 hours) in a future session, then decide based on whether best macro_f1 clears 0.56. See ["Experiments in progress" in future-work](docs/future-work.md#experiments-in-progress).
- **Phase 4 — Text Generation / Case Generation Agent:** In progress. The Case Generation Agent (`src/proiect_licenta/case_generation.py`, `uv run generate_cases`) translates MIMIC-IV tabular rows into grounded natural-language patient cases (LLM phrases the narrative; everything else is deterministic from the row; a validator enforces no invented facts). `benchmarks/benchmark_pipeline_e2e.py` runs a stratified 20-case set end-to-end through the full crew and compares accuracy against three tabular baselines on the same stay_ids — **tool-direct**, **feature-vector** (unconditional, the existing benchmark methodology), and **feature-vector-gated** (gated by the disposition model's own verdict). The four columns cleanly separate the three losses: disposition-gate, runtime feature-degradation, and LLM/NLP. Offline/benchmark-only — not in the live patient crew. **20-case run** (13 admit / 7 discharge, 0 grounding-flagged, disposition threshold 0.40): acuity ties at 70% across the tabular columns (E2E carries LLM run-to-run jitter). The benchmark surfaced two fixes that closed most of the tool-direct↔feature-vector gap: (1) the **multi-reading-vitals fix** — the runtime under-admitted because it rebuilt longitudinal vitals from a single snapshot, so the nurse now collects an optional 2nd reading → shared `build_longitudinal_block` → new `vital_trajectory_json` arg on the disposition + v3 tools (snapshot fallback preserved); (2) **disposition threshold 0.50→0.40** (max F1/Youden on the 83K test split). Combined, tool-direct disposition rose 65%→**90%** and diagnosis@1 23%→**61.5%** (Dx coverage 6/13→**11/13**, vs the gated ceiling 12/13). The residual gap to the *ungated* feature-vector is the disposition gate itself (by design); next levers are a stronger disposition model and scaling beyond 20 cases. See [`docs/agents/case-generation-agent.md`](docs/agents/case-generation-agent.md).
- **Phase 5 — Hospital Infrastructure:** Planned. See [`docs/future-work.md`](docs/future-work.md).

Known issues, model-level improvement opportunities, and the full prioritized roadmap — including the analysis of *why* the v2 gains were modest and which unused MIMIC-IV tables could close the gap — are consolidated in [`docs/future-work.md`](docs/future-work.md).
