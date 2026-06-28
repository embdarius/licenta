# v3 Re-Benchmark Audit — 2026-06-28

A detailed, fully-auditable re-run of **every live v3 model**, plus the generated-cases
end-to-end benchmark on **both LLM backends** (Flash + MedGemma). Purpose: confirm the
documented numbers reproduce exactly, and expand the evidence with every relevant metric
(per-class, confusion, calibration, per-threshold operating points, graded ICD near-miss,
and the NLP-parser cost decomposition).

- **Raw artifacts:** [`benchmarks/audit/2026-06-28/`](../../../benchmarks/audit/2026-06-28/)
  (`tabular/` + `generated_cases/{flash,medgemma,comparison}/`).
- **Harness:** `benchmarks/benchmark_tabular_full.py`, `benchmarks/benchmark_pipeline_e2e.py --out-dir`,
  `benchmarks/compare_backends.py`, shared metrics in `benchmarks/_metrics.py`; driven by
  `notebooks/rebenchmark_v3_audit.ipynb` (Colab). Loaders are disk-cached
  (`src/proiect_licenta/loader_cache.py`, `LOADER_CACHE_DIR`) so the discharge.csv PMH parse
  runs once.
- **Split:** 80/20 stratified, `random_state=42` (identical to training). Tabular test sets:
  triage/disposition **83,617**, doctor diagnosis/department/ICD **20,420**. Generated cases:
  **250** (163 admitted), `--parser-llm --skip-e2e`, clinical-term parse prompt, no clinicalize map.

## ✅ Regression gate: PASS (29/29)

Every headline metric reproduces `artifacts/benchmarks/tabular/ceiling.json` within **<1e-4**
(rounding only) — see [`regression_check.json`](../../../benchmarks/audit/2026-06-28/tabular/regression_check.json).
Internal invariants hold: graded@k ≥ strict@k (0 violations), per-class supports sum to test_n,
confusion totals match.

---

## Tabular results

### Triage — Acuity (ESI 1–5, ordinal), n = 83,617

| metric | value |
|---|---|
| exact | **0.6755** |
| within-1 / within-2 | 0.9818 / 0.9991 |
| top-2 / top-3 | 0.9426 / 0.9922 |
| MAE / mean signed error | 0.344 / −0.010 (near-unbiased) |
| κ quadratic / linear / unweighted | **0.6449** / 0.549 / 0.473 |
| balanced accuracy | 0.574 |
| macro AUC (OvR) | 0.886 |
| log-loss / multiclass Brier / ECE (top-label) | 0.753 / 0.440 / 0.0266 |
| under-triage / over-triage | 0.1556 / 0.1690 |
| **critical under-triage** (true ESI 1–2 → pred 3–5) | **0.0840 overall / 0.2148 of critical** |

Per-class recall (precision): ESI-1 0.603 (0.574), ESI-2 0.708 (0.636), ESI-3 0.672 (0.778),
ESI-4 0.617 (0.424), **ESI-5 0.268 (0.178)** — ESI-5 (support 220) is essentially unpredictable,
the known rarest class. By arrival transport: ambulance/helicopter exact **0.693** (under 0.099,
real vitals) vs walk-in **0.666** (under 0.190, vitals masked) — the real-vitals cohort is easier,
as designed. CSVs: `triage_acuity_per_class.csv`, `triage_acuity_confusion_*.csv`,
`triage_acuity_by_arrival.csv`.

### Triage — Disposition (binary, raw/uncalibrated)

acc@0.5 **0.780**, ROC AUC **0.864**, PR-AUC 0.792, Brier 0.149, ECE **0.0746** (poorly
calibrated — the reason the doctor stage re-does disposition with calibration). At 0.5: sens 0.776,
spec 0.782, under 0.224, over 0.218. Full operating-point table {0.15…0.90}:
`triage_disposition_thresholds.csv`.

### Doctor — Diagnosis (13-class), v3_base → v3_nurse (n = 20,420)

| | top-1 | top-3 | top-5 | κ | macro-F1 | balanced-acc | macro-AUC | MRR |
|---|---|---|---|---|---|---|---|---|
| v3_base | 0.601 | 0.827 | 0.906 | 0.549 | 0.516 | 0.522 | 0.889 | 0.732 |
| v3_nurse | **0.641** | 0.857 | 0.928 | 0.591 | 0.553 | 0.542 | 0.912 | 0.763 |
| **Δ nurse** | **+4.0** | +3.0 | +2.2 | +0.042 | +3.7 | +2.0 | +0.023 | +3.1 |

The nurse lift is **broad and genuine** — recall rises across most classes (Circulatory +10.0pp,
Genitourinary +7.2, Endocrine +4.5, Respiratory +5.3, Digestive +2.6), with only small dips
(Blood −3.4, Other −1.9). Both macro-F1 and balanced accuracy improve, so it is *not* a
majority-class artifact. CSV: `doctor_diagnosis_per_class.csv`, `doctor_diagnosis_confusion_*`.

### Doctor — Department (11-class), v3_base → v3_nurse

| | top-1 | top-3 | κ | macro-F1 | precision-macro | **balanced-acc** |
|---|---|---|---|---|---|---|
| v3_base | 0.607 | 0.923 | 0.422 | 0.445 | 0.414 | 0.496 |
| v3_nurse | **0.708** | 0.941 | 0.495 | 0.490 | 0.582 | **0.452** |
| **Δ nurse** | **+10.1** | +1.8 | +0.072 | +4.5 | +16.8 | **−4.4** |

> **Honest caveat (document this):** the headline **+10.1pp** department gain is *partly
> majority-class concentration*. General Medicine recall jumps 0.654 → **0.873**, while most
> minority departments drop (General Surgery 0.560 → 0.365, Other 0.239 → 0.082, Neurosurgery
> 0.419 → 0.347, Trauma 0.595 → 0.540, OB/GYN 0.462 → 0.400). **Balanced accuracy therefore
> falls 4.4pp** even as top-1 rises 10.1pp. Precision-macro rises sharply (+16.8pp), so the nurse
> data + isotonic calibration make the model *more confident and precise on the dominant
> General-Medicine prediction* rather than uniformly better. Quote top-1 **with** macro-F1
> (+4.5) and balanced-acc (−4.4). This contrasts with diagnosis, where the gain is uniform.

### Doctor — Disposition v3 (calibrated binary) — calibration + operating points

**Calibration is excellent:** ECE **0.0036**, MCE **0.0148**; the 10-bin reliability table is
essentially diagonal (max gap 1.5pp). ROC AUC **0.9138**, PR-AUC 0.860, Brier 0.1128,
log-loss 0.358. This is what makes the threshold knob meaningful — a probability maps to a real
admit rate. (`doctor_disposition_calibration.csv`.)

**Per-threshold operating points** (`doctor_disposition_thresholds.csv` has {0.15,0.20,0.30,
0.40-live,0.50} + the 0.10–0.90 sweep):

| threshold | accuracy | sensitivity | specificity | **under-triage** | **over-triage** | bal-acc | MCC |
|---|---|---|---|---|---|---|---|
| 0.15 | 0.750 | 0.946 | 0.636 | **0.054** | 0.364 | 0.791 | 0.568 |
| 0.20 | 0.782 | 0.924 | 0.699 | 0.076 | 0.301 | 0.811 | 0.601 |
| 0.30 | 0.820 | 0.865 | 0.793 | 0.135 | 0.207 | 0.829 | 0.638 |
| **0.40 (LIVE)** | **0.833** | **0.819** | **0.842** | **0.181** | **0.158** | **0.830** | 0.650 |
| 0.50 (orig. doc) | 0.840 | 0.765 | 0.883 | 0.235 | 0.117 | 0.824 | 0.653 |

The live **0.40** threshold trades **0.6pp accuracy** vs 0.50 to cut **under-triage 23.5% → 18.1%**
(−5.4pp; sensitivity 0.765 → 0.819 — fewer missed admissions) at the cost of +4.1pp over-triage.
Balanced accuracy peaks at ~0.40. Stricter under-triage targets are reachable and quantified
(0.30 → 13.5%, 0.20 → 7.6%, 0.15 → 5.4%), so any operating point is now defensible.

**Subgroup lift over the triage-v3 cascade** (accuracy, at live 0.40 —
`doctor_disposition_subgroups.csv`): elderly **+7.7pp** (0.719 → 0.796), prior-admission **+7.4**,
polypharmacy **+6.8**, repeat-visitor +6.4, abnormal-vitals +4.6, non-sinus rhythm +3.3 — exactly
the high-value cohorts the design targeted. **Feature-group gain** (uncalibrated model,
`doctor_disposition_feature_groups.csv`): TF-IDF 84.6%, longitudinal vitals 4.6%, structured 4.1%,
PMH 1.9%, soft-cascade 1.9%, snapshot vitals 1.7%, medications 1.2%. Wiring fully exercised —
**all 19 PMH** and **all 6 soft-cascade** columns earn non-zero gain; the single top feature is the
triage soft-cascade `triage_disposition_proba_admit` (gain 55.6), then `n_tachypnea_readings` and
`elderly`.

> **Lift-vs-triage interpretation caveat:** the doctor model's honest lift over the triage
> cascade is the **threshold-independent** set — ROC AUC **+0.024**, accuracy **+5.8pp**, Brier
> −0.023, ECE **0.0036 vs 0.075**. The `lift_vs_triage_at_live` under/over-triage deltas compare
> two *differently-calibrated* probability scales at one fixed cut and look unfavorable; that's an
> artifact, not a regression. Use the threshold table to pick the doctor model's *own* operating
> point.

### Stage-2 exact-ICD resolver (rollup primary + full-code secondary)

Oracle ceiling (true category known), rollup, **blend+vitals**: @1 0.331, **@5 0.691**, @10 0.813,
MRR 0.496 (blend-without-vitals @5 0.670; prevalence-only 0.552; cosine-only 0.503 — the blend is
clearly best). Full-code blend+vitals @5 0.525.

**Graded near-miss credit is the headline for ICD:** even when the exact 3-char code isn't in the
top-5, the predicted codes are clinically adjacent. Rollup blend+vitals graded@5 — **Gemini 0.915**
(semantic), **ICD-tree 0.784** (shared rubric/chapter), **TF-IDF 0.726** (lexical). So strict @5 of
0.69 *understates* clinical usefulness; lead with the graded metrics.

End-to-end (predicted category — the real pipeline), v3-nurse rollup blend+vitals: union **0.649**,
flat@10 0.569, conditional 0.699, Stage-1 category recall top-1 0.641 / top-5 **0.928**. Conditional
≈ oracle, so **almost the entire end-to-end gap is Stage-1 category error, not Stage-2 resolution.**
Nurse-vs-base ICD lift is modest (union +1.1pp, flat +1.7pp) and comes entirely via better Stage-1
category recall. Full breakdown (every variant × granularity × engine):
`icd_resolution_summary.csv` / `icd_resolution_full.json`.

---

## Generated cases (250 NL narratives, `--parser-llm`)

### Reference modes (backend-invariant) and the parser cost

`feature_vector` (full cached features) → `feature_vector_gated` (disposition-gate) → `tool_direct`
(exact tabular complaint) → `parser_llm` (LLM-parsed complaint). The first three never touch the
LLM, so they are **identical** for both backends and serve as the reference ladder. The clean
**NLP-parser cost** is `parser_llm − tool_direct` (`parser_cost.csv`):

| metric (admitted-GT) | feature_vector_gated | tool_direct | Flash parser-llm | MedGemma parser-llm |
|---|---|---|---|---|
| acuity exact | 0.684 | 0.680 | **0.660** | 0.604 |
| diagnosis top-1 | 0.472 | 0.491 | 0.423 | 0.423 |
| department top-1 | 0.589 | 0.595 | 0.479 | **0.497** |
| ICD union | 0.534 | 0.552 | **0.466** | 0.460 |

The gate and runtime-feature steps cost ~nothing here (`tool_direct` ≈ `feature_vector_gated`), so
**almost the whole E2E drop is the LLM parser**. It hurts the complaint-sensitive heads most:
department −10 to −12pp, ICD −9 to −12pp, diagnosis −6.8pp (both backends); acuity/disposition are
robust (−2 to −8pp).

### Flash vs MedGemma (head-to-head, parser-llm mode)

`backend_comparison_metrics.csv` (every metric) + `backend_comparison_parser_per_case.csv`
(per-narrative, who parsed better).

| dimension | Flash | MedGemma | winner |
|---|---|---|---|
| complaint token Jaccard | **0.598** | 0.529 | **Flash** (+6.9pp) |
| age extracted ok | 250/250 | 250/250 | tie |
| gender extracted ok | 232/250 | 232/250 | tie |
| arrival transport ok | 242/250 | 241/250 | ~tie |
| acuity exact / quad-κ | **0.660 / 0.615** | 0.604 / 0.539 | **Flash** |
| acuity bias (mean signed err) | +0.008 (unbiased) | −0.152 (over-acute) | **Flash** |
| diagnosis top-1 / top-3 | 0.423 / 0.626 | 0.423 / 0.626 | tie |
| department top-1 / top-3 | 0.479 / 0.681 | **0.497 / 0.687** | **MedGemma** |
| disposition ROC AUC / acc@0.40 | **0.888 / 0.792** | 0.873 / 0.784 | **Flash** |
| acuity under-triage | 0.160 | **0.132** | MedGemma |
| acuity over-triage | **0.180** | 0.264 | Flash |
| ICD strict@1 / @5 / union | **0.178 / 0.307 / 0.466** | 0.172 / 0.301 / 0.460 | Flash (marginal) |

**What each backend is better at:**
- **Flash** maps lay language to the *charted clinical short-form* more faithfully (higher complaint
  Jaccard), which flows into **better ESI acuity** (decisively), **better disposition** (AUC + acc),
  marginally **better exact-ICD**, and an **unbiased** acuity error. E.g. "pain in my left testicle"
  → Flash "testicular pain" (matches chart) vs MedGemma "left testicular pain" (over-specified, misses
  the charted token).
- **MedGemma** edges **department top-1** (+1.8pp) and has slightly lower acuity **under-triage**
  (0.132 vs 0.160) — but at the cost of an **over-triage / over-acute bias** (over-triage 0.264;
  mean signed error −0.152) and notably weaker acuity agreement (quad-κ 0.539 vs 0.615).
- **Diagnosis is a dead tie** (0.423 / 0.626) — the parser differences don't change the diagnosis
  outcome distribution on these 250 cases.

Net: **Flash is the stronger parser for this pipeline overall** (acuity, disposition, ICD, complaint
fidelity); MedGemma's only clear edge is department, plus a lower-under-triage-but-higher-over-triage
acuity profile. Consistent with the prior `docs/llm-backend.md` headline.

---

## Key takeaways

1. **The v3 stack is stable and the documented numbers are accurate** — exact ceiling reproduction.
2. **Disposition is the strongest new evidence:** excellent calibration (ECE 0.0036) makes the
   0.40 operating point quantitatively defensible (−5.4pp under-triage vs the old 0.50 doc), and the
   model beats the triage cascade most in elderly / prior-admission / polypharmacy cohorts.
3. **Three caveats to carry into the thesis:** (a) the department nurse gain is majority-class
   concentrated (top-1 +10.1 but balanced-acc −4.4); (b) ICD strict @5 understates clinical value —
   lead with graded (Gemini 0.915 / tree 0.784); (c) the disposition lift over triage is AUC / Brier /
   calibration, not the fixed-threshold under/over.
4. **The NL gap is the parser, not the pipeline** — `tool_direct ≈ feature_vector_gated`, and
   Flash's main advantage over MedGemma is preserving acuity fidelity.
