# Doctor v3 — Group-1 hyperparameter search (results)

Committed copy of the per-trial logs + results from the **Group-1** sweep of the
doctor v3 department + disposition heads. The live run writes these to
`artifacts/doctor/v3/hpo/` on Drive (gitignored, MIMIC DUA); these JSONs hold
**only hyperparameters and aggregate metrics — no patient data** — so they are
safe to commit and are the permanent record for thesis citation.

- Engine: [`scripts/tune_doctor_v3_heads.py`](../../../scripts/tune_doctor_v3_heads.py) (`--group group1`)
- Notebook: [`notebooks/tune_doctor_hpo_v3.ipynb`](../../../notebooks/tune_doctor_hpo_v3.ipynb) — Section 5
- Write-up: [`docs/agents/doctor-agent.md`](../../agents/doctor-agent.md) → "Hyperparameter search — Group-1".
- Sibling Group-2 study: [`docs/results/doctor_hpo/`](../doctor_hpo/).
- Run date: **2026-06-27**, Colab GPU, 10 trials/head. Committed logs: [`tuning_log_department_g1.json`](tuning_log_department_g1.json), [`tuning_log_disposition_g1.json`](tuning_log_disposition_g1.json), [`doctor_hpo_g1_results.json`](doctor_hpo_g1_results.json).

## TL;DR — reporting-only, not redeployed

Group-1 confirms the live config is **near-optimal for both heads**. No searched
config beats the incumbent on the held-out outer-test split; the live
`class_weight_exponent = 0.5` (department) and `scale_pos_weight` sqrt
(exponent 0.5, disposition) are validated. **Live models unchanged.** This sits
alongside the Group-2 study's identical verdict.

## What is searched

Group-1 is the per-head cost/weighting config the Group-2 sweep held frozen. Per
the doctor Group-1 plan this is **single-objective** (lower priority than
triage's; mostly a cost/fidelity envelope — the class-weight exponent was already
validated ≈0.52 by the older diagnosis sweep). It keeps the **same objectives +
constraints** as the Group-2 study; Group-2 stays frozen at `FROZEN_GROUP2`. The
diagnosis head is out of scope (already swept).

**Department** — maximize macro-F1 (unconstrained):

| Knob | Search range | Live / incumbent |
|---|---|---|
| `class_weight_exponent` | [0.3, 0.8] | 0.5 (sqrt-inverse) |
| `learning_rate` | [0.01, 0.05] log | 0.02 |
| `n_estimators` | {2000, 3000, 5000} | 3000 |

**Disposition** — maximize ROC AUC **s.t. under-triage ≤ incumbent** (the
constraint and trial-0 incumbent baseline are identical to the Group-2 study):

| Knob | Search range | Live / incumbent |
|---|---|---|
| `scale_pos_weight_exponent` | [0.3, 1.0] | 0.5 (sqrt N_neg/N_pos) |
| `learning_rate` | [0.01, 0.05] log | 0.02 |
| `n_estimators` | {3000, 5000, 8000} | 5000 |

The decision threshold is **not** an Optuna dimension (ROC AUC is threshold-free,
and the live **0.40** threshold already came from
[`benchmarks/sweep_disposition_threshold.py`](../../../benchmarks/sweep_disposition_threshold.py)).
Instead the report stage emits an **operating-point table** over
{0.30…0.60}, with the headline anchored at 0.40. Isotonic calibration stays a
fixed post-hoc step (re-fit per candidate in the report), never a knob.

## Method

- Optuna TPE (department: unconstrained; disposition: `constraints_func`), single
  inner split (`random_state=1`), native early-stopping, lean per-trial fits
  (calibration only in the report stage). Department reuses the live diagnosis
  cascade. Resumable SQLite (`optuna_doctor_g1.db`), separate from the Group-2 DB.
- **Reporting-only** — never overwrites the live joblibs.

## Results (held-out outer-test, calibrated; 10 trials/head)

Inner-validation metrics rank the trials; these are the definitive
incumbent-vs-best numbers from `--group group1 --stage report`.

**Department** (best Group-1 = trial 7: `lr=0.027, n_estimators=5000, class_weight_exponent=0.78`):

| Metric | Incumbent | Best Group-1 | Δ |
|---|---|---|---|
| macro-F1 | 0.4898 | 0.4908 | **+0.10pp** |
| accuracy | 70.79% | 70.52% | **−0.27pp** |

Group-1 found nothing usable: the best inner-val trial (macro-F1 0.5126) *did
not* hold out — on the test split it is +0.10pp macro-F1 for −0.27pp accuracy,
i.e. a slightly higher `class_weight_exponent` (0.78) over-boosts minority
classes at a net accuracy cost. The live 0.5 is near-optimal — and independently
matches the older diagnosis sweep's ≈0.52.

**Disposition** (best Group-1 = trial 4: `lr=0.013, n_estimators=5000, scale_pos_weight_exponent=0.504`; report anchored at the live 0.40 threshold):

| Metric | Incumbent | Best Group-1 | Δ |
|---|---|---|---|
| ROC AUC (threshold-free) | 0.9138 | 0.9140 | **+0.0001** |
| accuracy @0.40 | 83.33% | 83.35% | +0.02pp |
| under-triage @0.40 | 18.14% | 18.52% | +0.38pp (worse) |
| Brier / ECE | 0.1128 / 0.0036 | 0.1126 / 0.0039 | flat |

The selected `scale_pos_weight_exponent ≈ 0.504` *is* the live sqrt (0.5) to
three decimals — Group-1 reproduced the incumbent. The `lr × n_estimators`
envelope is flat: raising the tree cap to 8000 (trial 9) used 6080 trees and did
not improve AUC — the documented disposition tree-ceiling plateau, not
under-regularization. The report also emits an **operating-point table**
(thresholds 0.30–0.60); the best-Group-1 curve sits on top of the incumbent's, a
useful thesis figure for the under-triage ↔ over-triage trade at the 0.40 anchor.

## Early stopping / overtraining (sanity)

`mlogloss` (department) / `logloss` (disposition) remain the early-stopping
metrics — Group-1 did not change them (the eval metric was deliberately excluded
from the search). `n_estimators` is a *ceiling*: `best_iteration` is the real
tree count, chosen where validation loss plateaus. High-`lr` department trials
stop early (e.g. trial 6, `lr=0.035` → `best_iteration=549`), which is the
overtraining penalty working as intended.

## Optional future work (not pursued — low value)

- **Group-2 disposition config (`doctor_hpo`, trial 7).** The *only* movement in
  the whole doctor HPO program is a Group-2 (regularization) result, not a
  Group-1 one: +0.0034 ROC AUC, +0.42pp accuracy, and **−1.15pp under-triage**
  (the clinically good direction) at threshold 0.5. It is kept **documented-only,
  not redeployed** — the gain is below the redeploy bar and re-opens the
  "is +0.4pp worth a retrain" question. If a future iteration specifically wants
  the under-triage reduction it is the obvious candidate, but it is **not
  necessarily worth the compute**.
- **Group-2 chaining re-run (`--use-group1-best`).** Skipped for the doctor: the
  department Group-1 `selected` (cw 0.78) is *worse* out-of-sample, and the
  disposition `selected` ≈ the incumbent, so chaining would either hurt or just
  reproduce the existing Group-2 result.

## Honest caveats

- 10 trials/head; TPE exploits ~15–25. The flat basins make a different verdict
  unlikely, but a longer budget could surface marginal noise-level movement.
- Group-1's `lr`/`n_estimators` is a cost/fidelity envelope that plateaus under
  low-lr + early-stopping — disclose as a conditional-on-`lr` validation, not a
  fix.
