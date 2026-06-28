# Triage v3 — Group-1 multi-objective hyperparameter search (results)

Committed copy of the per-trial logs + Pareto-frontier results from the
**Group-1** sweep of the triage v3 acuity + disposition heads. The live run
writes these to `artifacts/triage/v3/hpo/` on Drive (gitignored, MIMIC DUA);
these JSONs hold **only hyperparameters and aggregate metrics — no patient
data** — so they are safe to commit and are the permanent record for thesis
citation.

- Engine: [`scripts/tune_triage_v3.py`](../../../scripts/tune_triage_v3.py) (`--group group1`)
- Notebook: [`notebooks/tune_triage_v3.ipynb`](../../../notebooks/tune_triage_v3.ipynb) — Section 5
- Write-up: [`docs/agents/triage-agent.md`](../../agents/triage-agent.md) → "Hyperparameter search — Group-1".
- Sibling Group-2 study: [`docs/results/triage_hpo/`](../triage_hpo/).
- Run date: **2026-06-27**, Colab GPU, 10 trials/head. Committed logs: [`tuning_log_acuity_g1.json`](tuning_log_acuity_g1.json), [`tuning_log_disposition_g1.json`](tuning_log_disposition_g1.json), [`triage_hpo_g1_results.json`](triage_hpo_g1_results.json).

## TL;DR — reporting-only, not redeployed

The strongest "keep live" result in the whole HPO program: the multi-objective
search **re-selected the incumbent as its deployed point on both heads**, and
**no frontier config dominates the incumbent** on the held-out outer-test split.
The acuity incumbent holds the **highest ESI-5 recall of the entire Group-1
frontier** while staying competitive on exact accuracy and under-triage — the
hand-tuned `ESI_EXTREME_BOOST` (1.5/1.3/2.0) is empirically validated as a
well-placed point on the (exact, −under-triage, ESI-5 recall) trade-off. The
disposition incumbent has the **best ROC AUC of the frontier**; the rest of that
frontier is just the decision threshold sliding along a fixed ROC curve.
**Live models unchanged.**

## What is searched (and why multi-objective)

Group-1 is the documented clinical-safety config that the Group-2 sweep held
frozen. Here it is the thing being searched — and because these knobs *are* the
safety levers, this is a **multi-objective Pareto** study (NSGA-II), not a
constrained single-objective one. Group-2 stays frozen at the original incumbent
(`FROZEN_GROUP2`) throughout, so the study isolates Group-1.

**Acuity** — `directions = [maximize exact-accuracy, minimize under-triage, maximize ESI-5 recall]`:

| Knob | Search range | Live / incumbent |
|---|---|---|
| `boost_esi1` | [1.0, 2.5] | 1.5 |
| `boost_esi2` | [1.0, 1.8] | 1.3 |
| `boost_esi5` | [1.0, 3.0] | 2.0 |
| `learning_rate` | [0.005, 0.05] log | 0.01 |
| `n_estimators` | {3000, 5000, 8000} | 5000 |

(ESI 3/4 boosts are pinned at 1.0 — the reference floor. The incumbent is
enqueued as trial 0 so it sits on the frontier.)

**Disposition** — `directions = [maximize accuracy, minimize under-triage]`:

| Knob | Search range | Live / incumbent |
|---|---|---|
| `scale_pos_weight_exponent` | [0.5, 1.3] | 1.0 (raw N_neg/N_pos) |
| `learning_rate` | [0.005, 0.05] log | 0.01 |
| `n_estimators` | {3000, 5000, 8000} | 5000 |
| `decision_threshold` | [0.2, 0.6] | 0.5 |

`scale_pos_weight` (a training-time class weight) and the decision threshold (an
operating-point cutoff) are the two tuned levers. **Isotonic calibration is not a
knob** (triage disposition is uncalibrated; the doctor heads keep their fixed
post-hoc calibration).

## Method

- Optuna NSGA-II, single inner train/val split (`random_state=1`) of the
  outer-train, native `mlogloss`/`logloss` early-stopping during search.
- Disposition cascades `predicted_acuity` from an **incumbent** acuity model
  (frozen G1+G2) so the disposition study isolates its own Group-1 levers.
- Resumable SQLite study (`optuna_triage_g1.db`), separate from the Group-2 DB.
- **Reporting-only** — never overwrites the live joblibs.

## Results (held-out outer-test, 83,617 rows; 10 trials/head)

Inner-validation metrics rank the trials; these are the definitive
incumbent-vs-frontier numbers from `--group group1 --stage report`.

### Acuity — Pareto frontier on (exact ↑, under-triage ↓, ESI-5 recall ↑)

| Config (boost 1/2/5) | exact | under-triage | ESI-5 recall | qwk |
|---|---|---|---|---|
| **Incumbent (live) 1.5/1.3/2.0** | **67.55%** | **15.56%** | **26.82%** | 0.6449 |
| trial 1 — 1.23/1.12/1.12 | 68.25% | 16.43% | 16.36% | 0.6470 |
| trial 6 — 1.66/1.10/1.99 | 68.26% | 16.17% | 10.00% | 0.6380 |
| trial 7 — 1.47/1.42/2.09 | 67.18% | 15.13% | 25.45% | 0.6421 |
| trial 8 — 2.34/1.48/2.84 | 66.47% | 14.63% | 25.45% | 0.6347 |

**No frontier point dominates the incumbent.** Every config that raises exact
accuracy (1, 6, 9) worsens *both* under-triage and ESI-5 recall — i.e.
under-boosts the extreme classes, the exact trade the iter-2 `ESI_EXTREME_BOOST`
was designed to refuse. Every config that lowers under-triage (2, 5, 7, 8) gives
up exact accuracy and/or ESI-5 recall. The incumbent has the **highest ESI-5
recall (26.8%) of the whole frontier** across boost-5 values 1.1×–2.9×, and the
knee heuristic's `selected` block is **trial 0 — the incumbent itself**.

### Disposition — Pareto frontier on (accuracy ↑, under-triage ↓)

This frontier is essentially a **decision-threshold operating-point curve** (the
trials move `decision_threshold` 0.22→0.59). The decisive fact:

| Config | threshold | accuracy | under-triage | ROC AUC |
|---|---|---|---|---|
| **Incumbent (live) spw 1.0** | 0.50 | 77.98% | 8.21% | **0.8644** |
| trial 3 — spw 0.74 | 0.41 | 76.75% | 6.72% | 0.8635 |
| trial 6 — spw 1.27 | 0.52 | 77.91% | 8.11% | 0.8633 |
| trial 9 — spw 1.25 | 0.56 | 78.50% | 9.06% | 0.8638 |
| trial 2 — spw 0.52 | 0.59 | 78.89% | 13.34% | 0.8609 |

**ROC AUC is flat across the entire frontier (0.8607–0.8644) with the incumbent
at the top.** Varying `scale_pos_weight`/`lr` does not improve discrimination —
the incumbent is the best model; everything else is the threshold sliding along a
fixed ROC curve. `selected` = **trial 0, the incumbent** again.

## Early stopping / overtraining (sanity)

`mlogloss` (acuity) / `logloss` (disposition) remain the early-stopping metrics
(unchanged; the eval metric was excluded from the search). `n_estimators` is a
*ceiling*: the acuity trials that raised it to 8000 (trials 3, 9) hit
`best_iteration ≈ 7999` — mlogloss still falling at the cap — yet **did not beat
the incumbent**, because QWK plateaus far earlier (~2,400 trees). More trees buy
nothing on the metric that matters; overtraining stays bounded.

## Optional future work (not pursued — low value)

- **Disposition threshold operating-point curve.** The disposition frontier is a
  clean accuracy ↔ under-triage threshold curve (the triage analog of the
  doctor's 0.40 sweep) — a nice thesis figure, but it matters less for triage
  because the triage disposition is a *baseline* the doctor disposition later
  refines, so the triage threshold is not the final admit decision.
- **Acuity trial 7** (under-triage 15.13% vs 15.56%, ESI-5 recall 25.45% vs
  26.82%, exact −0.37pp) is the closest "marginally more conservative" point if a
  future iteration ever wanted slightly lower under-triage — but it lowers ESI-5
  recall *and* exact, so it is **not** a clear win. Not recommended.
- **Group-2 chaining re-run (`--use-group1-best`)** is moot: the Group-1
  `selected` is the incumbent for both heads, so it would re-run Group-2 with no
  effective change.

## Honest caveats

- 10 trials/head; NSGA-II populates a frontier more slowly than TPE, so a larger
  budget (40–60) could add frontier points — but it would not change the verdict
  that the incumbent is non-dominated and clinically well-placed.
- A "winner" would need a clinical rationale, not just a metric delta; here the
  search confirms the deployed point rather than proposing a new one.
