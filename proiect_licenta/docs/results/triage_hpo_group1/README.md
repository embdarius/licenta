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
- Run date: **RUN PENDING** (Colab GPU).

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

## Results

**RUN PENDING.** After the Colab run, drop in `tuning_log_acuity_g1.json`,
`tuning_log_disposition_g1.json`, and `triage_hpo_g1_results.json`, then fill:

| Head | Frontier size | Incumbent (trial 0) | Notable Pareto points | Deployed point + clinical rationale |
|---|---|---|---|---|
| Acuity | — | exact —, under —, ESI-5 recall — | — | — |
| Disposition | — | acc —, under — @0.5 | — | — |

The acuity frontier is the headline figure: **exact accuracy vs under-triage vs
ESI-5 recall**, with the incumbent located on the curve and any proposed
"deployed" point justified clinically (a metric delta alone is not enough — the
boost vector reshapes the whole safety profile).

## Chaining note

Once a deployed Pareto point is confirmed (the `selected` block in
`tuned_params_triage_g1.json` — defaults to a knee heuristic, edit by hand to
override), the Group-2 sweep can be re-run on top of it with
`--group group2 --use-group1-best` to search the regularization conditioned on
the confirmed Group-1 config.

## Honest caveats

- Inner-validation search metrics rank trials; the held-out incumbent-vs-frontier
  numbers come from `--group group1 --stage report` (`triage_hpo_g1_results.json`).
- NSGA-II needs a larger budget than TPE to populate the frontier well — prefer
  40–60 trials per head.
- A "winner" needs a clinical rationale, not just a headline delta; the deployed
  point must stay reporting-only unless separately justified.
