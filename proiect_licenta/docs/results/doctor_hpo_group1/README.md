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
- Run date: **RUN PENDING** (Colab GPU).

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

## Results

**RUN PENDING.** After the Colab run, drop in `tuning_log_department_g1.json`,
`tuning_log_disposition_g1.json`, and `doctor_hpo_g1_results.json`, then fill:

| Head | Metric | Incumbent | Best Group-1 | Δ |
|---|---|---|---|---|
| Department | macro-F1 | — | — | — |
| Department | accuracy | — | — | — |
| Disposition | ROC AUC | — | — | — |
| Disposition | under-triage @0.40 | — | — | — |
| Disposition | Brier / ECE | — | — | — |

## Chaining note

Once a Group-1 best is confirmed (the `selected` block in
`tuned_params_doctor_g1.json`), the Group-2 sweep can be re-run on top of it with
`--group group2 --use-group1-best`.

## Honest caveats

- Inner-validation search metrics rank trials; held-out incumbent-vs-best numbers
  come from `--group group1 --stage report` (`doctor_hpo_g1_results.json`).
- This is the conditional-on-`lr` limitation, not a fix: Group-1's `lr`/`n_estimators`
  is a cost envelope that plateaus under low-lr + early-stopping. Expected payoff is
  low; the value is a defensibility result, not a redeploy.
