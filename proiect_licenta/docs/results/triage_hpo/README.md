# Triage v3 — constrained Optuna hyperparameter search (results)

Committed copy of the per-trial logs from the constrained Group-2 hyperparameter
sweep of the triage v3 acuity + disposition heads. The live run writes these to
`artifacts/triage/v3/hpo/` on Drive (gitignored, MIMIC DUA); these JSONs hold
**only hyperparameters and aggregate metrics — no patient data** — so they are
safe to commit and are the permanent record for thesis citation.

- Engine: [`scripts/tune_triage_v3.py`](../../../scripts/tune_triage_v3.py)
- Notebook: [`notebooks/tune_triage_v3.ipynb`](../../../notebooks/tune_triage_v3.ipynb)
- Write-up: [`docs/agents/triage-agent.md`](../agents/triage-agent.md) → "Hyperparameter search".
- Run date: 2026-06-25/26, Colab L4 GPU.

## What was searched

Only the **inherited "Group-2" XGBoost regularization knobs** (`max_depth`,
`subsample`, `colsample_bytree`, `colsample_bylevel`, `min_child_weight`,
`gamma`, `reg_alpha`, `reg_lambda`). The documented "Group-1" clinical-safety
config (`lr=0.01`, `n_estimators=5000`, `early_stopping=150`,
`ESI_EXTREME_BOOST`, `neg_quadratic_kappa`) was frozen.

- **Acuity** objective: maximize quadratic-weighted κ (QWK) **subject to**
  under-triage ≤ the in-sample baseline (the search cannot trade away the
  iter-2 under-triage win). Metrics on the 66,894-row inner-validation split.
- **Disposition** objective: maximize ROC AUC (no constraint; binary head).

## Headline findings (inner-validation)

| Head | Trials | Best config | vs incumbent |
|---|---|---|---|
| Acuity | 9 | trial #5: QWK **0.6448**, under-triage 15.38%, within-1 98.07% | within the noise floor of the incumbent (QWK ~0.6425–0.6449); **no config beats it** |
| Disposition | 10 | **trial #0 = the incumbent config**: ROC AUC **0.8641** | incumbent is the **outright best**; all searched configs scored 0.8565–0.8641 |

**Conclusion:** the inherited Group-2 regularization is **near-optimal**. For
disposition the deployed config won outright; for acuity every feasible config
sits within ~0.002 QWK of the incumbent and the constraint correctly excluded
the one trial (acuity #6) that raised under-triage above baseline. This turns
the previously-unjustified Group-2 hyperparameters into a defended choice.

## Caveats for honest reporting

- These are **inner-validation search metrics**, not held-out test-set numbers.
  The definitive incumbent-vs-best comparison comes from the `report` stage
  (`--stage report`), which loads the live deployed model read-only and
  re-evaluates the best config on the outer-test split.
- The **acuity** study's constraint baseline is trial #2 (the first completed
  trial), because the enqueued incumbent trial-0 did not complete during the
  initial CPU run; this does not change the conclusion given the tight
  clustering. The disposition study's trial-0 incumbent completed normally.
