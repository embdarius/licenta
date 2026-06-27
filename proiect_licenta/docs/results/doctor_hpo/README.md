# Doctor v3 — constrained Optuna hyperparameter search (results)

Committed copy of the final `report`-stage output from the constrained Group-2
hyperparameter sweep of the doctor v3 **department** + **disposition** heads.
The live run writes this to `artifacts/doctor/v3/hpo/` on Drive (gitignored,
MIMIC DUA); [`doctor_hpo_results.json`](doctor_hpo_results.json) holds **only
hyperparameters and aggregate metrics — no patient data** — so it is safe to
commit and is the permanent record for thesis citation. (The per-trial logs
`tuning_log_{department,disposition}.json` stay on Drive; the report JSON here is
the definitive held-out comparison.)

- Engine: [`scripts/tune_doctor_v3_heads.py`](../../../scripts/tune_doctor_v3_heads.py)
- Notebook: [`notebooks/tune_doctor_hpo_v3.ipynb`](../../../notebooks/tune_doctor_hpo_v3.ipynb)
- Write-up: [`docs/agents/doctor-agent.md`](../../agents/doctor-agent.md) → "Hyperparameter search".
- Run date: 2026-06-26, Colab GPU. **10 trials per head** (the incumbent is
  enqueued as trial 0).

**Reporting-only — not redeployed.** The differences are within noise (department)
or too small to justify a redeploy (disposition); the live doctor v3 models are
unchanged. This run *validates* the inherited Group-2 regularization rather than
replacing it.

## What was searched

Only the **inherited "Group-2" XGBoost regularization knobs** (`max_depth`,
`subsample`, `colsample_bytree`, `colsample_bylevel`, `min_child_weight`,
`gamma`, `reg_alpha`, `reg_lambda`). Each head's "Group-1" config was frozen
(department: `lr=0.02`, `n_estimators=3000`, `early_stopping=100`, sqrt-inverse
class weighting, 13-col diagnosis-softmax cascade; disposition: `lr=0.02`,
`n_estimators=5000`, `early_stopping=150`, `scale_pos_weight=sqrt(N_neg/N_pos)`,
full 425K + triage v3 soft cascade). The diagnosis head was already tuned by the
older write-mode `scripts/tune_doctor_v3.py` and is excluded here.

- **Department** objective: maximize macro-F1 across the 11 service classes. No
  constraint.
- **Disposition** objective: maximize ROC AUC **subject to** under-triage (admit
  predicted as discharge, threshold 0.5) ≤ the in-sample incumbent baseline.

## Headline findings (held-out outer-test, calibrated)

These are the `report`-stage numbers on the held-out test split — department on
20,420 rows (live diagnosis cascade used for both arms to isolate the department
Group-2; isotonic-calibrated), disposition on 83,617 rows (live calibrated
incumbent vs. retrained-and-calibrated best, both at the 0.5 threshold).

| Head | Metric | Incumbent | Best searched | Δ |
|---|---|---|---|---|
| **Department** | accuracy | 70.79% | 70.75% | **−0.04pp** |
| | macro-F1 | 0.4898 | 0.4967 | **+0.70pp** |
| **Disposition** | ROC AUC | 0.9138 | 0.9172 | **+0.0034** |
| | accuracy | 83.96% | 84.38% | **+0.42pp** |
| | under-triage | 23.51% | 22.36% | **−1.15pp** (better) |
| | sensitivity | 76.49% | 77.64% | +1.15pp |
| | specificity | 88.30% | 88.30% | flat |
| | Brier | 0.1128 | 0.1103 | −0.0025 (better) |
| | ECE (10-bin) | 0.0036 | 0.0048 | +0.0012 (both excellent) |

Best Group-2 configs (see the JSON for full precision):

- **Department** — `max_depth=6` (vs the incumbent's 10), `gamma≈0.94`,
  `reg_alpha≈1.21`: a shallower, more-regularized tree that trades a hair of
  accuracy for slightly more balanced minority recall (classes 3/4/5/8/10 up,
  class 0 down) — hence macro-F1 +0.70pp at flat accuracy.
- **Disposition** — `max_depth=8`, `min_child_weight=19`, `reg_lambda≈7.9`: a
  more heavily regularized fit that nudges every headline metric the right way,
  **including lowering under-triage** while satisfying the constraint.

**Conclusion:** the inherited Group-2 regularization is **near-optimal** for both
heads. Department is a wash (macro-F1 +0.70pp, accuracy −0.04pp — within noise).
Disposition's best config is a genuine but small improvement on every axis that
matters (AUC +0.0034, accuracy +0.42pp, under-triage −1.15pp, Brier better) —
notably the **constrained** search lowered the dangerous error (under-triage)
rather than trading it away. The gains are below the bar for a redeploy, so this
stands as a *defensibility* result: the previously-unjustified Group-2 box is now
a validated, clinically-safe choice rather than an inherited one.

## Caveats for honest reporting

- **10 trials per head.** TPE starts exploiting around trial 15–25; a longer
  budget (30–50, as recommended for the triage sweep) could surface a marginally
  better config, but the flat department basin and the small disposition deltas
  make a materially different verdict unlikely.
- These **are** held-out outer-test, calibrated numbers (the `report` stage),
  not inner-validation search metrics — so they are directly thesis-citable.
- Department's two report arms share the **live** `diagnosis_model.joblib` for
  the cascade, so the reported delta isolates the department Group-2 effect (it
  is not confounded by a different diagnosis model). The live diagnosis model is
  on hand-picked defaults (`metadata.json: tuned_params_applied = null`) — the
  same parameter family as v3_base.
- **Search/report cascade caveat.** During the *search*, the cascade diagnosis
  model is trained on the inner-train split with the **tuned** diagnosis params
  (`artifacts/doctor/v3/tuned_params.json`, present on Drive: `lr≈0.0098`,
  `max_depth=9`), whereas the *report* and the deployed pipeline cascade from the
  **hand-picked** live model. So the winning Group-2 config was *selected* under
  a slightly different cascade than it was *reported* under. The cascade is held
  constant across all department trials, so this does not bias the Group-2
  ranking, and the effect is immaterial given the flat basin — but for a faithful
  future run the search cascade should be built with hand-picked params to match
  deployment.
- Both heads remain **frozen on Group-1**. A separate future Group-1 study (e.g.
  the disposition tree ceiling, or a joint `lr × max_depth`) could shift these
  optima — see the "Future work — Group-1" notes in the doctor-agent write-up.
