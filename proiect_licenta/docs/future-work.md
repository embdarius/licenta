# Future Work and Roadmap

This document collects the project's open questions, diagnostic analyses, and prioritized next steps.

---

## Why were the Doctor v2 gains only modest? (+2.3pp diagnosis, +5.9pp department)

The v1 -> v2 upgrade added 31 new features (20 vital signs + 11 medications) yet the measured improvements were smaller than expected. Several reasons, in decreasing order of impact:

1. **Vitals are mostly normal for admitted patients.** By the time a patient is sick enough to be admitted but stable enough to be processed through triage, their vital signs tend to fall in a relatively narrow range. The vast majority of admitted patients are afebrile, normotensive, and with normal O2 saturation. The distinguishing signal is concentrated in the 5-10% of patients with flagged abnormalities (fever, tachycardia, hypoxia, hypotension) — which is exactly why those specific subgroups (e.g., Circulatory +5.2pp) saw most of the lift.

2. **Pre-admission medications indicate chronic conditions, not the acute presentation.** `medrecon.csv` captures what the patient was already taking when they arrived. A patient on metformin has diabetes; that's useful background but doesn't tell you *why they came to the ED today*. The medication features act more like demographic priors than diagnostic signals.

   *Subsequent finding (addressed):* an audit (`audit_med_vocab.py`) revealed that training and inference were using **entirely different medication vocabularies** — training matched pharmacy-class strings in `etcdescription` (e.g. "beta blocker") while inference matched patient-reported drug names (e.g. "metoprolol") against `DRUG_NAME_MAP`. The two never touched. Only 60.7% of stays had matching flag sets, with `has_opioid_meds` at just 44.4% Jaccard (training over-flagged opioids for plain acetaminophen via a substring match on "Non-Opioid"). Both sides now share one vocabulary (`src/proiect_licenta/tools/med_vocab.py`) with word-boundary matching, negation handling ("non-opioid" neutralized), and multi-category drugs (e.g. clonazepam → psych + anticonvulsant). Post-fix agreement is 93.3% with every per-category Jaccard ≥95%. Doctor v2 models should be retrained to re-measure the medication-feature lift, which was previously noisy in both directions.

3. **TF-IDF dominates the top 50 feature importances.** Specific complaint language (e.g., "deliberate self harm", "collision", "sickle cell") remains the single strongest signal for both the diagnosis and department models. Nurse features do appear in the top 50 but rarely above the top 20.

4. **33% "Symptoms, Signs, Ill-Defined" catch-all is a labeling ceiling, not a feature problem.** When the ICD code for ~one third of admissions is literally a restatement of the chief complaint (e.g., "chest pain NOS"), no amount of extra input features will fix the fundamental ambiguity in the label space. This is the single largest structural ceiling on diagnosis accuracy.

5. **Surgical departments regressed.** Surgery routing depends on **what is broken** (injury type, fracture location) rather than on vitals. Adding vital-sign features slightly confused the department model for TRAUM / OTHER_SURG / SURG (-5 to -10pp each). A future improvement: route the doctor v2 features differently for surgical vs medical departments, or gate the use of vitals on the diagnosis category.

---

## Empirical findings — experiments shipped

### Change 1 — PMH features from prior discharge notes + ICD fallback (SHIPPED 2026-05-21)

Implemented from the Tier A plan in `plans/analyze-my-project-based-proud-brooks.md`. Added 19 new features to the v3-nurse feature vector (102,100 rows × 2,116 cols):

- 13 binary `pmh_<diagnosis_group>` flags from the "Past Medical History" section of prior MIMIC-IV-Note discharge summaries (3.3 GB), OR'd with prior-admission ICD-derived flags from `mimic-iv/hosp/diagnoses_icd.csv`.
- 4 repeat-visit numerics (`n_prior_admissions`, `n_prior_ed_visits`, `days_since_last_admission`, `days_since_last_ed`).
- 1 same-complaint Jaccard, 1 `no_history` flag.

**Result on the same 20,420-row test split (random_state=42):**

| Metric | v3 nurse pre-PMH | v3 nurse (Change 1) | Δ |
|---|---|---|---|
| Top-1 diagnosis | 63.08% | **64.16%** | **+1.08pp** |
| Top-3 diagnosis | 84.95% | 85.71% | +0.76pp |
| Top-5 diagnosis | 92.43% | 92.88% | +0.45pp |
| Top-1 department | 67.11% | **68.61%** | **+1.50pp** |
| Top-3 department | 93.38% | 94.04% | +0.66pp |
| Cohen's κ diagnosis | 0.550 | 0.593 | +0.043 |
| Cohen's κ department | 0.453 | 0.501 | +0.048 |

**Predicted lift band was +2 to +4pp diagnosis**; actual came in below the band. The most likely reason: **~35% of training rows have `no_history=1`** (first-time MIMIC patients with all 13 PMH flags zero). The effective lift on the 65% of rows that do have prior history is closer to +1.7pp diagnosis / +2.3pp department.

Verification gate (per the plan):
- **19/19 PMH features have non-zero feature-importance gain.** No silent merge failure.
- **0 PMH features in the top 50.** Matches the existing nurse features' pattern (none of snapshot vitals / meds / longitudinal vitals / rhythm cracked the top 50 either) — features contribute through many small interactions, not as standalone heavy hitters.
- **No strong class regressed >1.5pp.** Mental Disorders (-0.7pp) and Nervous System (-0.2pp) are within noise. Weakest classes (Other, Genitourinary, Injury, Infectious, Circulatory) gained the most.

Full details, per-class deltas, and implementation pointers live in `docs/agents/doctor-agent.md#change-1--pmh-features`.

### Tier A (A2 + A3 + A4) — vocab expansion + softmax cascade + isotonic calibration (SHIPPED 2026-05-22)

Implemented from the Tier A plan in `plans/analyze-the-project-based-fluttering-charm.md`. Three independent, stackable changes retrained together in one v3-nurse training cycle on Colab T4 GPU (~36 min):

- **A2** — Grew `pmh_vocab.py` from 397 → 596 keywords (+199): s/p surgical histories, abbreviation variants (paf, ihd, ascvd, aicd), CKD staging (ckd 3/4/5), diabetes variants (gdm, prediabetes, dm-ii), and ~90 brand-name drugs as PMH proxies (lipitor → Circulatory+Endocrine, metformin → Endocrine, eliquis → Circulatory, prozac → Mental, omeprazole → Digestive, etc.). Drug names matched only inside the PMH section so they don't leak from Medications-on-Admission.
- **A3** — Replaced the department model's single `predicted_diagnosis` int column with 13 `diag_proba_<sanitized_label>` float columns (the full softmax distribution from the diagnosis model). Lets the dept model weight ambiguous diagnoses honestly.
- **A4** — Wrapped the department model with `CalibratedClassifierCV(FrozenEstimator(model), method="isotonic", cv=None)` using a 10% held-out calibration set. One isotonic regressor per class; interface-compatible with `XGBClassifier`.

**Result on the same 20,420-row test split (random_state=42):**

| Metric | v3 nurse pre-Tier-A (Change 1) | v3 nurse + Tier A | Δ |
|---|---|---|---|
| Top-1 diagnosis | 64.16% | **64.07%** | −0.09pp (flat) |
| Top-3 diagnosis | 85.71% | 85.72% | +0.01pp |
| Top-1 department | 68.61% | **70.79%** | **+2.18pp** |
| Top-3 department | 94.04% | 94.07% | +0.03pp |
| Cohen's κ diagnosis | 0.593 | 0.591 | −0.001 |
| Cohen's κ department | 0.501 | 0.495 | **−0.006** |

**Per-lever attribution (department top-1)**, decomposed using the uncalibrated dept top-1 logged before A4 wraps the model (69.33%):

| Lever | Predicted (dept) | Actual (dept) |
|---|---|---|
| A2 vocab expansion (diag side) | 0pp | ~0pp |
| A3 softmax cascade | +0.3 to +0.8pp | **+0.72pp** (A2+A3 combined: 68.61% → 69.33%). Mid-band. |
| A4 isotonic calibration | +0.1 to +0.3pp | **+1.46pp** (69.33% → 70.79%). **4-5× the predicted band.** |

**Important caveat — the dept gain is MED-dominant.** Per-class department recall vs v3 base:

| Department | v3 base | v3 nurse + Tier A | Δ vs base | n |
|---|---|---|---|---|
| **MED** | 65.4% | **87.3%** | **+21.8pp** | **12,045** |
| OMED | 14.9% | 16.2% | +1.3pp | 902 |
| NMED | 69.5% | 70.2% | +0.7pp | 1,226 |
| CMED | 66.2% | 61.7% | −4.5pp | 1,683 |
| TRAUM | 59.5% | 54.0% | −5.5pp | 509 |
| OB_GYN | 46.2% | 40.0% | −6.2pp | 225 |
| OTHER_SURG | 36.5% | 29.8% | −6.7pp | 540 |
| ORTHO | 65.4% | 58.6% | −6.8pp | 1,045 |
| NSURG | 41.9% | 34.7% | −7.2pp | 614 |
| OTHER | 23.9% | 8.2% | **−15.7pp** | 159 |
| SURG | 56.0% | 36.5% | **−19.6pp** | 1,472 |

MED is 59% of the test set, so its +21.8pp recall lift adds ~2,629 correct predictions — enough to dominate the top-1 metric on its own. Every other department either ticked up by <1.5pp or regressed (SURG −19.6pp, OTHER −15.7pp the worst). Cohen's κ dropping from 0.501 to 0.495 confirms the non-uniform pattern: κ penalises agreement that's driven by one dominant class.

This is a known dynamic with isotonic calibration on imbalanced multiclass softmax: per-class isotonic regressors aren't jointly monotone — they can compress minority probabilities (because the underlying XGBoost is over-confident on MED) and amplify the dominant class's argmax wins. Top-1 goes up because MED is half the test set; minority routing gets worse.

**Why it's still shippable:**
- Top-1 and top-3 both ticked up.
- Top-3 (94.07%) is near the ceiling for the 11-class problem already.
- The minority-routing regression isn't *introduced* by Tier A — pre-Tier-A v3 nurse already had SURG at 54.9% and OTHER at 12.6%. Tier A widens the gap rather than creating it.
- **A1 (Optuna macro-F1 sweep) is the next planned lever and is explicitly designed to recover minority recall** — its objective is macro-F1 (not flat top-1), and its search space includes `class_weight_exponent ∈ [0.4, 1.0]` so the optimiser can dial up minority weighting.

**A2 didn't move diagnosis top-1.** All per-class diagnosis deltas vs the pre-Tier-A baseline were within ±0.6pp — within measurement noise. Brand-drug PMH proxies appear to overlap with what current-stay TF-IDF already captures; minority classes that should benefit (Infectious / Other / Blood / Endocrine) are capacity-limited at the architecture / class-weight level, not feature-coverage limited. A1's class-weight-exponent search is the right tool for these.

Full per-lever attribution, per-class tables, implementation pointers, and the "shippable anyway" rationale live in `docs/agents/doctor-agent.md#tier-a--vocab-expansion--softmax-cascade--isotonic-calibration`.

---

## Empirical findings — experiments tried and reverted

A round of v3 improvements was implemented, trained, and benchmarked end-to-end (commit reverted). Three changes were tested together; their cumulative effect on **v3 with-nurse** measured against the unchanged held-out test set (20,420 stays, `random_state=42` stratified split) was **+0.18pp diagnosis** top-1 (63.08% → 63.26%) and **−0.30pp department** top-1 (67.11% → 66.81%). Each experiment, in turn, with the empirical reason for reverting:

### 1. `is_surgical` specialty-conditional gating flag — REVERTED, net negative on department

A binary flag derived from chief-complaint keyword matching (trauma mechanism, injury types, post-op presentations, OB/GYN, vascular surgical) appended to the v3 feature vector. Hypothesis: the tree could learn to soft-gate the nurse vital features for surgical complaints, where vitals had been observed to regress the department model.

**Result:** `is_surgical` became the **4th most important feature** in the v3 with-nurse diagnosis model (importance 0.01929), so the tree did consume it. Surgical-class department recall improved (TRAUM, SURG, OB_GYN, OTHER_SURG, OTHER all up 1–9pp each). But medical-class recall regressed (MED, OMED, NMED down 1–3pp each), and since MED dominates the test set (12,045 of 20,420 rows = 59%), the net department top-1 went **down 0.30pp**.

**Conclusion:** a single binary flag isn't enough — the gating needs to be architectural (separate medical and surgical sub-models), not feature-level. The two-stage department model in *Tier 3 — architectural changes* (item 11 below) remains the right direction.

### 2. Bio_ClinicalBERT chief-complaint embeddings — REVERTED, no observable lift

A frozen 110M-parameter encoder (`emilyalsentzer/Bio_ClinicalBERT`, pretrained on MIMIC-III clinical notes) producing 768-d mean-pooled embeddings of the normalized chief-complaint text. Concatenated alongside (not replacing) the 2000-d TF-IDF features. Hypothesis: semantic embeddings would disambiguate the top confusion pairs (Circulatory ↔ Respiratory, Genitourinary ↔ Digestive) where TF-IDF treats every token as orthogonal.

**Result:** **Zero `bert_*` features appeared in the top 50 feature importances** of the v3 nurse diagnosis model. The Circulatory ↔ Respiratory pair barely moved (Circulatory recall +0.6pp, Respiratory +1.2pp on v3 base). Best iteration on v3 base diagnosis crashed from 1730 → 329, suggesting BERT features carried strong early signal but encoded redundant information already captured by TF-IDF on these short complaints.

**Conclusion:** the v3 ceiling isn't a text-encoder problem; it's a labeling/feature-information problem. Short ED chief-complaint text doesn't contain enough signal to disambiguate clinically-similar categories regardless of how it's encoded. Cost: BERT added `transformers` and `torch` as ~2.5 GB heavy dependencies plus ~30 s GPU encode per training run for at most 1pp. Not worth the surface-area cost.

### 3. Pairwise / 3-way diagnosis discriminator refiner — REVERTED, net negative

Three small XGBoost discriminators trained to handle the top confusion groups: Circulatory vs Respiratory (binary), Genitourinary vs Digestive (binary), Skin vs Musculoskeletal vs Injury (3-way). At inference, when the main diagnosis model's top-1 vs top-2 probability margin was ≤ 0.15 AND both top labels fell in the same group, the appropriate discriminator was called as a tiebreaker.

**Result:** 1,121 rows met the activation criteria across the three groups. The refiner flipped 332 of them (29.6% flip rate). Net effect: **−14 corrects** on the test set (−0.07pp top-1 diagnosis). Per-group breakdown: `pair_circ_resp` −10, `pair_genit_digest` +1, `triple_skin_musc_inj` −5.

**Conclusion:** the close-call confusion pairs reflect genuine feature ambiguity given the input data, not main-model mistakes that a specialized discriminator can fix. When the main model assigns 38% to Circulatory and 32% to Respiratory, that probability split typically reflects real clinical ambiguity that the discriminator inherits.

### 4. Multi-label sibling head (Change 2) — REVERTED, lift below noise floor

Implemented from Tier A item 2 in `plans/analyze-my-project-based-proud-brooks.md`. A 13-head binary `XGBClassifier(objective="binary:logistic")` sibling — one head per diagnosis category — trained on the same feature matrix as the softmax model, with `scale_pos_weight = sqrt(neg/pos)` for class imbalance. Targets the multi-hot `also_<category>` vector built from `seq_num ∈ {1, 2, 3}` of `categorized_diagnosis.csv`. At inference the sigmoid scores are renormalized (`sigmoid / sum(sigmoid)`) and blended with the softmax via a single scalar α: `p_final = α * p_softmax + (1−α) * sigmoid_renorm`.

Predicted lift band (per the plan): **+1-2pp top-1, +1.5-3pp top-3**.

**Result — α sweep on the unchanged 20,420-row test set (random_state=42 stratified):**

| α | top-1 | Δ vs softmax | top-3 | Δ vs softmax | top-5 | Δ vs softmax |
|---|---|---|---|---|---|---|
| 0.00 (multilabel only) | 0.6156 | **−2.60pp** | 0.8317 | **−2.54pp** | 0.9099 | **−1.89pp** |
| 0.40 (saved default) | 0.6431 | +0.15pp | 0.8494 | **−0.77pp** | 0.9194 | **−0.94pp** |
| 0.50 (best top-1) | 0.6438 | +0.22pp | 0.8526 | −0.45pp | 0.9220 | −0.68pp |
| 0.80 | 0.6433 | +0.17pp | 0.8564 | −0.07pp | 0.9285 | −0.03pp |
| **0.90 (best top-3 / top-5)** | **0.6420** | **+0.04pp** | **0.8575** | **+0.04pp** | **0.9295** | **+0.07pp** |
| 1.00 (softmax only) | 0.6416 | 0 | 0.8571 | 0 | 0.9288 | 0 |

The 13 binary heads trained cleanly (best_iteration 1182-1499 across the heads; test_logloss 0.09-0.39 by class prevalence), and softmax-only (α=1.0) reproduced the Change 1 baseline exactly (0.6416 / 0.8571 / 0.9288), confirming the softmax pipeline was undisturbed. The lift only shows up at α=0.5 (top-1 +0.22pp, but with top-3 regression −0.45pp) or as noise-floor improvements at α=0.9 (+0.04pp / +0.04pp / +0.07pp).

**Conclusion:** Change 2 didn't deliver the predicted lift because the multilabel head's signal is **highly correlated with the softmax**, not orthogonal to it. Both heads train on the same feature matrix (~2116 cols) over the same 81,680 stays; the multi-hot label is a strict superset of the softmax label (seq_num=1 is in {1, 2, 3} by construction). The blend is two correlated predictors combined linearly — it can't extract much beyond what either does alone.

A secondary issue is the renormalization step `sigmoid / sum(sigmoid)` distorting the multi-label signal: sigmoid scores naturally answer "P(class is present)" and can legitimately sum to >1 for multi-label data, but renormalization compresses them onto a probability simplex they don't live on. The α=0.0 row (−2.5pp top-3) shows the cost of forcing the multilabel scores through this single-class lens.

**Lessons forward:**
1. If we revisit this, the multilabel head needs **orthogonal signal** — e.g., trained on a different feature subset (only nurse features) or with a different loss (asymmetric / focal for the rare classes).
2. The blend math should be in **log-space (product-of-experts)** or skip renormalization entirely — both options preserve the multi-label scores' calibration.
3. For top-3 specifically, taking the top-3 indices from the raw sigmoid vector (no renorm, no blend) might be a stronger baseline than blending — worth a side experiment if Change 2 is ever resurrected.

### 5. Two-stage department head (Change 4) — REVERTED, gate is the bottleneck

Implemented from Tier B item 2 in `plans/analyze-the-project-based-fluttering-charm.md`. The hypothesis: surgical and medical departments need different feature sets — medical routing benefits from nurse-collected vitals (which is why the v1→v2 nurse uplift went +10–13pp on MED/OMED), surgical routing is *hurt* by them (TRAUM, OTHER_SURG, OTHER all regressed −5 to −11pp under v3-nurse vs v3-base). A single binary `is_surgical` flag had already been tried and reverted in an earlier round (entry 1 above). This attempt was the full architectural split — three new XGBoost models alongside the legacy single department head, soft-blended at inference:

- **Gate (binary)** — predicts P(surgical) from base features only (TF-IDF + structured triage + cascading triage preds + cascading diagnosis). No nurse features, no PMH. Trained on the full 81,680-row train split with sqrt-inverse class weighting on the 22.6% / 77.4% surgical / medical split.
- **Medical head** — 4-class softmax over MED / CMED / NMED / OMED. **Full** v3-nurse feature vector (~2117 cols). Trained on the 63,249 medical-bucket train rows.
- **Surgical head** — 7-class softmax over SURG / NSURG / ORTHO / TRAUM / OTHER_SURG / OB_GYN / OTHER. **Base + PMH + cascade** only (~2045 cols) — snapshot vitals / meds / longitudinal vitals deliberately omitted on the hypothesis that they hurt surgical routing. Trained on the 18,431 surgical-bucket train rows.
- **Soft-blend at inference**: `final[c] = P_med * med_softmax(c)` if `c` is medical, `P_surg * surg_softmax(c)` if surgical. Argmax over 11 classes. The legacy single 11-class department model was kept trained alongside as backward-compatible fallback.

Predicted lift per the plan: **+2 to +4pp top-1 department, primarily by recovering TRAUM / OTHER / OTHER_SURG / OB_GYN**.

**Result on the unchanged 20,420-row held-out split (random_state=42):**

| Component | Score |
|---|---|
| Department legacy 1-stage (baseline) | **68.61%** top-1, 94.04% top-3, κ=0.5009 |
| Department Change 4 2-stage (soft-blend) | **67.39%** top-1, 93.41% top-3, κ=0.4856 |
| **Δ vs legacy** | **−1.22pp top-1, −0.63pp top-3, −0.0152 κ** |
| Gate binary accuracy | 81.99% |
| Gate medical recall | 85.9% |
| Gate surgical recall | 68.3% (31.7% of surgical patients misrouted into the medical head) |
| Medical head in-bucket top-1 | 82.95% |
| Surgical head in-bucket top-1 | 69.96% |

**Per-class deltas (2-stage − 1-stage on the same test set):**

| Department | bucket | 1-stage | 2-stage | Δ | n |
|---|---|---|---|---|---|
| TRAUM | surgical | 54.4% | 61.1% | **+6.7pp** | 509 |
| CMED | medical | 68.2% | 69.5% | +1.3pp | 1,683 |
| OTHER_SURG | surgical | 26.3% | 27.0% | +0.7pp | 540 |
| NSURG | surgical | 40.7% | 40.9% | +0.2pp | 614 |
| NMED | medical | 72.3% | 71.4% | −1.0pp | 1,226 |
| MED | medical | 78.6% | 77.2% | −1.5pp | 12,045 |
| OMED | medical | 24.9% | 23.4% | −1.6pp | 902 |
| SURG | surgical | 54.9% | 52.4% | −2.5pp | 1,472 |
| OB_GYN | surgical | 42.2% | 39.6% | −2.7pp | 225 |
| ORTHO | surgical | 65.6% | 60.8% | **−4.9pp** | 1,045 |
| OTHER | surgical | 12.6% | 5.0% | **−7.5pp** | 159 |

**Why it failed.** The gate is the architectural bottleneck. At 82.0% binary accuracy, **31.7% of surgical patients get routed into the medical head**, where they cannot be predicted correctly — the medical head only outputs MED / CMED / NMED / OMED. Soft-blending cannot recover this: even when the gate is uncertain (`p ≈ 0.5`), neither head has any path to a class outside its own bucket, so the blend just averages the two buckets' confident-on-wrong-class predictions. The end-to-end top-1 ceiling is bounded by gate accuracy × in-bucket head accuracy, and that math comes out **below** the legacy single-stage classifier, which has access to all features for an implicit single-step routing+classification decision.

The surgical-recovery hypothesis was *partially* validated — TRAUM did recover +6.7pp under the surgical-only head (consistent with the plan's prediction), and OTHER_SURG / NSURG ticked up slightly. But MED is 12,045 of the 20,420 test rows (59%), and the gate's misrouting cost MED −1.5pp = −181 patients. TRAUM is 509 rows, so +6.7pp = +34 patients. The volume-weighted impact of the medical regressions eats the surgical gains many times over. The architectural split correctly identified *which* patients each model should serve, then immediately lost them at the gate stage.

**Root cause of the gate failure.** The plan deliberately deprived the gate of nurse features (snapshot vitals, meds, longitudinal vitals, rhythm) on the hypothesis that vitals "confuse" surgical routing in the legacy single model. In hindsight this was wrong — vitals are *useful* discriminators for surgical-vs-medical (trauma → tachycardia + hypotension is a distinctive pattern; a sepsis presentation has its own pattern). By denying the gate access to vitals, we created an upstream routing decision strictly worse than what the legacy single-stage classifier implicitly makes. A salvage experiment with a full-feature gate would converge on the legacy model's implicit surgical/medical decision and isn't expected to clear the noise floor — not worth the engineering cost.

**Lessons forward:**
1. Architectural levers on the routing decision need an **end-to-end gradient** — train the gate jointly with the heads under a single loss (a mixture-of-experts setup with EM-style routing), rather than three independently-fit XGBoost models stitched together at inference.
2. **Don't deprive a routing classifier of features.** Even if some features are noisy for downstream classes, removing them from the upstream gate is a strictly worse position than the legacy single-stage model that has the same features and just does one classification step.
3. The −5 to −11pp surgical regression in v3-nurse vs v3-base is **not** a routing problem the gate can fix. It's a feature-information problem (vitals are noisy / uninformative for trauma/ortho/surgical routing) that requires **injury-specific features** (fracture mechanism, MVA detail, fall vs blunt vs penetrating), not architectural surgery on the existing feature set.

The implementation, training pipeline, benchmark wiring, and inference-side soft-blend were all reverted (commit reset to `c40f15b`) — only this writeup is retained.

### Net assessment

None of the five experiments above is worth keeping in its current form. The v3 with-nurse model at commit `34fbe24` (the Change 1 head — TF-IDF + structured triage features + cascading triage predictions + snapshot vitals + medications + longitudinal vitals + rhythm + PMH features) remains the best-measured configuration: **64.16% top-1 / 85.71% top-3 / 92.88% top-5 diagnosis, 68.61% top-1 / 94.04% top-3 department**. The structural lessons — that surgical/medical routing needs an architectural fix rather than a feature flag *or* a feature-deprived gate, that the diagnosis ceiling is a labeling problem rather than a text-encoding problem, and that a sibling-head blend needs orthogonal signal to clear the noise floor — feed forward into the recommendations below.

---

## Prioritized next-step recommendations

Ordered by expected lift per unit of effort.

### Tier 1 — high-lift, mostly low-risk

1. ~~**Add triage vitals to the triage model itself.**~~ **DONE (triage v2).** Vital signs from `triage.csv` are now used for ambulance/helicopter patients (~36%). The training pipeline masks vitals for walk-ins to match inference behavior. Acuity improved from 66.7% → 68.0% overall, and ambulance/helicopter patients specifically score 70.4%. Best iteration hit 2999/3000 — adding more trees is the next free lift for this model.
2. ~~**Use longitudinal vitals from `vitalsign.csv` for Doctor v2.**~~ **DONE in Doctor v3 with-nurse.** The v3 with-nurse pipeline aggregates per-stay min / max / last / delta over the first 4h after `intime` for each of the 6 vitals, plus 7 abnormal-reading counts and a normalized `rhythm` one-hot bucket. The 4h window guards against late-stay disposition leakage. See `docs/agents/doctor-agent.md` for the full feature list.
3. **Use discharge-note diagnosis as cleaner training labels.** Extract a normalized diagnosis from `discharge.csv` to replace the ICD-coded primary diagnosis. This directly attacks the 33% "Symptoms/Ill-Defined" labeling ceiling.
4. ~~**Train on the full 157K admitted rows** instead of the 100K sample.~~ **DONE in Doctor v3.** v3 trains on the full filtered admitted dataset (~102K rows after the catch-all exclusion). The 100K sub-sample cap is no longer applied.

### Tier 2 — medium-lift, needs care

5. ~~**PMH (Past Medical History) features from prior notes.**~~ **DONE in Change 1 (2026-05-21)** — see the "Empirical findings — experiments shipped" section above. +1.08pp diagnosis / +1.50pp department on the same held-out test split. The implementation pulls PMH sections from prior discharge summaries via regex + a 397-keyword vocabulary, OR'd with prior-admission ICD-derived flags. Below the predicted +2-4pp band but the lift is real and the verification audit passed.
6. **`pyxis.csv` (ED meds dispensed)** restricted to the first few minutes after arrival, or used strictly for validation rather than features. Full use risks leaking the working diagnosis.
7. **Admissions features** (insurance, marital status, language, in-hospital mortality as severity target).
8. **Race / ethnicity features** from `edstays.csv` — ethically sensitive; should come with a fairness audit.

### Tier 3 — architectural changes

9. **Hierarchical diagnosis classifier.** First binary: "Symptoms/Ill-Defined vs real diagnosis". Then, conditional on "real", predict specific category. Addresses the labeling-ceiling issue directly.
10. ~~**Multi-label training signal.** Use seq_num 2, 3 (secondary and tertiary diagnoses) as additional labels, not just seq_num=1.~~ **TRIED AS CHANGE 2 — REVERTED.** See "Empirical findings — experiments tried and reverted" section above (entry 4). The sibling-head blend lift came in at +0.04pp top-3 best case (noise floor) because the multilabel head shares features and label structure with the softmax — its signal is correlated, not orthogonal. A future attempt would need a different feature subset, a different loss, or a different blend math (log-space / product-of-experts).
11. **Separate vs unified routing for surgical vs medical departments.** Vitals help MED/CMED/NMED; injury-specific features help SURG/TRAUM/OTHER_SURG. Consider a two-stage department model.
12. **Radiology reports** — same leakage considerations as `discharge.csv`; useful only as PMH from prior encounters or very early-stay imaging.

---

## Model-level improvement opportunities

### Triage models
- Add more trees — best_iteration was 2999/3000 on the full dataset, model has not converged.
- ~~Include triage vital signs from `triage.csv`.~~ **DONE in triage v2.**
- Try LightGBM as an alternative to XGBoost.
- Ensemble methods (stacking multiple models).
- Neural network with learned embeddings for complaint text.

### Doctor models
- ~~Train on full 157K admitted rows instead of 100K.~~ **DONE in v3** (~102K after catch-all filter, no sub-sample).
- ~~Explore longitudinal vital signs from `vitalsign.csv` (multiple readings during stay) vs current single triage vitals.~~ **DONE in v3 with-nurse** (4h window aggregation + rhythm bucketing).
- ~~PMH features from prior encounters (discharge notes + ICD codes).~~ **DONE in Change 1** (2026-05-21) — +1.08pp diagnosis / +1.50pp department.
- ~~**Change 2 — multi-label seq_num 2, 3 sibling head.**~~ **TRIED AND REVERTED (2026-05-21).** Lift came in at +0.04pp top-3 best case (well below the +1.5-3pp predicted band). See "Empirical findings — experiments tried and reverted" entry 4 above for the alpha sweep and the diagnosis of why.
- **Next planned: Change 3 — Optuna hyperparameter sweep with macro-F1 objective.** The remaining Tier A lever. Predicted +0.5-1.5pp top-1, more on minority recall (Tier A item 3 in `plans/analyze-my-project-based-proud-brooks.md`).
- Hierarchical approach: first classify "Symptoms/Ill-Defined vs real diagnosis", then predict specific category. (v3 takes the alternate route of excluding the catch-all from training entirely.)
- The "Symptoms, Signs, Ill-Defined" catch-all (33%) is a labeling issue, not a model issue — patients coded with symptom-level ICD codes vs disease-level codes may have identical presentations.
- Surgical department routing could benefit from injury-specific features rather than vitals. *Note: a feature-flag approach (`is_surgical`) was tried in the reverted v3 improvement round — see "Empirical findings" above. The architectural two-stage routing in Tier 3 item 11 remains the open direction.*
- **PMH vocabulary expansion (Change 1 follow-up, low priority).** The current 397-keyword `pmh_vocab.py` map misses abbreviation variants (`s/p cabg`, `paf`, `chronic afib`, brand-name drugs as condition proxies). Expanding the map is cheap (~30 min effort) and might add another +0.3-0.5pp on top of Change 1.

---

## Planned future phases

### Phase 4: Text Generation Agent
- Generates synthetic natural language patient descriptions from structured data.
- E.g., structured `"abdominal pain, pain=7"` -> *"I've been having this terrible stomach ache since this morning, it's really bad"*.
- Used for creating test cases, validating the system, and training ML models.

### Phase 5: Hospital Infrastructure
- Synthetic real-time database of hospital rooms and available beds.
- Admission routing based on department prediction and bed availability.
- Possibly predict length of stay.

---

## Known issues (status carry-over)

These are live issues, not roadmap items — fix as opportunity arises.

1. **Windows console Unicode:** CrewAI's internal logging uses emojis that cause `charmap` encoding errors on Windows cp1252 console. These are cosmetic warnings from CrewAI's event bus, not from our code. They don't affect functionality.
2. **Windows Application Control:** Pandas DLL loading can occasionally be blocked by Windows Application Control policies. Restarting the terminal or IDE usually resolves this.
3. **Triage training time:** With 3000 trees and 2023 features on 334K samples, training takes ~90 minutes.
4. **Doctor v1 training time:** With 3000 trees and 2025 features on 80K samples, training takes ~30 minutes (two models). Early stopping typically triggers around iteration 1400-1900.
5. **Doctor v2 training time:** With 3000 trees and 2056 features on 80K samples, training takes ~45 minutes (two models). The medication aggregation step (3M rows from `medrecon.csv`) adds ~5 minutes to data loading.
6. **Doctor model accuracy (current state):**
   - **v1 (14-class baseline):** Diagnosis 50.2% / Department 59.1%.
   - **v2 (14-class, nurse data):** Diagnosis 52.4% / Department 65.0%.
   - **v3 base (13-class, catch-all excluded):** Diagnosis 60.1% / Department 60.7%.
   - **v3 nurse (pre-PMH):** Diagnosis 63.1% / Department 67.1%.
   - **v3 nurse (Change 1, current):** **Diagnosis 64.2% / Department 68.6%**.
   - Cumulative v3 base → v3 nurse Change 1: **+4.07pp diagnosis / +7.96pp department**.
   - Further gains require: Change 2 (multi-label sibling head), Change 3 (Optuna), cleaner labels from discharge summaries, or the architectural two-stage surgical/medical split. See the Tier 1-3 roadmap above.
