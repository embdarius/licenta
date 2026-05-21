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

### Net assessment

None of the three is worth keeping in its current form. The v3 with-nurse model at commit `0852cff` (TF-IDF + structured triage features + cascading triage predictions + snapshot vitals + medications + longitudinal vitals + rhythm) remains the best-measured configuration. The structural lessons — that surgical/medical routing needs an architectural fix rather than a feature flag, and that the diagnosis ceiling is a labeling problem rather than a text-encoding problem — feed forward into the recommendations below.

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
10. **Multi-label training signal.** Use seq_num 2, 3 (secondary and tertiary diagnoses) as additional labels, not just seq_num=1.
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
- **Next planned: Change 2 — multi-label seq_num 2, 3 sibling head.** Train a 13-d binary "also-this-category" head on the same feature matrix as the softmax model and blend logits at inference. Predicted: +1-2pp top-1, +1.5-3pp top-3 (Tier A item 2 in `plans/analyze-my-project-based-proud-brooks.md`).
- **Next planned: Change 3 — Optuna hyperparameter sweep with macro-F1 objective.** After Changes 1+2 land. Predicted +0.5-1.5pp top-1, more on minority recall.
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
