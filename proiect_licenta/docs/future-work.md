# Future Work and Roadmap

This document collects the project's open questions, diagnostic analyses, and prioritized next steps.

---

## Why were the Doctor v2 gains only modest? (+2.3pp diagnosis, +5.9pp department)

The v1 -> v2 upgrade added 31 new features (20 vital signs + 11 medications) yet the measured improvements were smaller than expected. Several reasons, in decreasing order of impact:

1. **Vitals are mostly normal for admitted patients.** By the time a patient is sick enough to be admitted but stable enough to be processed through triage, their vital signs tend to fall in a relatively narrow range. The vast majority of admitted patients are afebrile, normotensive, and with normal O2 saturation. The distinguishing signal is concentrated in the 5-10% of patients with flagged abnormalities (fever, tachycardia, hypoxia, hypotension) — which is exactly why those specific subgroups (e.g., Circulatory +5.2pp) saw most of the lift.

2. **Pre-admission medications indicate chronic conditions, not the acute presentation.** `medrecon.csv` captures what the patient was already taking when they arrived. A patient on metformin has diabetes; that's useful background but doesn't tell you *why they came to the ED today*. The medication features act more like demographic priors than diagnostic signals.

3. **TF-IDF dominates the top 50 feature importances.** Specific complaint language (e.g., "deliberate self harm", "collision", "sickle cell") remains the single strongest signal for both the diagnosis and department models. Nurse features do appear in the top 50 but rarely above the top 20.

4. **33% "Symptoms, Signs, Ill-Defined" catch-all is a labeling ceiling, not a feature problem.** When the ICD code for ~one third of admissions is literally a restatement of the chief complaint (e.g., "chest pain NOS"), no amount of extra input features will fix the fundamental ambiguity in the label space. This is the single largest structural ceiling on diagnosis accuracy.

5. **Surgical departments regressed.** Surgery routing depends on **what is broken** (injury type, fracture location) rather than on vitals. Adding vital-sign features slightly confused the department model for TRAUM / OTHER_SURG / SURG (-5 to -10pp each). A future improvement: route the doctor v2 features differently for surgical vs medical departments, or gate the use of vitals on the diagnosis category.

---

## Prioritized next-step recommendations

Ordered by expected lift per unit of effort.

### Tier 1 — high-lift, mostly low-risk

1. **Add triage vitals to the triage model itself.** The vitals are already in `triage.csv` and *are* recorded at triage time (no leakage). Triage acuity is inherently a physiological-plus-complaint decision; the current triage model uses only complaints. Expected lift: meaningful (vitals are the single largest input to clinical ESI scoring in practice).
2. **Use longitudinal vitals from `vitalsign.csv` for Doctor v2.** Replace the single triage snapshot with (min, max, last, trend) over the first N minutes of the stay. Include the `rhythm` categorical signal. Must be carefully time-windowed to avoid leakage — only readings before the decision point.
3. **Use discharge-note diagnosis as cleaner training labels.** Extract a normalized diagnosis from `discharge.csv` to replace the ICD-coded primary diagnosis. This directly attacks the 33% "Symptoms/Ill-Defined" labeling ceiling.
4. **Train on the full 157K admitted rows** instead of the 100K sample. Free lift.

### Tier 2 — medium-lift, needs care

5. **PMH (Past Medical History) features from prior notes.** For each encounter, pull the patient's earlier discharge summaries and extract PMH. Use those as features — not the *current* visit's discharge. This captures chronic conditions far better than `medrecon.csv` alone and has no leakage if restricted to prior encounters.
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
- Add more trees (models still weren't converging at 3000 — best_iteration near 2990).
- Include triage vital signs from `triage.csv` (see Tier 1 #1 above).
- Try LightGBM as an alternative to XGBoost.
- Ensemble methods (stacking multiple models).
- Neural network with learned embeddings for complaint text.

### Doctor models
- Train on full 157K admitted rows instead of 100K.
- Add second/third diagnoses (seq_num 2, 3) as multi-label training signal.
- Hierarchical approach: first classify "Symptoms/Ill-Defined vs real diagnosis", then predict specific category.
- Explore longitudinal vital signs from `vitalsign.csv` (multiple readings during stay) vs current single triage vitals.
- The "Symptoms, Signs, Ill-Defined" catch-all (33%) is a labeling issue, not a model issue — patients coded with symptom-level ICD codes vs disease-level codes may have identical presentations.
- Surgical department routing could benefit from injury-specific features rather than vitals.

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
6. **Doctor model accuracy:** Diagnosis accuracy improved from 50.2% (v1) to 52.4% (v2) with vital signs and medications. Department accuracy improved from 59.1% to 65.0%. Further gains likely require richer features (labs, imaging, longitudinal vitals) or a hierarchical classification approach (see Tier 3 above).
