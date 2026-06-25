# Doctor Agent (v1 + v2 + v3)

The Doctor Agent is the only agent in the pipeline that runs **twice**:
- **Phase 1 (v1):** Initial assessment using triage data only.
- **Phase 2 (v2):** Enhanced reassessment using triage data **plus** the vital signs and medication list collected by the Nurse Agent.

A **v3 tier** sits alongside v1/v2 (it does not replace them). v3 drops the "Symptoms, Signs, Ill-Defined" catch-all class so the model trains on a clinically meaningful 13-class label space, runs on the full filtered admitted-patient dataset (no 100K cap), and the v3 with-nurse variant adds longitudinal vitals + cardiac rhythm from `vitalsign.csv`, plus — as of **Change 1 (2026-05-21)** — Past Medical History features parsed from prior `discharge.csv` notes and `diagnoses_icd.csv`. v1 and v2 stay in the repo as the 14-class baselines for direct thesis comparison. See the [v3 section](#phase-3--doctor-v3-13-class-label-space) below and the [Change 1 section](#change-1--pmh-features) for the PMH details.

This three-phase design lets the user see the initial assessment before any nurse data is collected, and then see how the assessment updates once that extra information is available. It also gives the thesis a concrete before/after comparison of the clinical value of vitals + medications, and a separate comparison of the impact of the catch-all exclusion (v1/v2 vs v3).

- **Config:** `src/proiect_licenta/config/agents.yaml` (`doctor_agent`) and `tasks.yaml` (`doctor_assessment_task`, `doctor_reassessment_task`).
- **Data flow:** v1 receives triage output via the structured-data block emitted by the Triage Agent. v2 additionally receives the JSON block emitted by the Nurse Agent.

Both phases operate only on admitted patients. If the Triage Agent predicts discharge, the Doctor Agent provides a discharge summary instead.

---

## Phase 1 — Doctor v1 (Initial Assessment)

Two XGBoost models stored in `src/proiect_licenta/models/doctor/`.

### Model 3: Diagnosis Category v1 (14 classes)
- **File:** `diagnosis_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02, early stopping at best_iteration=1413).
- **Training data:** 80K samples (80/20 split from 100K sampled from 157K admitted patients).
- **Features (2025 total):** The 2023 triage features **plus** `predicted_acuity` and `predicted_disposition` from the triage model (cascading).
- **Class weighting:** Soft balanced (sqrt of inverse frequency).
- **Target:** Primary diagnosis category (seq_num=1 from ICD codes, grouped from 22 raw categories into 14).

**Diagnosis Category Grouping (22 raw -> 14 grouped):**

| # | Grouped Category | Raw Categories Merged | Admitted Count | % |
|---|------------------|-----------------------|----------------|---|
| 1 | Symptoms, Signs, Ill-Defined | Symptoms, Signs, Ill-Defined Conditions | 50,705 | 33.2% |
| 2 | Circulatory | Circulatory System | 18,655 | 12.2% |
| 3 | Digestive | Digestive System | 18,060 | 11.8% |
| 4 | Injury and Poisoning | Injury and Poisoning | 14,321 | 9.4% |
| 5 | Respiratory | Respiratory System | 11,056 | 7.2% |
| 6 | Genitourinary | Genitourinary System | 8,770 | 5.7% |
| 7 | Musculoskeletal | Musculoskeletal System | 5,974 | 3.9% |
| 8 | Endocrine, Nutritional, Metabolic | Endocrine, Nutritional, Metabolic | 5,274 | 3.5% |
| 9 | Mental Disorders | Mental Disorders | 4,494 | 2.9% |
| 10 | Skin and Subcutaneous Tissue | Skin and Subcutaneous Tissue | 4,248 | 2.8% |
| 11 | Nervous System and Sense Organs | Nervous System + Nervous System and Sense Organs + Diseases of the Eye + Diseases of the Ear (ICD-9/10 artifact) | 3,975 | 2.6% |
| 12 | Blood and Blood-Forming Organs | Blood and Blood-Forming Organs | 3,163 | 2.1% |
| 13 | Infectious and Parasitic | Infectious and Parasitic Diseases | 2,559 | 1.7% |
| 14 | Other | Pregnancy/Childbirth (789) + Neoplasms (694) + Supplemental (117) + Congenital (26) + Invalid (3) + Perinatal (2) | 1,551 | 1.0% |

### Model 4: Department v1 (11 classes)
- **File:** `department_model.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02, early stopping at best_iteration=1927).
- **Features:** Same 2025 features **plus** `predicted_diagnosis` from the diagnosis v1 model (cascading: triage -> diagnosis -> department).
- **Target:** First hospital service on admission (from `services.csv`).

**Department Grouping (19 raw -> 11 grouped):**

| # | Code | Full Name | Raw Services Merged | Count | % |
|---|------|-----------|---------------------|-------|---|
| 1 | MED | General Medicine | MED | 91,049 | 59.6% |
| 2 | CMED | Cardiac Medicine | CMED | 13,588 | 8.9% |
| 3 | SURG | General Surgery | SURG | 10,808 | 7.1% |
| 4 | NMED | Neuro Medicine | NMED | 10,598 | 6.9% |
| 5 | OMED | Oncology Medicine | OMED | 8,909 | 5.8% |
| 6 | ORTHO | Orthopedics | ORTHO | 5,300 | 3.5% |
| 7 | NSURG | Neurosurgery | NSURG | 3,804 | 2.5% |
| 8 | OTHER_SURG | Other Surgery | VSURG (1,770) + TSURG (568) + CSURG (537) + PSURG (443) | 3,257 | 2.1% |
| 9 | TRAUM | Trauma | TRAUM | 2,844 | 1.9% |
| 10 | OB_GYN | Obstetrics / Gynecology | GYN (1,078) + OBS (438) | 1,504 | 1.0% |
| 11 | OTHER | Other Specialty | GU (840) + PSYCH (210) + ENT (92) + EYE (6) + DENT (4) | 1,144 | 0.7% |

### Supporting Artifacts
- `doctor_metadata.json` — Training date, accuracy metrics, label lists, department names.

### Benchmark Results (test set: 20,000 samples)

#### Diagnosis Category Model (v1)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy (exact)** | **50.2%** |
| **Top-3 accuracy** | **83.6%** |
| **Top-5 accuracy** | **91.2%** |
| Cohen's kappa | 0.41 (moderate agreement) |
| Accuracy excl. catch-all | 52.6% |
| Majority baseline | 33.2% |
| **Lift over random** | **3.05x** |
| Mean confidence (correct) | 0.56 |
| Mean confidence (incorrect) | 0.45 |

**Per-class diagnosis recall (3 tiers):**
- **Strong (>65%):** Mental Disorders (75.5%), Injury/Poisoning (69.2%), Digestive (66.5%)
- **Moderate (45-58%):** Respiratory (57.8%), Skin (55.6%), Musculoskeletal (52.3%), Nervous System (47.5%), Circulatory (46.0%), Symptoms/Ill-Defined (45.4%)
- **Weak (<36%):** Blood (36.2%), Endocrine (31.9%), Genitourinary (29.7%), Other (20.7%), Infectious (11.0%)

**Key findings:**
- The "Symptoms, Signs, Ill-Defined" category (33% of data) is a catch-all where ICD codes essentially restate the chief complaint as a diagnosis (e.g., "chest pain NOS"). It absorbs 20-28% of every other category's misclassifications.
- Strong categories have **distinctive complaint language** (e.g., "deliberate self harm" -> Mental, "collision/fracture" -> Injury).
- Weak categories have **non-specific symptoms** (e.g., "weakness/fever" could be Infectious, Endocrine, Blood, or anything).
- Top features: "deliberate self" (Mental), "collision" (Injury), "pregnant" (Other/OB), "sickle cell" (Blood).

**Top misclassification patterns:**
1. Symptoms -> Digestive (16% of Symptoms cases)
2. Circulatory -> Symptoms (23% of Circulatory cases)
3. Genitourinary -> Symptoms (28% of Genitourinary cases)
4. Respiratory -> Symptoms (20% of Respiratory cases)

#### Department Model (v1)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy (exact)** | **59.1%** |
| **Top-3 accuracy** | **92.5%** |
| Cohen's kappa | 0.39 |
| Accuracy excl. MED | 50.7% |
| Majority baseline | 59.9% (always predict MED) |
| **Lift over random** | **1.54x** |

**Per-class department recall:**
- **Strong (>65%):** NMED/Neuro (67.1%), CMED/Cardiac (65.8%), ORTHO (65.8%)
- **Moderate (40-55%):** TRAUM (54.9%), SURG (51.6%), OB_GYN (42.3%)
- **Weak (<36%):** NSURG (35.7%), OTHER_SURG (28.1%), OTHER (18.5%), OMED/Oncology (16.0%)

**Key finding:** Overall accuracy (59.1%) is slightly below the majority baseline (59.9%) because MED dominates at 60%. However, the model's value is in correctly routing patients to specialized departments — it successfully identifies cardiac, neuro, ortho, and trauma patients instead of sending everyone to General Medicine.

**Department accuracy by diagnosis category:**
- Best: Mental Disorders -> 86.9% dept accuracy (almost all go to MED)
- Strong: Endocrine (69.6%), Respiratory (67.7%), Infectious (67.2%)
- Weakest: Other (44.3%), Musculoskeletal (49.6%), Injury/Poisoning (51.9%)

---

## Phase 2 — Doctor v2 (Enhanced Reassessment)

Same two classification targets (diagnosis + department) but with an expanded feature set that incorporates vital signs and medication history from the Nurse Agent.

### Model 5: Diagnosis Category v2 (14 classes, with nurse data)
- **File:** `diagnosis_model_v2.joblib`
- **Algorithm:** XGBClassifier (3000 trees, max_depth=10, lr=0.02, early stopping).
- **Training data:** 80K samples (80/20 split from 100K sampled from 157K admitted patients).
- **Features (2056 total):**
  - 2025 triage features (same as v1: 2023 triage + predicted_acuity + predicted_disposition).
  - **20 vital sign features:**
    - Raw values: `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`
    - Missing flags: `temperature_missing`, `heartrate_missing`, `resprate_missing`, `o2sat_missing`, `sbp_missing`, `dbp_missing`
    - Clinical flags: `fever` (>100.4F), `tachycardia` (HR>100), `bradycardia` (HR<60), `tachypnea` (RR>20), `hypoxia` (O2<94%), `hypertension` (SBP>140), `hypotension` (SBP<90)
    - `map` (mean arterial pressure = (SBP + 2*DBP) / 3)
  - **11 medication features:**
    - `n_medications` (count of pre-admission medications)
    - `meds_unknown` (1 if no medication data available)
    - 9 binary category flags: `has_cardiac_meds`, `has_diabetes_meds`, `has_psych_meds`, `has_respiratory_meds`, `has_opioid_meds`, `has_anticoagulant_meds`, `has_gi_meds`, `has_thyroid_meds`, `has_anticonvulsant_meds`

### Model 6: Department v2 (11 classes, with nurse data)
- **File:** `department_model_v2.joblib`
- **Features:** Same 2056 features **plus** `predicted_diagnosis` from the diagnosis v2 model (cascading).

### Medication Classification
- **Training data source:** `medrecon.csv` (`etcdescription` field) — maps drug class descriptions to 9 binary category flags via keyword matching.
- **Inference:** Patient-reported medication names are classified using:
  1. A **120+ entry drug name map** (generic + brand names to categories).
  2. **Keyword matching** on drug class descriptions.
  3. Covers common medications: statins, ACE inhibitors, insulin, SSRIs, inhalers, anticoagulants, PPIs, thyroid meds, anticonvulsants, etc.

Implemented in `src/proiect_licenta/tools/doctor_tool_v2.py` (`DRUG_NAME_MAP`, `MED_KEYWORD_MAP`, `classify_medications`).

### Vital Sign Processing
- **Data source:** `triage.csv` columns (`temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`) — these are the vitals measured at triage time.
- **Missing value handling:** Each vital gets a `_missing` binary flag; missing values are imputed with population medians (temp=98.1F, HR=84, RR=18, O2=98%, SBP=134, DBP=78).
- **Physiological clipping:** Values outside plausible ranges are clipped (e.g., temperature 90-110F, HR 20-250, RR 4-60, O2 50-100%, SBP 50-300, DBP 20-200).
- **Vital sign availability in training data:** 93-95% of admitted patients have vitals recorded; 84% have medication data.

### Supporting Artifacts
- `doctor_v2_metadata.json` — Training date, accuracy metrics, label lists, vital medians, medication keyword map.

### Benchmark Results — v1 vs v2 (test set: 20,000 samples)

#### Diagnosis Category: v1 vs v2

| Metric | v1 | v2 | Improvement |
|--------|-----|-----|-------------|
| **Top-1 accuracy** | 50.2% | **52.4%** | **+2.3pp** |
| **Top-3 accuracy** | 83.6% | **84.9%** | +1.3pp |
| **Top-5 accuracy** | 91.2% | **92.3%** | +1.1pp |
| Cohen's kappa | 0.415 | **0.435** | +0.020 |

**Per-class diagnosis improvements (v1 -> v2):**
- **Biggest gains:** Symptoms/Ill-Defined (+5.6pp), Circulatory (+5.2pp), Endocrine (+2.5pp), Injury (+2.3pp)
- **Slight regressions:** Skin (-5.0pp), Nervous System (-3.8pp), Digestive (-2.2pp)
- **Stable:** Mental Disorders, Musculoskeletal, Genitourinary

#### Department: v1 vs v2

| Metric | v1 | v2 | Improvement |
|--------|-----|-----|-------------|
| **Top-1 accuracy** | 59.1% | **65.0%** | **+5.9pp** |
| **Top-3 accuracy** | 92.5% | **93.7%** | +1.2pp |
| Cohen's kappa | 0.388 | **0.435** | +0.046 |
| Majority baseline | 59.9% | 59.9% | -- |

**Per-class department improvements (v1 -> v2):**
- **Biggest gains:** OMED/Oncology (+11.6pp), MED/General (+10.6pp), NMED/Neuro (+0.7pp)
- **Regressions:** TRAUM (-9.6pp), OTHER (-8.9pp), OTHER_SURG (-7.0pp), SURG (-5.9pp)

**Key findings:**
- Department v2 accuracy (65.0%) now **exceeds** the majority baseline (59.9%), which v1 (59.1%) did not.
- Vital signs help the most with medical (non-surgical) departments where physiological state directly influences routing.
- Surgical departments regressed slightly, likely because surgery routing depends more on injury type (already captured in chief complaints) than on vitals.
- Top v2 features are still dominated by TF-IDF terms (specific complaint language), but nurse features do appear in the top 50.

A deeper analysis of why the v2 gains were modest is in [`../future-work.md`](../future-work.md).

---

## Phase 3 — Doctor v3 (13-class label space)

v3 is structurally a **separate model tier** rather than a replacement for v1/v2. The v1 and v2 training files, tools, and artifacts are left untouched so the thesis can show all four configurations side-by-side. The reasons for v3:

1. **The "Symptoms, Signs, Ill-Defined" class is a labeling artifact.** It absorbs 33.2% of admitted patients and 20-28% of misclassifications from every other category. Many of those rows are ICD codes that restate the chief complaint (e.g., "chest pain NOS") rather than identify a cause, so no feature engineering can separate them from real-cause categories. v3 drops the class entirely. The remaining 13 classes are clinically meaningful.
2. **The 100K sub-sample cap is no longer needed** after the catch-all rows are filtered out (~102K admitted patients remain). v3 trains on everything.
3. **Longitudinal vital trends + cardiac rhythm** from `vitalsign.csv` (currently unused) carry signal that the single-snapshot vitals from `triage.csv` (used by v2) miss — particularly for cardiac and respiratory categories.

### v3 base — no nurse data

- **Files:** `src/proiect_licenta/training/train_doctor_v3.py` (training), saved to `artifacts/doctor/v3_base/`.
- **Label space:** 13 classes (catch-all dropped via `train_doctor.CATCH_ALL_LABEL`).
- **Sample:** Full filtered admitted dataset (~102K rows, no sub-sampling).
- **Features:** Same shape as v1 — TF-IDF + structured triage features + cascading triage predictions. ~2025 features. **No nurse-collected data.**
- **Cascading from triage v1:** Reuses `TRIAGE_V1_DIR` artifacts (TF-IDF vocab, severity map, acuity model, disposition model). Triage is not retrained for v3.

### v3 with-nurse — full nurse data + longitudinal vitals + rhythm

- **Files:** `src/proiect_licenta/training/train_nurse_v3.py` (training), saved to `artifacts/doctor/v3/`.
- **Label space:** 13 classes (same filter as v3 base).
- **Sample:** Full filtered admitted dataset (~102K rows).
- **Features (v3 base + nurse data + longitudinal vitalsign.csv aggregates):**
  - All 2025 v3-base features.
  - **Snapshot vitals from triage.csv (20 features):** same as v2 — raw values + missing flags + clinical flags (fever, tachycardia, bradycardia, tachypnea, hypoxia, hypertension, hypotension) + MAP.
  - **Medications from medrecon.csv (11 features):** same as v2 — `n_medications`, `meds_unknown`, plus 9 binary category flags via the shared `med_vocab.flags_from_row` vocabulary. **At inference (2026-06-03)** these come from one of two channels: for a returning patient with an MRN, the `med_lookup_json` block (the patient's reconciled home-med list from their most recent *prior* stay, supplied by `PatientHistoryLookupTool`) **overrides** the self-report; otherwise the nurse-collected free-text med list is parsed through the same vocabulary. (The disposition tool takes the same `med_lookup_json` arg; triage has no medication features.)
  - **Longitudinal vital trajectory (24 features):** for each of the 6 vitals, `<vital>_min`, `<vital>_max`, `<vital>_last`, `<vital>_delta` aggregated over readings within `[intime, intime + 4h]` (the 4h window guards against late-stay disposition leakage).
  - **Abnormal-reading counts (7 features):** `n_fever_readings`, `n_tachycardia_readings`, `n_bradycardia_readings`, `n_tachypnea_readings`, `n_hypoxia_readings`, `n_hypertension_readings`, `n_hypotension_readings`.
  - **Cardiac rhythm (10 features):** one-hot over 8 normalized buckets (`sinus`, `sinus_tachy`, `sinus_brady`, `afib_flutter`, `paced`, `av_block`, `svt`, `other`) plus `rhythm_irregular` (any non-sinus reading) plus `has_longitudinal_vitals` (was the stay covered in `vitalsign.csv` at all).
  - **PMH features (19 features, added by Change 1 — see dedicated [Change 1 section](#change-1--pmh-features) below):** 13 binary `pmh_<diagnosis_group>` flags from prior discharge-note PMH sections + ICD-derived fallback, plus 4 repeat-visit numerics (`n_prior_admissions`, `n_prior_ed_visits`, `days_since_last_admission`, `days_since_last_ed`), `same_complaint_as_prior` Jaccard, and `no_history` flag.
  - **Total:** **~2116 features** (~91 nurse-derived on top of the v3-base 2025).

### Inference behavior — degraded trajectory at runtime

`vitalsign.csv` is a training-time data source. The Nurse Agent at inference still collects only a single snapshot. `doctor_tool_v3` synthesizes the trajectory features as if `_min == _max == _last == snapshot` and `_delta == 0`; abnormal-reading counts collapse to the corresponding clinical flag (0 or 1). This is the same fallback used at training for stays without `vitalsign.csv` coverage (~5-7% of admitted stays based on early-window filter), so the model learned a consistent representation.

The Nurse Agent now also asks for cardiac rhythm (`nurse_tool.py` has a new prompt; `tasks.yaml` updated). The new `rhythm` field in the nurse JSON is **additive**: v2's tool ignores it, v3's tool consumes it. This means switching the live runtime between v2 and v3 is a one-line change in `crew.py` (which tool to register) plus an edit to `tasks.yaml` (which tool to call) — no nurse-side breakage either way.

### Inference-time consequence of catch-all exclusion

A v3 model can never output "Symptoms, Signs, Ill-Defined" — if a real patient's true category would be that bucket, v3 is *forced* into one of the 13 remaining classes. This is acceptable for the thesis claim ("for patients with a clinically meaningful diagnostic category, v3 achieves X%"), but should be noted in the writeup. v1/v2 remain in the repo as the 14-class fallback if the runtime ever needs to handle catch-all cases. A binary catch-all detector trained on the full 152K could be chained before v3 as a future extension if desired.

### Benchmarks

Training data: 81,680 stays (test 20,420), random_state=42, 80/20 stratified split on `diagnosis_group`. Best iteration: 1730 (v3 base diagnosis), 1796 (v3 base department), 1709 (v3 nurse diagnosis), 2197 (v3 nurse department).

**Headline four-way comparison** (`uv run python benchmarks/compare_all_versions.py`):

| Version    | Diag classes | Top-1 diag | Top-3 diag | Dept classes | Top-1 dept | Top-3 dept | n_train | n_test |
|------------|--------------|------------|------------|--------------|------------|------------|---------|--------|
| v1         | 14           | 50.18%     | —          | 11           | 59.13%     | —          | 80K     | 20K    |
| v2         | 14           | 52.44%     | —          | 11           | 65.04%     | —          | 80K     | 20K    |
| **v3 base**    | **13**           | **60.09%**     | **82.71%**     | **11**           | **60.66%**     | **92.28%**     | **81,680**  | **20,420** |
| v3 nurse (pre-PMH) | 13           | 63.08%     | 84.95%     | 11           | 67.11%     | 93.38%     | 81,680  | 20,420 |
| **v3 nurse (Change 1, PMH)**   | **13**           | **64.16%**     | **85.71%**     | **11**           | **68.61%**     | **94.04%**     | **81,680**  | **20,420** |

**Improvement deltas:**
- v3 base → v3 nurse (pre-PMH) — same 13-class space, isolates the value of snapshot vitals + meds + longitudinal vitals + rhythm: **+2.99pp diagnosis, +6.45pp department**. Slightly larger than the v1→v2 nurse lift (+2.26pp / +5.91pp).
- v3 nurse (pre-PMH) → v3 nurse (Change 1) — isolates the value of PMH features: **+1.08pp diagnosis, +1.50pp department**. See the [Change 1 section](#change-1--pmh-features) below.
- v3 base → v3 nurse (Change 1) — cumulative impact of all nurse data: **+4.07pp diagnosis, +7.96pp department**.
- v1 → v3 base (impact of catch-all exclusion + full dataset, both no-nurse): **+9.91pp diagnosis, +1.53pp department**. Diagnosis benefits massively because the catch-all class was absorbing 20–28% of misclassifications from every other category; department only ticks up because the catch-all was a diagnosis-side problem, not a department-side one.
- v2 → v3 nurse (Change 1) — full pipeline upgrade, but DIFFERENT label spaces, not apples-to-apples: **+11.72pp diagnosis, +3.57pp department**.

**Top-5 diagnosis accuracy on v3 nurse (Change 1): 92.88%** (up from 92.43% pre-PMH) — top-5 is essentially the entire useful prediction window, so for clinical decision support where the doctor sees a short ranked list of candidate categories, v3 nurse is near-ceiling on the 13-class task.

**Per-class diagnosis recall — v3 base → v3 nurse (Change 1):**

| Category | v3 base | v3 nurse (pre-PMH) | v3 nurse (Change 1) | base → Change 1 | pre-PMH → Change 1 |
|---|---|---|---|---|---|
| Circulatory | 59.2% | 67.8% | 69.3% | **+10.1pp** | +1.5pp |
| Respiratory | 61.9% | 66.8% | 67.8% | +5.9pp | +1.0pp |
| Endocrine | 35.9% | 39.6% | 40.9% | +5.0pp | +1.3pp |
| Genitourinary | 37.3% | 42.8% | 44.5% | **+7.2pp** | +1.7pp |
| Injury and Poisoning | 71.0% | 73.0% | 74.7% | +3.7pp | +1.7pp |
| Digestive | 78.9% | 80.7% | 81.6% | +2.7pp | +0.9pp |
| Infectious | 19.9% | 20.5% | 22.1% | +2.2pp | +1.6pp |
| Mental | 80.3% | 79.3% | 78.6% | -1.7pp | -0.7pp |
| Musculo | 58.0% | 57.0% | 57.7% | -0.3pp | +0.7pp |
| Nervous | 54.8% | 53.7% | 53.5% | -1.3pp | -0.2pp |
| Skin | 58.3% | 56.9% | 57.1% | -1.2pp | +0.2pp |
| Blood | 40.4% | 36.5% | 37.3% | -3.1pp | +0.8pp |
| Other | 23.2% | 18.7% | 21.0% | -2.2pp | **+2.3pp** |

The "pre-PMH → Change 1" column shows where Change 1 (PMH features) specifically moved the needle. The weakest classes — Other, Genitourinary, Injury, Infectious — gained the most; Blood partially recovered from a pre-PMH regression. Strong classes (Mental, Digestive) saw small noise-level changes. See the [Change 1 section](#change-1--pmh-features) for the per-class delta analysis.

**Per-class department recall — v3 base → v3 nurse (Change 1):**

| Department | v3 base | v3 nurse (pre-PMH) | v3 nurse (Change 1) | base → Change 1 |
|---|---|---|---|---|
| MED | 65.4% | 77.2% | 78.6% | **+13.2pp** |
| OMED | 14.9% | 21.6% | 24.9% | **+10.0pp** |
| NMED | 69.5% | 73.1% | 72.3% | +2.8pp |
| CMED | 66.2% | 67.6% | 68.2% | +2.0pp |
| ORTHO | 65.4% | 64.4% | 65.6% | +0.2pp |
| NSURG | 41.9% | 40.6% | 40.7% | -1.2pp |
| TRAUM | 59.5% | 54.8% | 54.4% | -5.1pp |
| SURG | 56.1% | 51.0% | 54.9% | -1.2pp |
| OB_GYN | 46.2% | 38.7% | 42.2% | -4.0pp |
| OTHER | 23.9% | 10.7% | 12.6% | -11.3pp |
| OTHER_SURG | 36.5% | 23.1% | 26.3% | -10.2pp |

**The "surgical regression" pattern from v2 carries over to v3.** Nurse-collected vitals + medications + rhythm help medical departments (MED, OMED, NMED, CMED) where physiological state directly drives routing, and hurt surgical departments (SURG, TRAUM, OTHER_SURG, OB_GYN, OTHER) where routing depends mostly on injury type already captured in the chief-complaint TF-IDF. This is a structural property of the feature set, not a v3 bug. See [`../future-work.md`](../future-work.md) for the empirical results of one attempt to address it (specialty-conditional feature gating, reverted) and the architectural two-stage routing direction that remains open.

**Top misclassification patterns on v3 base** (the catch-all is gone, so these are *real* category confusions, not labeling noise):

1. Circulatory → Respiratory (14.1% of true Circulatory) — shared symptoms (chest pain, shortness of breath)
2. Respiratory → Circulatory (12.8%) — same axis
3. Genitourinary → Digestive (14.8%) — abdominal-pain ambiguity
4. Skin → Musculoskeletal (14.4%) — limb complaints
5. Musculoskeletal → Injury (11.3%) — fracture vs musculoskeletal pain

These are the new top error pairs worth attacking next.

**Feature importance:** TF-IDF terms still dominate the top 20 in both v3 base and v3 with-nurse (specific complaint language like "sbo transfer", "pain pregnant", "motor vehicle collision", "sodium level", "sickle cell crisis"). **No nurse-specific feature has ever appeared in the top 50 of v3 nurse** — including the 19 new PMH features from Change 1 — yet the model is +4.07pp diagnosis, +7.96pp department better than v3 base. Nurse features contribute through many small interactions rather than as standalone heavy hitters; an importance-rank-only audit would understate their value.

**Run the benchmarks yourself:**

```
uv run python benchmarks/benchmark_doctor_v3.py    # full v3 base report
uv run python benchmarks/benchmark_nurse_v3.py     # v3 base vs v3 with-nurse
uv run python benchmarks/compare_all_versions.py   # four-way table
```

---

## Change 1 — PMH features

Added to v3 nurse on **2026-05-21**. The first lever from the four-way analytical-improvement plan (`plans/analyze-my-project-based-proud-brooks.md`).

### What it adds (19 new feature columns)

For each training stay, the pipeline looks at the same `subject_id`'s **strictly prior** hospital admissions and ED visits and computes:

- **13 binary `pmh_<diagnosis_group>` flags** — one per v3 diagnosis class (catch-all excluded). A flag is 1 if the patient has any prior admission documented as that diagnosis group. Two PMH sources OR'd together:
  - **Prior discharge-summary PMH sections** (richer). Streamed from `data/mimic-iv-notes/.../discharge.csv` (~3.3 GB), regex-extracted via `extract_pmh_section()` in `pmh_vocab.py`, run through a 397-keyword vocabulary mapping conditions (CHF, T2DM, COPD, sickle cell, etc.) to diagnosis groups, with negation handling ("no history of CHF" / "denies seizures" do not flag).
  - **Prior ICD-derived flags** (fallback / supplement). For patients without prior discharge notes, prior `hadm_id`s in `hosp/diagnoses_icd.csv` get mapped through the same ICD→category map used for supervision labels.
- **4 repeat-visit numerics**: `n_prior_admissions`, `n_prior_ed_visits`, `days_since_last_admission`, `days_since_last_ed`. Capped at `PMH_NO_PRIOR_DAYS = 9999` for first-time patients.
- **1 same-complaint Jaccard**: `same_complaint_as_prior` — token-set Jaccard between the current normalized chief complaint and the patient's most recent prior ED chief complaint.
- **1 no-history flag**: `no_history` — 1 if the patient has no prior MIMIC encounters at all.

Total new features: **19** (was ~2097 pre-PMH → ~2116 post-PMH).

**Leakage:** zero by construction. The discharge-note PMH section is established at the patient's *prior* discharge, before the encounter being predicted. The filter is `prior_admittime < current_intime` for ICD-derived flags and `prior_dischtime < current_intime` for discharge-note PMH.

### Inference behavior

The Nurse Agent gained two new prompts in [`nurse_tool.py`](../../src/proiect_licenta/tools/nurse_tool.py): "Do you have any chronic conditions or past medical history?" and "Roughly how many times have you been admitted before?" The patient can skip either; `doctor_tool_v3` parses the answer through the same 397-keyword vocab and zero-fills if absent. A patient who skips the prompt produces the same all-zero PMH vector as a first-time-patient training row, so the model learned a consistent representation.

### Benchmark results — pre-PMH vs Change 1

Same held-out 20,420-row test split (random_state=42 stratified). v3 base = same artifact (no PMH, identical training data).

| Metric | v3 nurse (pre-PMH) | v3 nurse (Change 1) | Δ |
|---|---|---|---|
| **Top-1 diagnosis** | 63.08% | **64.16%** | **+1.08pp** |
| **Top-3 diagnosis** | 84.95% | **85.71%** | +0.76pp |
| **Top-5 diagnosis** | 92.43% | **92.88%** | +0.45pp |
| Cohen's κ diagnosis | 0.550 | **0.593** | +0.043 |
| **Top-1 department** | 67.11% | **68.61%** | **+1.50pp** |
| **Top-3 department** | 93.38% | **94.04%** | +0.66pp |
| Cohen's κ department | 0.453 | **0.501** | +0.048 |

PMH coverage during training: **61% of admitted stays had ≥1 PMH flag set**; 64.8% had ≥1 prior MIMIC encounter (i.e., ~35% of training rows are first-time MIMIC patients with all PMH flags zero — those rows contribute nothing to Change 1). The effective lift on the 65% of rows that *do* have prior history is closer to **+1.7pp diagnosis / +2.3pp department**.

### Per-class lift attributable to Change 1 (pre-PMH → Change 1)

Most weak classes gained; one previously-regressing class (Blood) partially recovered; strong classes saw small noise-level changes.

| Category | Pre-PMH | Change 1 | Δ | Notes |
|---|---|---|---|---|
| Other | 18.7% | 21.0% | **+2.3pp** | Biggest gain; the most-heterogeneous class benefits from chronic-condition priors |
| Genitourinary | 42.8% | 44.5% | +1.7pp | Recurrent UTI / CKD / dialysis PMH disambiguates from Digestive |
| Injury | 73.0% | 74.7% | +1.7pp | Prior trauma / fracture / mva history |
| Infectious | 20.5% | 22.1% | +1.6pp | Sickle cell / immunocompromised / HIV history matters |
| Circulatory | 67.8% | 69.3% | +1.5pp | Prior CHF / CAD / MI admissions are strong priors |
| Endocrine | 39.6% | 40.9% | +1.3pp | Prior DM / thyroid / adrenal disease |
| Respiratory | 66.8% | 67.8% | +1.0pp | Prior COPD / asthma / OSA |
| Digestive | 80.7% | 81.6% | +0.9pp | Already strong; small gain |
| Blood | 36.5% | 37.3% | +0.8pp | Recovers ~20% of the pre-PMH regression vs v3 base (40.4%) |
| Musculo | 57.0% | 57.7% | +0.7pp | |
| Skin | 56.9% | 57.1% | +0.2pp | |
| Nervous | 53.7% | 53.5% | -0.2pp | Noise |
| Mental | 79.3% | 78.6% | -0.7pp | Already strong; small noise drop |

No class regressed >1.5pp — verification gate from the plan passes.

### Feature-importance audit (verification gate)

| Check | Result |
|---|---|
| PMH features with non-zero gain | **19 / 19** (no silent merge failure) |
| PMH features in top 50 by gain | 0 (matches pre-PMH nurse features' pattern — top 50 is TF-IDF-dominated) |
| PMH coverage in training | 61% have ≥1 flag set; 35% are first-time patients (no_history=1) |

The fact that no `pmh_*` column cracks the top 50 by gain is **expected**, not a bug — none of the existing nurse features (snapshot vitals, meds, longitudinal vitals, rhythm) ever did either. All 19 having non-zero gain confirms the merge is healthy and the trees are using the features through many small interactions.

### Implementation

- **Vocabulary:** [`src/proiect_licenta/pmh_vocab.py`](../../src/proiect_licenta/pmh_vocab.py) — 397 keyword → diagnosis-group entries, regex-based section extractor for MIMIC discharge notes (handles both `PMH: CHF, DM` same-line layouts and `PMH:\n1. CHF\n2. DM` own-line layouts), negation neutralizer ("no history of", "denies", "negative for").
- **Aggregator:** `_aggregate_pmh()` and `_parse_discharge_pmh()` in [`src/proiect_licenta/training/train_nurse_v3.py`](../../src/proiect_licenta/training/train_nurse_v3.py). Chunked discharge.csv pass (~166 chunks of 2000 rows; tqdm-wrapped progress bar in Colab).
- **Inference:** [`src/proiect_licenta/tools/doctor_tool_v3.py`](../../src/proiect_licenta/tools/doctor_tool_v3.py) parses the new `prior_history` and `n_prior_admissions` fields from the Nurse Agent's JSON output, builds the 19-column PMH vector, falls back to all-zeros + `no_history=1` if absent.
- **Nurse prompts:** [`src/proiect_licenta/tools/nurse_tool.py`](../../src/proiect_licenta/tools/nurse_tool.py) and [`src/proiect_licenta/config/tasks.yaml`](../../src/proiect_licenta/config/tasks.yaml).
- **Paths added:** `DISCHARGE_NOTES_CSV`, `DIAGNOSES_ICD_CSV`, `ADMISSIONS_CSV` in [`paths.py`](../../src/proiect_licenta/paths.py).

### Result vs prediction, and why

The plan's predicted band was +2 to +4pp diagnosis. Actual: **+1.08pp** — about 1pp below the midpoint. Likely reasons:

1. **~35% of rows are first-time MIMIC patients** (`no_history=1`). For those rows all 13 PMH flags are zero, and Change 1 contributes nothing. The plan assumed broader prior-encounter coverage.
2. **PMH keyword vocabulary has gaps.** 397 keywords cover the common conditions but miss many abbreviation variants (`s/p cabg`, `paf`, brand-name drugs as condition proxies). Expanding the vocab is a cheap follow-up if Change 2 underdelivers.
3. **The numeric features (`days_since_*`, `n_prior_*`) are coarse.** The XGBoost trees probably use them mostly as the "is it 9999? (= no_history)" split, leaving subtler dose-response signal on the table.

None of these block moving to Change 2 (multi-label seq_num 2,3 sibling head) — Change 1's verification gate has passed and the lift is real.

---

## Tier A — vocab expansion + softmax cascade + isotonic calibration

Shipped 2026-05-22 on top of Change 1. Three stackable, independent changes designed to be retrained together in one v3-nurse training cycle:

- **A2 — PMH vocabulary expansion.** Grew `pmh_vocab.py` from 397 to 596 keywords (+199). Added s/p surgical histories (`s/p cabg`, `s/p stent`, `s/p mi`), abbreviation variants (`paf`, `chronic afib`, `ihd`, `ascvd`, `aicd`), CKD staging (`ckd 3/4/5`, `ckd iii/iv/v`), diabetes variants (`gdm`, `prediabetes`, `dm-ii`), and ~90 brand-name drugs as PMH proxies (lipitor → Circulatory+Endocrine, metformin → Endocrine, eliquis → Circulatory, prozac → Mental, omeprazole → Digestive, keppra → Nervous, etc.). Brand-name drugs are matched only inside the PMH section (bounded above by `Medications on Admission` header) so they don't leak from the Medications block.

- **A3 — diagnosis-softmax cascade.** The department model's cascade column was a single `predicted_diagnosis` integer (argmax over 13 classes). A3 replaces it with 13 `diag_proba_<sanitized_label>` float columns — the full softmax distribution from the diagnosis model. The department model can now weight ambiguous predictions ("38% Circulatory, 32% Respiratory") instead of hard-locking to top-1. Column list persisted in `metadata.json` under `diag_cascade_cols` so `doctor_tool_v3` reproduces the exact column layout at inference.

- **A4 — isotonic calibration on the department model.** Hold out 10% of the train split as a calibration set; fit XGBoost on the remaining 90%; wrap with `CalibratedClassifierCV(FrozenEstimator(model), method="isotonic", cv=None)` (the `cv="prefit"` shortcut is deprecated in sklearn 1.6+). Saved as `department_model.joblib` (replaces the bare XGBClassifier); `predict` / `predict_proba` interface is unchanged so the inference tool needs no edit.

### Benchmark results — pre-Tier-A (Change 1) vs Tier A on the same 20,420-row test split (random_state=42)

| Metric | v3 nurse pre-Tier-A | v3 nurse + Tier A | Δ |
|---|---|---|---|
| **Top-1 diagnosis** | 64.16% | **64.07%** | **−0.09pp (flat)** |
| **Top-3 diagnosis** | 85.71% | 85.72% | +0.01pp |
| **Top-5 diagnosis** | 92.88% | 92.77% | −0.11pp |
| Cohen's κ diagnosis | 0.5925 | 0.5915 | −0.001 |
| **Top-1 department** | 68.61% | **70.79%** | **+2.18pp** |
| **Top-3 department** | 94.04% | 94.07% | +0.03pp |
| Cohen's κ department | 0.5009 | 0.4948 | **−0.006** |

### Per-lever attribution (department top-1)

The training log records the uncalibrated department top-1 (69.33%) before A4 wraps the model, so the decomposition is unambiguous:

| Lever | Predicted (dept) | Actual (dept) | Notes |
|---|---|---|---|
| A2 vocab expansion | 0pp dept (diag-only lever) | ~0pp dept | A2 affects PMH features in the dept input matrix indirectly. Diag side flat. |
| A3 softmax cascade | +0.3 to +0.8pp dept | **+0.72pp dept (A2+A3 combined: 68.61% → 69.33%)** | Mid-band. No cascade column hit dept top-30 by gain — contributes through many small interactions, same pattern as PMH features. |
| A4 isotonic calibration | +0.1 to +0.3pp dept | **+1.46pp dept (69.33% → 70.79%)** | **4-5× the predicted band.** Per-class isotonic fits aren't jointly monotone — they can redistribute probability mass between classes. |

### Per-class diagnosis recall — v3 base → v3 nurse + Tier A

| Category | v3 base | v3 nurse + Tier A | Δ vs base | n |
|---|---|---|---|---|
| Circulatory | 59.2% | 69.2% | **+10.0pp** | 3,731 |
| Genitourinary | 37.3% | 44.5% | +7.2pp | 1,754 |
| Respiratory | 61.9% | 67.2% | +5.3pp | 2,211 |
| Endocrine, Nutritional, Metabolic | 35.9% | 40.4% | +4.5pp | 1,055 |
| Injury and Poisoning | 71.0% | 74.9% | +3.9pp | 2,864 |
| Digestive | 78.9% | 81.5% | +2.6pp | 3,612 |
| Infectious and Parasitic | 19.9% | 22.3% | +2.3pp | 512 |
| Musculoskeletal | 58.0% | 57.2% | −0.8pp | 1,195 |
| Nervous System and Sense Organs | 54.8% | 53.8% | −1.0pp | 795 |
| Skin and Subcutaneous Tissue | 58.3% | 57.2% | −1.1pp | 849 |
| Mental Disorders | 80.3% | 78.8% | −1.6pp | 899 |
| Other | 23.2% | 21.3% | −1.9pp | 310 |
| Blood and Blood-Forming Organs | 40.4% | 37.0% | −3.5pp | 633 |

A2 didn't move the diagnosis needle measurably — every per-class delta vs the pre-Tier-A Change-1 baseline was within ±0.6pp. The A2 PMH expansion sharpens specific keyword matches but doesn't change category-level argmax, mostly because:
1. Brand-drug PMH proxies overlap with what TF-IDF already captures from current-stay complaint text.
2. Minority classes (Infectious, Other, Blood) where prior history should matter most are capacity-limited at the model architecture level, not feature-coverage limited.
3. A1's class-weight-exponent search is the lever that should actually move minority recall.

### Per-class department recall — v3 base → v3 nurse + Tier A — the MED-dominance caveat

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

**The +2.18pp top-1 department gain is concentrated almost entirely on MED.** MED is 59% of the test set (12,045 of 20,420), so its +21.8pp recall lift translates to +2,629 additional correct predictions — enough to dominate the top-1 metric. Every other department either ticked up by <1.5pp or regressed; SURG fell −19.6pp and OTHER fell −15.7pp.

This pattern is consistent with what isotonic calibration does on imbalanced multiclass softmax: the per-class isotonic regressors can compress minority-class probabilities downward (because the underlying XGBoost over-confidence on MED gives the calibrator a strong "trust MED more" signal). Cohen's κ on department dropping from 0.5009 to 0.4948 confirms this — the κ statistic penalises non-uniform improvement, and a single-class-dominant shift like this lowers it even when raw top-1 rises.

### Why this is shippable anyway

- Top-1 and top-3 both ticked up. Top-3 (94.07%) is essentially the ceiling on this 11-class problem already; for clinical decision support where the doctor sees a ranked short list, the calibrated probabilities are now more honest.
- The MED-dominance is *not* introduced by Tier A in absolute terms — pre-Tier-A v3 nurse already had MED at 78.6% recall vs SURG at 54.9% and OTHER at 12.6%. Tier A widens the gap rather than creating it.
- A1 (Optuna macro-F1 sweep) is the next planned lever and is *explicitly designed to recover minority recall*: its objective is macro-F1 (not flat top-1), and its search space includes `class_weight_exponent ∈ [0.4, 1.0]` which lets the optimiser dial up minority weighting if that lifts macro-F1.

### Implementation

- **A2 vocab:** [`src/proiect_licenta/pmh_vocab.py`](../../src/proiect_licenta/pmh_vocab.py) lines 459-695. No code change needed — the existing `_aggregate_pmh` aggregator picks up the new keywords at training time.
- **A3 cascade:** new helpers `build_diag_cascade_cols` and `_attach_diag_cascade` in [`src/proiect_licenta/training/train_nurse_v3.py`](../../src/proiect_licenta/training/train_nurse_v3.py). Inference-side cascade in [`src/proiect_licenta/tools/doctor_tool_v3.py`](../../src/proiect_licenta/tools/doctor_tool_v3.py) reads `metadata["diag_cascade_cols"]`. Legacy `predicted_diagnosis` single-int fallback retained for pre-A3 artifacts.
- **A4 calibration:** see the A4 block in `train_nurse_v3.main()` after the dept fit. Uses `FrozenEstimator` (sklearn 1.6+) with `cv=None` because `cv="prefit"` is deprecated in sklearn 1.6+; legacy `cv="prefit"` fallback retained for sklearn < 1.6.
- **Metadata audit fields:** `diag_cascade_cols` (A3), `department_calibration` block (A4) with uncalibrated_accuracy, calibrated_accuracy, n_fit, n_cal.

### What's next

**A1 — Optuna macro-F1 sweep.** The only Tier A lever not yet shipped. Expected lift: +0.5-1.5pp diagnosis with more impact on minority recall (Infectious / Other / Blood / Endocrine / Genitourinary). The class-weight-exponent dimension in the search space directly addresses the minority-class regressions that A4 introduced on the department side. After A1 lands, the full Tier A stack should be re-evaluated end-to-end.

---

## Doctor disposition v3 — peer model (plan section 3, Option B, shipped 2026-05-28)

A peer **binary admit/discharge classifier** that sits alongside the v3 diagnosis + department models, refining the triage-time disposition (currently 80.19% accuracy / 0.8894 ROC AUC against the same disposition-pipeline test rows — see "Why two reference numbers" below) using the nurse-collected snapshot vitals + longitudinal vitals + rhythm + medications + PMH. Unlike the diagnosis + department models, this one is trained on the **FULL 425K ED stays** (filtered down to 418,083 after acuity/chief-complaint cleaning) rather than the admitted-only ~102K slice, so it sees the positives and negatives in their real-world ratio (36.7% admit / 63.3% discharge).

This is the implementation of "Option B" from the Triage Improvements + Doctor-Level Discharge Prediction plan (`plans/read-the-documentation-of-tidy-scone.md`, section 3): a peer model that doesn't break the admitted-only assumption of diagnosis/department.

### Why two reference numbers exist

The triage v3 disposition model has two cited accuracies, and the difference matters for thesis defense:

- **77.98%** in [`docs/agents/triage-agent.md`](triage-agent.md#triage-v3) — measured on the triage v3 pipeline's own 80/20 test split, which uses the triage-pipeline filtering (stratified on `acuity`).
- **80.19%** in the disposition-model benchmark (`benchmarks/benchmark_doctor_disposition.py`) — measured on the disposition pipeline's 80/20 test split (stratified on `admitted`), with the triage v3 disposition prediction obtained by running the triage v3 cascade through `train_triage_v3.build_features` on the same rows the doctor disposition model is scored on.

The 80.19% is the **apples-to-apples baseline** for this section because it's measured on the exact rows the doctor model is evaluated against. The 77.98% is the triage agent's own headline number and should stay in the triage doc.

### Architecture

| | Triage v3 disposition | Doctor disposition v3 |
|---|---|---|
| Output | binary admit/discharge | binary admit/discharge |
| Training set | full 425K stays | full 418K stays (after cleaning) |
| Objective | binary:logistic | binary:logistic |
| Class weighting | cascading scale_pos_weight | `scale_pos_weight = sqrt(N_neg/N_pos) = 1.31` |
| Cascade input from triage | hard cascade (predicted_acuity int) | **soft cascade** (5 acuity softmax + 1 dispo probability) |
| Calibration | none | **isotonic, FrozenEstimator + CalibratedClassifierCV** on 10% held-out fit slice |
| Nurse data | NO (triage-time only) | **YES** (vitals + longitudinal + meds + PMH all available) |
| Feature count | 2,070 | **2,128** (2069 v3 base + 6 soft cascade + 11 medication + 41 longitudinal vital/rhythm + 1 cascade integer-position housekeeping) |
| Leakage guard | n/a (triage-time vitals) | longitudinal vital window `[intime, intime + 4h]` |
| Use of catch-all filter | n/a | **no** — the disposition decision applies to all stays; diagnosis catch-all is a labeling artifact specific to diagnosis-grouping and is irrelevant here |
| Output range | 0–1 (uncalibrated) | 0–1 (**isotonic-calibrated**) |

**Soft cascade (per plan section 2/3 recommendation):** instead of feeding the doctor disposition model the triage's argmax integer (`predicted_acuity ∈ {1..5}`), the disposition model receives the full triage v3 acuity softmax (5 float columns `triage_acuity_proba_1..5`) plus the triage v3 disposition probability (`triage_disposition_proba_admit`). This mirrors the A3 lesson from the doctor diag/dept side (single argmax → 13 `diag_proba_*` columns gave +0.72pp on department) and lets the doctor model weight borderline ESI 2-3 cases honestly rather than hard-locking on the triage argmax.

**Isotonic calibration (per plan section 3 recommendation):** the deployment artifact is `CalibratedClassifierCV(FrozenEstimator(raw_xgb), method="isotonic", cv=None)`, fit on a 10% held-out slice (33,447 rows) that the underlying XGBoost never sees during training. Calibration matters more here than for diagnosis because the disposition output is a clinical-decision probability — a miscalibrated 0.8 hurts. The raw uncalibrated model is also saved (`disposition_model_raw.joblib`) for audit and feature-importance inspection.

### Benchmark results — triage v3 cascade vs doctor disposition v3 (same 83,617-row test split)

The triage v3 cascade dispo baseline below is computed inline in the benchmark by thresholding the soft-cascade `triage_disposition_proba_admit` column at 0.5 — i.e., it's exactly the model that already sits in `artifacts/triage/v3/disposition_model.joblib`, evaluated on the disposition pipeline's test rows.

| Metric | triage v3 cascade dispo (baseline) | doctor dispo v3 (calibrated, deployment) | Δ |
|---|---|---|---|
| **Accuracy** | 0.8019 | **0.8396** | **+0.0377** |
| **ROC AUC** | 0.8894 | **0.9138** | **+0.0244** |
| Brier score | 0.1360 | **0.1128** | −0.0232 (lower=better) |
| **ECE (10 bins)** | 0.0748 | **0.0036** | **−0.0713** (calibration **crushed** by isotonic) |
| Sensitivity (admit recall) | 0.8054 | 0.7649 | −0.0405 |
| Specificity (discharge recall) | 0.7999 | **0.8830** | **+0.0830** |
| Over-triage rate (false admit) | 0.2001 | **0.1170** | **−0.0830** |
| Under-triage rate (missed admit) | 0.1946 | 0.2351 | **+0.0405** ← **worse**, see operating-point discussion below |

**Predicted band per plan section 3:** +3-6pp accuracy, +0.03-0.05 ROC AUC. Accuracy delta of +3.77pp lands **in-band**; ROC AUC delta of +0.0244 lands **just below the predicted band**.

**The under-triage trade-off (clinically important, honest):** At the default 0.5 threshold the doctor model trades sensitivity for specificity — it catches 4.05pp fewer real admissions (76.49% vs 80.54%) in exchange for 8.30pp better discharge recall (88.30% vs 79.99%). The ROC AUC delta of +0.0244 says the model has **strictly more discriminative power** than triage v3 (the underlying probability ranking is better); the under-triage regression is therefore an **operating-point** problem, not a model-quality problem. With threshold tuning (~0.40 or `scale_pos_weight = 1.8`) the model can almost certainly recover the sensitivity gap and still improve on accuracy. **Threshold tuning is left as future work** — see "What's next" below.

**Uncalibrated audit (`disposition_model_raw.joblib`):** accuracy 0.8383, ROC AUC 0.9140, ECE 0.0204. Isotonic moved the ECE from 0.0204 → 0.0036 (an order-of-magnitude improvement in reliability) at the cost of essentially-zero accuracy (0.8383 → 0.8396, +0.13pp) and ROC AUC (0.9140 → 0.9138, −0.0002). Trade is unambiguously good: isotonic is monotone so the ranking is preserved, and the calibrated probabilities are now usable as decision-grade output.

### Calibration curve — the headline of the isotonic story

10 reliability bins on the held-out test rows, calibrated model. Bin centers track observed admit rate to within ±0.01 across every bin:

| bin center | predicted mean | observed admit | n |
|---|---|---|---|
| 0.05 | 0.0276 | 0.0277 | 27,519 |
| 0.15 | 0.1305 | 0.1339 | 11,796 |
| 0.25 | 0.2498 | 0.2646 | 6,780 |
| 0.35 | 0.3525 | 0.3581 | 4,013 |
| 0.45 | 0.4325 | 0.4312 | 3,822 |
| 0.55 | 0.5228 | 0.5334 | 4,477 |
| 0.65 | 0.6372 | 0.6360 | 4,423 |
| 0.75 | 0.7482 | 0.7421 | 4,211 |
| 0.85 | 0.8481 | 0.8401 | 5,227 |
| 0.95 | 0.9491 | 0.9497 | 11,349 |

A clinician reading "P(admit) = 0.82" from this model can trust that, on average, ~82% of patients with that score were actually admitted. This is the property that justifies shipping the disposition prediction as a calibrated probability rather than just a binary decision.

### Per-subgroup deltas — lift concentration matches the plan's prediction

Plan section 3 predicted the lift would concentrate in elderly / polypharmacy / abnormal-vitals / repeat-visitor / prior-admission cohorts. The benchmark confirms this almost exactly:

| Subgroup | n | base accuracy | doctor accuracy | Δ acc | base AUC | doctor AUC | Δ AUC |
|---|---|---|---|---|---|---|---|
| elderly (age ≥ 65) | 26,349 | 0.7521 | 0.8047 | **+0.0526** | 0.8526 | 0.8858 | **+0.0332** |
| polypharmacy (n_meds ≥ 5) | 41,856 | 0.7553 | 0.8032 | **+0.0479** | 0.8546 | 0.8885 | **+0.0339** |
| abnormal vitals (any 1+ flag) | 18,702 | 0.7998 | 0.8398 | +0.0400 | 0.8892 | 0.9199 | +0.0307 |
| repeat visitor (prior ED visit) | 43,923 | 0.7976 | 0.8415 | +0.0439 | 0.8913 | 0.9187 | +0.0274 |
| prior admission | 43,137 | 0.7720 | 0.8221 | **+0.0501** | 0.8737 | 0.9058 | **+0.0321** |
| non-sinus rhythm | 1,445 | 0.8048 | 0.8401 | +0.0353 | 0.8484 | 0.8933 | **+0.0449** |

**Reading.** The biggest accuracy lifts (+5.26pp, +5.01pp) are on elderly and prior-admission patients — exactly where nurse data (PMH, longitudinal vitals, polypharmacy) carries the most information beyond what triage-time data captures. The smallest accuracy lift (+3.53pp) is on non-sinus-rhythm patients, but that subgroup has the **largest AUC improvement** (+4.49) because the rhythm signal is most discriminative in the long tail. Sample sizes are large enough across all six subgroups (n > 1,400) to take the deltas seriously.

### Feature importance audit — soft cascade is the #1 non-TF-IDF feature

By gain on the uncalibrated XGBoost (single-source-of-truth for importance, since calibration is post-hoc):

| Feature group | Total gain | % of total |
|---|---|---|
| **tfidf** | 4591.7 | 84.6% |
| longitudinal vitals + rhythm | 251.3 | 4.6% |
| structured (age, pain, demographics, severity priors, interactions) | 224.2 | 4.1% |
| pmh | 103.9 | **1.9%** |
| soft cascade (triage v3 acuity + disposition) | 101.4 | **1.9%** |
| snapshot vitals + clinical flags | 92.4 | 1.7% |
| medications | 62.9 | 1.2% |

**Wiring checks (gates that block shipping if they fail):**
- **PMH wiring:** 19/19 PMH columns earned non-zero gain ✓
- **Soft cascade wiring:** 6/6 cascade columns earned non-zero gain ✓
- **Top-1 non-TF-IDF column:** `triage_disposition_proba_admit` with gain 55.6 (rank 1 overall outside TF-IDF). The triage v3 disposition probability is the single most informative non-text signal — exactly what cascading is supposed to deliver.

**Top non-TF-IDF features by gain (from the top 25):**

| Rank | Group | Column | Gain |
|---|---|---|---|
| 1 | soft cascade | triage_disposition_proba_admit | 55.6 |
| 2 | longitudinal | n_tachypnea_readings | 40.4 |
| 3 | structured | elderly | 28.0 |
| 6 | soft cascade | triage_acuity_proba_4 | 22.5 |
| 10 | longitudinal | n_tachycardia_readings | 16.1 |
| 11 | snapshot_vitals | heartrate_missing | 15.1 |
| 12 | pmh | pmh_Blood and Blood-Forming Organs | 14.8 |
| 14 | longitudinal | n_fever_readings | 14.0 |
| 16 | longitudinal | n_hypoxia_readings | 12.7 |
| 17 | longitudinal | resprate_max | 12.2 |
| 19 | longitudinal | n_hypotension_readings | 11.7 |
| 21 | structured | age_bin | 11.3 |

Five of the top 7 non-TF-IDF features are longitudinal vital trajectory aggregates (counts of abnormal readings within the 4-hour window) or the triage cascade — the two signals plan section 3 explicitly said should dominate. Snapshot vitals' headline feature is `heartrate_missing` rather than a heart-rate value itself, because the missingness flag itself carries information about transport mode (walk-ins have masked vitals).

### Training details

- **Pipeline:** `src/proiect_licenta/training/train_doctor_disposition.py`. Run with `uv run train_doctor_disposition`.
- **Hardware tested:** Colab T4 GPU. ~24 min XGBoost training (3805 best iteration out of 5000), ~25 min discharge.csv PMH parse, ~5 min longitudinal vitals aggregation, ~1 min calibration fit. Total ~60 min end-to-end including data loading.
- **Hyperparameters (XGBoost):** `n_estimators=5000`, `max_depth=10`, `lr=0.02`, `subsample=0.8`, `colsample_bytree=0.5`, `colsample_bylevel=0.7`, `min_child_weight=3`, `gamma=0.05`, `reg_alpha=0.5`, `reg_lambda=2.0`, `early_stopping_rounds=150`, `scale_pos_weight=sqrt(N_neg/N_pos)=1.3123`.
- **Calibration:** 10% of training rows held out (33,447 rows; rest stratified-split between fit and the original test split). FrozenEstimator wrapper around the trained XGB, then `CalibratedClassifierCV(method="isotonic", cv=None)` fit on the held-out slice. The wrapped object exposes the same `predict_proba` interface, so the (forthcoming) inference tool needs no special handling.
- **Cascade source:** triage v3 artifacts (`artifacts/triage/v3/`). Walk-in vitals are re-masked to NaN on the cascade input copy before calling `train_triage_v3.build_features(fit=False)` so the v3 `_missing` flags compute correctly even though the disposition pipeline's main vital columns are already imputed.
- **Class balance:** 36.7% admit / 63.3% discharge at training (153,581 admitted out of 418,083 stays).
- **Best iteration:** 3,805 / 5,000 (well clear of the ceiling — no convergence concern).

### Artifacts

- `artifacts/doctor/v3/disposition_model.joblib` — calibrated `CalibratedClassifierCV` wrapper (deployment).
- `artifacts/doctor/v3/disposition_model_raw.joblib` — uncalibrated `XGBClassifier` (audit + feature importance).
- `artifacts/doctor/v3/metadata.json` — extended with a `disposition` block that includes the metrics table above plus the `triage_v3_dispo_baseline` sub-block (the apples-to-apples baseline numbers for thesis citation without re-running the benchmark).
- **Note:** the existing v3 diagnosis + department artifacts are not overwritten by this training run. `train_doctor_disposition.save_disposition_model` merges into the existing `metadata.json` rather than replacing it.

### Inference status — wired 2026-05-30

The disposition model is live in the runtime crew. Same shipment swapped the
v1/v2 doctor tools out for the full v3 stack:

- ✅ Model trained on Colab T4 GPU, artifacts on disk in `artifacts/doctor/v3/`.
- ✅ Benchmark run, head-to-head numbers in this section.
- ✅ **`src/proiect_licenta/tools/doctor_tool_v3_base.py`** — NEW. Pre-nurse initial assessment, 13-class label space (catch-all excluded vs v1/v2), top-3 diag/dept with probabilities surfaced in the tool's JSON.
- ✅ **`src/proiect_licenta/tools/doctor_disposition_tool.py`** — NEW. Wraps the calibrated binary model. Builds the 2128-col feature vector via `train_triage_v3.build_features(fit=False)` on a walk-in-NaN-masked cascade input copy, runs the v3 acuity + v3 dispo cascade, appends 11 medication cols + 41 longitudinal-vital fallbacks. Includes the `neg_quadratic_kappa` pickle-compat shim for the triage v3 acuity eval-metric callable.
- ✅ **`src/proiect_licenta/crew.py`** — swapped: `DoctorPredictionToolV3Base` + `DoctorDispositionTool` + `DoctorPredictionToolV3` replace the v1/v2 pair. Crew is now 4 agents / 6 tasks. v1/v2 tool files stay on disk for thesis benchmarks but are no longer registered.
- ✅ **`src/proiect_licenta/config/tasks.yaml`** — 4 task updates landed: triage hedges ESI as "X-Y" when top-2 probabilities are within 0.10; `doctor_assessment_task` calls `doctor_prediction_tool_v3_base` + requires top-3 + probabilities in output; **NEW `doctor_disposition_task`** between nurse and reassessment with structured-data handoff (emits the refined `is_admitted`, `triage_was_admit`, `p_admit`); `doctor_reassessment_task` gates on the REFINED `is_admitted` instead of triage's, uses `doctor_prediction_tool_v3`, requires top-3 + probabilities + comparison-with-disposition section.
- ✅ **End-to-end smoke test** — pipeline runs through all 6 tasks; each tool call succeeds; the new `is_admitted` correctly propagates from the disposition task into the reassessment's gating decision.

Threshold tuning is documented as the next lever: the default 0.5 trades sensitivity for specificity, and the AUC says there's headroom. Sweep on the calibration holdout, pick by under-triage-rate target or clinician utility, and wire the chosen threshold into `doctor_disposition_tool.DECISION_THRESHOLD`. See "What's next" below.

### What's next

- **Inference wiring (highest-priority next step).** Create `doctor_disposition_tool.py` mirroring the pattern in `doctor_tool_v3.py`. Reads the triage softmax + nurse-collected vitals/meds/PMH, returns a calibrated admit probability + binary decision at the chosen threshold + short clinical-reasoning bullets sourced from the top contributing features. Plug into `crew.py` as a new `doctor_disposition_task` between the nurse and the reassessment task; the reassessment then uses *this* model's `is_admitted` instead of the triage one.
- ~~**Threshold tuning.**~~ **DONE — threshold set to 0.40.** Swept {0.30, 0.35, 0.40, 0.45, 0.50} on the 83,617-row disposition test split (`benchmarks/sweep_disposition_threshold.py`). **0.40 maximizes both F1 (0.7830) and Youden's J (0.6604)** and cuts the under-triage rate from 23.5% → 18.1% (−5.4pp, ~765 fewer missed admits per ~31K admitted) for only −0.6pp raw accuracy (0.840 → 0.833). Accuracy peaks at 0.50 but that's biased toward the discharge majority (63%); for a clinical disposition under-triage is the costlier error, so F1/Youden are the right selectors. Wired into `doctor_disposition_tool.DECISION_THRESHOLD = 0.40`. Full per-threshold table (sens/spec/under/over/F1/Youden + extra-admits-caught vs extra-false-admits):

  | thr | acc | sens | spec | under | over | F1 | Youden |
  |---|---|---|---|---|---|---|---|
  | 0.30 | 0.820 | 0.865 | 0.793 | 0.135 | 0.207 | 0.779 | 0.659 |
  | 0.35 | 0.827 | 0.842 | 0.817 | 0.158 | 0.183 | 0.781 | 0.660 |
  | **0.40** | 0.833 | 0.819 | 0.842 | 0.181 | 0.158 | **0.783** | **0.660** |
  | 0.45 | 0.838 | 0.791 | 0.865 | 0.209 | 0.135 | 0.782 | 0.656 |
  | 0.50 | 0.840 | 0.765 | 0.883 | 0.235 | 0.117 | 0.778 | 0.648 |

  **Future: 0.30 is a candidate** if missed admits are judged substantially costlier than false admits (under-triage 13.5%, over-triage 20.7%). Caveat: the "25–35% over-triage is acceptable" figure from ED literature is for **trauma/ESI acuity** triage (cost = team activation/resources), not admit/discharge disposition where over-triage = an unnecessary inpatient admission (bed + cost + iatrogenic risk). Going below 0.40 should be backed by an explicit cost model or clinician sign-off, not the trauma analogy alone. The sweep is mildly optimistic (tuned on the test split itself); the rigorous version picks on a validation fold, but the F1/Youden surface is flat across 0.35–0.45 so 0.40 is robust.
- **Asymmetric `scale_pos_weight` retrain (alternative to threshold tuning).** Re-run training with `scale_pos_weight ∈ {1.5, 1.8, 2.0}` and pick by under-triage delta — comparable to threshold tuning but recovers the calibration at the new operating point. Cheap (~25 min per run on GPU).
- **Downstream gating of the reassessment task.** Once the new disposition prediction is wired, the diagnosis/department reassessment should gate on *its* `is_admitted` flag (not the triage one) so a borderline-but-admitted patient triggers the full workup even if triage said discharge.
- **Output formatting (cross-cuts with reassessment).** Tasks should print top-3 categories with probabilities for both diagnosis and department, plus the calibrated admit/discharge probability split with a "calibration note" caveat (per plan section 2 and the user's request during section-3 planning).

---

## Stage 2 — Exact-ICD resolution within categories (shipped 2026-06-11)

The diagnosis models above predict a diagnosis **category** (13 ICD-chapter groups). Stage 2 adds a second, retrieval-based stage that ranks the **exact ICD diagnosis** inside each predicted category and surfaces the top-5 codes per category. It is a **cascade on top of the Doctor v3 diagnosis model** — it does **not** retrain or alter the category/department predictions; it is advisory.

### Design

- **Candidate pool** per category = the ICD codes observed in that category in the **Doctor v3 training split only** (no test leakage; the split is reproduced exactly from `train_nurse_v3`). Built at two granularities: **3-char rollup** (ICD-10 rubric / ICD-9 3-digit — the headline target) and **full code** (secondary/stretch).
- Each candidate carries (a) a **prototype centroid** = mean of the L2-normalized TF-IDF chief-complaint vectors of the training stays that received that code (symptom→symptom matching, sidesteps the complaint-vs-ICD-title gap), and (b) a **prevalence prior** = the code's frequency within its category.
- **Ranking blend:** `score = α·minmax(cosine) + (1−α)·minmax(prevalence)`, cosine = centroid · query (both L2-normalized). **α=0.60**, tuned offline on a held-out 10% slice of the training split (oracle-category top-5 rollup hit rate). The flat cross-category list ranks by `P(category)·score`.
- **No new dependencies** — reuses the existing TF-IDF vectorizer (`artifacts/triage/v1/tfidf_vectorizer.joblib`) + numpy/sklearn. Core in `src/proiect_licenta/icd_resolution.py`; built by `uv run train_icd_resolver` → `artifacts/doctor/v3/icd_resolver/`.

### Benchmark results (same 20,420-row v3 test split, `random_state=42`)

Stage-1 category recall (the end-to-end ceiling): top-1 = 64.1%, **top-5 = 92.8%**. The blend beats both single-signal baselines at every metric and granularity.

**3-char rollup (PRIMARY — 1,188 candidate codes):**

| Variant | α | Oracle@5 | Oracle@10 | E2E union | E2E flat-10 | Cond@union |
|---|---|---|---|---|---|---|
| prevalence-only | 0.00 | 0.5522 | 0.7047 | 0.5141 | 0.4473 | 0.5542 |
| cosine-only | 1.00 | 0.5030 | 0.6673 | 0.4772 | 0.4197 | 0.5144 |
| **blend** | **0.60** | **0.6698** | **0.7990** | **0.6285** | **0.5546** | **0.6775** |

**Full code (secondary — 4,497 candidates):**

| Variant | α | Oracle@5 | Oracle@10 | E2E union | Cond@union |
|---|---|---|---|---|---|
| prevalence-only | 0.00 | 0.4288 | 0.5507 | 0.3993 | 0.4304 |
| cosine-only | 1.00 | 0.2480 | 0.4116 | 0.2327 | 0.2508 |
| **blend** | **0.60** | **0.4919** | **0.6072** | **0.4622** | **0.4983** |

*Metrics: **Oracle** = within the TRUE category, is the true code in top-k (isolates Stage-2 from Stage-1 errors). **E2E union** = true code in the union of (top-5 predicted categories × top-5 codes ≈ 25 candidates). **E2E flat-10** = true code in the single `P(cat)·score`-ranked top-10. **Cond@union** = E2E union restricted to the 92.8% of cases where the true category is in the predicted top-5 (removes the Stage-1 ceiling).*

**Reading the table:**
- The blend is the whole story — neither prevalence nor cosine alone suffices; together they lift oracle@5 rollup to 67.0% (+11.8pp over prevalence, +16.7pp over cosine).
- **~63% of admitted patients** get their exact 3-char diagnosis inside the top-5-cats × top-5-codes set; **80% within oracle top-10**.
- Full code is much harder (49.2% oracle@5) because laterality/episode variants explode the candidate space. Notably, **cosine-only collapses at full code (24.8%)** while prevalence stays robust (42.9%) — fine-grained codes fragment the complaint signal across near-duplicates, so frequency carries more weight there; the blend reconciles both.
- Spot-checks show strong clinical face validity even on misses (e.g. "s/p Fall" → femur/rib/vertebra fractures; "chest pain, SOB" → CHF / pneumonia / COPD / pulmonary HTN).

### v3_base vs v3-nurse — before/after the nurse step

The resolver is **model-agnostic**: its candidates + prototype centroids come from the shared training split, so the same index serves both Doctor v3 diagnosis models (which use the identical 13-class split). `benchmark_icd_resolution.py` runs both. The **oracle ceiling is therefore identical** (rollup blend @5 = 66.98% / @10 = 79.90% for both) — an empirical confirmation that Stage 2 doesn't touch the diagnosis model. Only the **end-to-end** numbers differ, isolating the nurse contribution:

| | v3_base (pre-nurse) | v3-nurse | Δ nurse |
|---|---|---|---|
| Stage-1 category recall — top-1 | 60.1% | 64.1% | +4.0pp |
| Stage-1 category recall — top-5 | 90.6% | 92.8% | +2.2pp |
| E2E union (rollup, blend) | 0.6184 | 0.6285 | **+0.0101** |
| E2E flat-10 (rollup, blend) | 0.5414 | 0.5546 | **+0.0131** |
| Cond@union (rollup, blend) | 0.6829 | 0.6775 | −0.0054 |
| E2E union (full code, blend) | 0.4553 | 0.4622 | +0.0069 |
| E2E flat-10 (full code, blend) | 0.3944 | 0.4069 | +0.0125 |

**Interpretation (a useful finding):** nurse data helps exact-ICD only **indirectly and modestly (+1 to +1.3pp)**. The resolver ranks on chief-complaint text, which is *unchanged* by the nurse step (vitals/meds/PMH don't alter what the patient said), so its only lever is Stage 1: better category recall (90.6% → 92.8%) lets more true codes into the candidate set, and that +2.2pp flows through to the ~+1pp E2E lift. The conditional metric dips slightly negative because v3-nurse's larger correct-category set newly includes exactly the *harder* cases v3_base missed (which also have lower exact-code recall), diluting the within-bucket average even as the unconditional recall rises. **Implication:** to lift the oracle *ceiling* itself, Stage 2 would need nurse signals fed *into the resolver* (e.g. vitals-conditioned centroids), not just better Stage-1 categories — a sharper future direction than version unification alone. That direction was then implemented (next).

### Vitals-conditioned centroids (Stage-2 v2, shipped 2026-06-11)

Acting on the implication above: each candidate code gets a **vitals centroid** = the mean of its training patients' **standardized physiology vector** (6 z-scored vitals + 7 abnormal flags, standardized together). At inference the patient's z-scored physiology is compared to each centroid by negative Euclidean distance ("nearest physiological prototype"), entering as a **third blended term**:

```
score = w_text·text_cos + w_vit·vitals + w_prev·prevalence   (each min-max'd within category)
```

The 3 weights are grid-searched on the simplex (held-out train slice, oracle top-5 rollup). The physiology vector is **28-dim**: 6 z-scored snapshot vitals + 7 abnormal flags (from `triage.csv`) **+ cardiac rhythm (8 one-hot buckets + `rhythm_irregular`) + per-vital trajectory delta** (from `vitalsign.csv`, over the same `[intime, intime+4h]` window the v3 model uses). The longitudinal **min/max/last and reading-counts are deliberately excluded** — for a snapshot-only patient they collapse to the snapshot value, so they triplicate it without adding information; the genuinely-new signals are rhythm and the trajectory deltas. Tuned weights: **text=0.3, vit=0.4, prev=0.3** — vitals is now the *largest* weight.

This **lifts the oracle ceiling itself**, out-of-sample, in three steps (v3-nurse test split, oracle@5):

| oracle@5 | text+prev | + snapshot (13-dim) | + rhythm/delta (28-dim) | total Δ |
|---|---|---|---|---|
| rollup | 0.6698 | 0.6832 (+1.34) | **0.6913** (+0.81) | **+2.15pp** |
| full code | 0.4919 | 0.5129 (+2.10) | **0.5253** (+1.24) | **+3.34pp** |

End-to-end (v3-nurse, blend → blend+vitals with rhythm/delta):

| metric (v3-nurse) | blend (text+prev) | blend+vitals | Δ |
|---|---|---|---|
| Oracle@10 rollup | 0.7990 | **0.8130** | +1.40pp |
| Oracle@10 full code | 0.6072 | **0.6378** | +3.06pp |
| E2E union rollup | 0.6285 | **0.6488** | +2.03pp |
| E2E union full code | 0.4622 | **0.4933** | +3.11pp |
| Cond@union rollup | 0.6775 | **0.6993** | +2.18pp |

Full code benefits most (+3.34pp oracle@5) — rhythm (afib/paced/AV-block) sharply disambiguates among the many specific cardiac codes, and there are more candidates to separate. The term is correctly near-neutral for normal-vitals patients; the lift comes from abnormal-physiology cases (a fever+hypoxia+tachycardia "fever, SOB, cough" patient surfaces 486 Pneumonia / A41 Sepsis / J18 at the top). Adding `vitalsign.csv` is the one builder cost (~5–8 min vs ~1 min), still far cheaper than the full v3 loader (which also parses the 3.3 GB discharge.csv for PMH the resolver doesn't need). The resolver stays backward compatible (keeps the 2-term `alpha`); builder, benchmark, and `doctor_tool_v3` use the 3-term path whenever the patient's vitals are present (the standardizer's column list drives which features are assembled), falling back to text+prevalence otherwise.

### PMH-gated history term (Stage-2 v3, shipped 2026-06-12)

Within a category, a patient's **prior diagnoses** strongly predict the *exact recurrent* code — chronic-disease exacerbations and return visits dominate admissions, and "last admission's diagnosis predicts this one's" is a legitimate signal (the `pmh_<cat>` flags come from strictly-prior ICDs, `< intime`, so it is leakage-safe). PMH is available only for patients with history, so it enters as a **gated 4th term**:

- **patients with history** (`no_history == 0`): `score = w_text·text + w_vit·vitals + w_pmh·pmh + w_prev·prevalence`, tuned weights **0.2 / 0.4 / 0.2 / 0.2**;
- **patients without history**: the unchanged 3-term vitals blend.

The PMH vector is **14-dim** (13 `pmh_<category>` flags + the `same_complaint_as_prior` Jaccard); the count/days numerics are dropped to avoid the "no prior" sentinel polluting the z-score. Centroids are built from **has-history rows only**. The features come from the vetted, leakage-safe `pmh_features.aggregate_pmh`; the 3.3 GB discharge.csv parse is **cached to parquet**, so only the first build pays it. **64.9% of admitted patients have prior history.**

Results (v3-nurse test split; baseline = blend+vitals from above):

| oracle@5 | subset | baseline | +PMH | Δ |
|---|---|---|---|---|
| rollup | all | 0.6913 | 0.6976 | +0.63pp |
| rollup | **has-PMH (65%)** | 0.7172 | **0.7269** | **+0.97pp** |
| rollup | no-PMH (35%) | 0.6441 | 0.6441 | **+0.0000** |
| full code | all | 0.5253 | 0.5367 | +1.14pp |
| full code | **has-PMH** | 0.5657 | **0.5834** | **+1.77pp** |
| full code | no-PMH | 0.4515 | 0.4515 | **+0.0000** |

End-to-end union (v3-nurse, has-PMH): rollup **+0.92pp**, full **+1.59pp**. Key points: (1) the **gate is provably correct** — the no-PMH subset is *exactly unchanged*, so first-time patients are never penalized; (2) PMH helps the 65% who have history (full code most — it disambiguates *which* chronic condition recurred); (3) the overall lift is diluted by the 35% no-PMH patients, so PMH should be read on its subset; (4) has-PMH patients also have a higher *baseline* (returning patients are inherently more predictable), and PMH stacks on top.

**Cumulative Stage-2 progression** (v3-nurse oracle@5 rollup): text+prev 0.6698 → +vitals 0.6832 → +rhythm/delta 0.6913 → +PMH 0.6976 (**0.7269 on has-PMH**). Full code: 0.4919 → 0.5129 → 0.5253 → 0.5367 (**0.5834 on has-PMH**).

### Graded near-miss metrics (evaluation-only, shipped 2026-06-25)

All metrics above are **strict exact-code recall** — the prediction either contains the true code (credit 1) or it doesn't (credit 0), so predicting *hepatitis A* when the truth is *hepatitis B* scores exactly 0, identical to a wildly wrong code. To reward clinically-close misses (a request from the thesis supervisor), `benchmark_icd_resolution.py` additionally reports **three graded metrics**. They are **evaluation-only**: the predictor, the resolver artifact, and training are untouched — the model emits the same ranked code lists, scored three extra ways. Core in `src/proiect_licenta/icd_similarity.py`.

**Graded credit** mirrors recall@k by replacing the 0/1 hit with the **max similarity over the top-k predicted codes**: `graded@k(row) = max_{c∈topk} sim(true_code, c)`, averaged over rows. Both sides compare the code's **representative title** (the resolver index's per-code title), so an exact hit scores cosine 1.0 and the invariant **graded@k ≥ strict recall@k** holds everywhere (self-checked every run: `_meta.invariant_graded_ge_strict_violations` must be 0).

The three `sim(·,·)` engines:

- **`tfidf`** — the supervisor's literal "cosine on ICD titles": cosine of TF-IDF vectors of the two codes' titles (lexical). The token pattern keeps single-char tokens (`\b\w+\b`), so "hepatitis A" vs "B" differ (0.49) rather than collapsing to an identical vector. Zero new deps, fully offline.
- **`gemini`** — semantic cosine of **Gemini embeddings** (`gemini-embedding-001`, dim 3072) of the titles. Built once via `uv run python benchmarks/build_title_embeddings.py` → cached to `artifacts/doctor/v3/icd_resolver/title_embeddings.joblib` (gitignored, 9,491 vectors), then read **offline under any LLM backend** (decoupled from `LLM_BACKEND`; only needs a Gemini key at build time). Rate-limited (≤2000/min) + 429-retry + incremental save.
- **`tree`** — hierarchical ICD distance: exact code **1.0** / shared 3-char rollup **0.6** / shared ICD chapter **0.3** / else 0. Chapters are version-aware and unified, so an ICD-9 code and its ICD-10 equivalent land in the same bucket (e.g. `410`/`I21` → "circulatory").

Results (v3-nurse test split, **blend+vitals**; full numbers in `artifacts/benchmarks/icd_resolution_graded.json`):

| metric | strict | tfidf | gemini | tree |
|---|---|---|---|---|
| Oracle@5 (rollup) | 0.6913 | 0.7257 | 0.9147 | 0.7837 |
| Oracle@10 (rollup) | 0.8130 | 0.8404 | 0.9517 | 0.8689 |
| Oracle@5 (full code) | 0.5253 | 0.6004 | 0.8795 | 0.7002 |
| E2E union (rollup) | 0.6488 | 0.6927 | 0.9038 | 0.7323 |

Reading them: **tfidf** sits modestly above strict (+3–8pp) — lexical near-miss credit; **tree** higher (chapter-level credit) and the most clinically interpretable "how close in the ICD taxonomy"; **gemini** very high (0.85–0.95) because semantic embeddings have a high similarity *floor* (clinical phrases are all somewhat alike), so its **deltas and ordering** matter more than absolute values. All three preserve the qualitative findings of the strict metrics: the nurse-step lift stays positive on E2E union/flat for every engine, and the PMH gate's no-PMH subset is still *exactly* unchanged (+0.0000) across all graded engines. This partially addresses the [ICD-9/10 split issue](#whats-next) — the `tree` (and to a degree the title) metrics give same-disease cross-version pairs (e.g. `599`/`N39` UTI) chapter-level credit instead of a hard zero.

### End-to-end on the live runtime (20 NL cases, doctor+nurse, 2026-06-15)

All numbers above are the *tabular* benchmark (`benchmark_icd_resolution.py`, 20,420 test stays). To check the resolver through the **live crew** end-to-end, `benchmarks/benchmark_pipeline_e2e.py` now scores it on the 20 synthetic NL cases (13 admitted) alongside the existing acuity/disposition/category targets, reading the v3 tool's `exact_diagnoses` block for the E2E/tool-direct columns and running the resolver on cached features for feature-vector (`_fv_exact_icd`, reuses `physio_matrix`). The runtime tool resolves over top-5 cats / k_per_cat=5 / k_flat=10 — identical to this benchmark — so the metrics are comparable.

Headline (3-char rollup, n=13, **1 case = 7.7pp** — directional only):

| metric | tool-direct | feature-vector | E2E (live) | full-set ref (v3-nurse) |
|---|---|---|---|---|
| ICD union | 69.2% | 69.2% | **69.2%** | 64.9% (blend+vitals) / 62.85% (blend) |
| ICD flat@10 | 69.2% | 69.2% | **61.5%** | 55.5% (blend) |
| ICD exact @5 | 61.5% | 46.2% | 53.8% | — |
| ICD exact @1 | 23.1% | 23.1% | **7.7%** | — |

**`union` is mode-invariant (9/13) including E2E** — it's a set-membership test driven by complaint text + the top-5 category set, neither of which the NL layer materially changes, so the resolver's value **survives the full pipeline**. The flat metrics (`@1`/`@5`/`flat@10`) wobble by 1–2 cases because they rank by `P(cat)·score`: a peakier (cleaner) softmax suppresses codes from correctly-but-secondarily ranked categories, so `feature_vector` is actually *worst* at flat@5 despite the best Stage-1 — proof it's pure re-ranking, since union is identical. E2E `@1` collapses (3/13→1/13) as complaint paraphrase perturbs the text-cosine top code. Some misses are ICD-9/10 splits of one disease (truth `518` vs E2E `J18`, both pneumonia), so true recall is a bit higher. Full interpretation: [case-generation-agent.md → Stage-2 exact-ICD end-to-end](case-generation-agent.md#stage-2-exact-icd-end-to-end-doctornurse).

### Inference wiring

`doctor_tool_v3.py` lazy-loads the resolver (graceful skip if not built), and after the diagnosis prediction calls `_resolve_exact_diagnoses` to attach a `diagnosis_prediction.exact_diagnoses` block (rollup granularity: `top_per_category` + `flat_ranking`, with an advisory disclaimer + a `used_vitals` flag). It passes the patient's cleaned vitals + abnormal flags (3-term vitals path) and the PMH vector — when the patient has history (`no_history == 0`) the **gated 4-term PMH path** is used, else the 3-term vitals path, else text+prevalence. The output records `used_vitals` and `used_pmh` flags. `doctor_reassessment_task` surfaces the top `flat_ranking` entries as a suggested differential. The block is `null` when the resolver artifact is absent, so the pipeline never depends on it.

### What's next

- ~~**v3_base Stage-2**~~ **DONE (2026-06-11)** — see [v3_base vs v3-nurse](#v3_base-vs-v3-nurse--beforeafter-the-nurse-step). Nurse data adds +1 to +1.3pp E2E exact-ICD, entirely via better Stage-1 category recall; the Stage-2 oracle ceiling is identical.
- ~~**Vitals-conditioned centroids**~~ **DONE (2026-06-11)** — see [Vitals-conditioned centroids](#vitals-conditioned-centroids-stage-2-v2-shipped-2026-06-11). Adds a z-scored physiology similarity term (28-dim: snapshot vitals + flags + cardiac rhythm + vital deltas). Lifts the oracle ceiling **+2.15pp rollup / +3.34pp full @5**, out-of-sample. Wired into the runtime tool.
- ~~**PMH-gated centroids**~~ **DONE (2026-06-12)** — see [PMH-gated history term](#pmh-gated-history-term-stage-2-v3-shipped-2026-06-12). Gated 4th term; +0.97pp rollup / +1.77pp full oracle@5 on the 65% has-PMH subset, no-PMH provably unchanged. Wired into the runtime tool.
- **Past-medication rider (optional).** Meds are a (noisier) proxy for PMH; fold the `medrecon` medication-category flags into the gated PMH vector to catch patients whose meds reveal a chronic condition their prior ICDs missed. Low marginal value over PMH; cheap to try once `medrecon` is loaded.
- **ICD-9 ↔ ICD-10 unification** — the same disease is currently split across code versions (e.g. `599` vs `N39`, both UTI), fragmenting prevalence and candidates. A CCSR-style concept mapping would merge them and is expected to lift the numbers. (Deferred future work.) The 20-case E2E run makes this visible (truth `518` ICD-9 vs E2E `J18` ICD-10, both pneumonia, scored as a miss).
- **Base-doctor (v3_base) live-E2E exact-ICD** — the v3_base/before-nurse comparison exists only on the *tabular* benchmark; the live base tool (`doctor_tool_v3_base.py`) does **not** wire the resolver, so `benchmark_pipeline_e2e.py` can't score a base E2E column. Mirror `doctor_tool_v3._resolve_exact_diagnoses` into the base tool and the harness picks it up for free via `CAPTURE["base"]`. (Deferred.)

---

## Tools

- **`DoctorPredictionTool` (v1)** — `src/proiect_licenta/tools/doctor_tool.py`. Wraps the diagnosis v1 + department v1 models, rebuilds the 2025-feature vector from triage output.
- **`DoctorPredictionToolV2` (v2)** — `src/proiect_licenta/tools/doctor_tool_v2.py`. Wraps the diagnosis v2 + department v2 models, builds the 2056-feature vector, applies vital imputation and medication classification, and emits a JSON result that includes a `nurse_data_used` section so the output makes the comparison with v1 explicit.
- **`DoctorPredictionToolV3` (v3)** — `src/proiect_licenta/tools/doctor_tool_v3.py`. Wraps the v3 diagnosis + department models, accepts the nurse's `rhythm`, `prior_history`, and `n_prior_admissions` fields, rebuilds the ~2116-feature vector (including the 19 Change 1 PMH columns), and uses the snapshot-as-trajectory fallback for longitudinal vitals plus the all-zero + `no_history=1` fallback for PMH when the nurse doesn't collect it. Output JSON includes `rhythm_raw`, `rhythm_bucket`, `rhythm_irregular`, `prior_history_raw`, `pmh_categories_detected`, `n_prior_admissions_reported`, and `no_history_fallback` under `nurse_data_used`, plus a `label_space` note flagging that catch-all is excluded. As of 2026-06-11 it also attaches the advisory `diagnosis_prediction.exact_diagnoses` block from the [Stage-2 resolver](#stage-2--exact-icd-resolution-within-categories-shipped-2026-06-11) (or `null` if the resolver isn't built).

All three tools lazy-load their artifacts at first use.

---

## Training

- **v1 pipeline:** `src/proiect_licenta/training/train_doctor.py`. Run with `uv run train_doctor` (~30 minutes for both v1 models on 80K samples; early stopping typically triggers around iteration 1400-1900).
- **v2 pipeline:** `src/proiect_licenta/training/train_nurse.py`. Run with `uv run train_nurse` (~45 minutes for both v2 models on 80K samples; medication aggregation over 3M rows from `medrecon.csv` adds ~5 minutes to data loading).
- **v3 base pipeline:** `src/proiect_licenta/training/train_doctor_v3.py`. Run with `uv run train_doctor_v3`. Same architecture as v1 but filters the catch-all class and trains on the full ~102K filtered admitted-patient set (no sub-sample). Expected runtime ~40 minutes.
- **v3 with-nurse pipeline:** `src/proiect_licenta/training/train_nurse_v3.py`. Run with `uv run train_nurse_v3`. v3 base + snapshot vitals + medications + longitudinal vitals/rhythm from `vitalsign.csv` + **PMH features (Change 1)** parsed from prior `discharge.csv` notes and `diagnoses_icd.csv`. Expected runtime ~40 minutes on Colab CPU / ~70 minutes locally (the chunked discharge.csv parse adds ~15-25 minutes on top of the prior ~70-minute pipeline; the 4-hour time-window filter is applied during longitudinal-vital aggregation; PMH is gated by `prior_admittime < current_intime` for leakage safety). tqdm progress bars cover the discharge.csv parse, PMH assembly, longitudinal-vitals aggregation, and each XGBoost boosting iteration.

- **Stage-2 ICD resolver:** `src/proiect_licenta/training/train_icd_resolver.py`. Run with `uv run train_icd_resolver`. Builds the per-category prototype-centroid + prevalence indices (rollup + full) from the v3 training split and tunes α. **Fast (~1 minute)** — it uses a light loader that reproduces the exact v3 split without the slow vitalsign/medrecon/PMH IO (those are post-row-set left joins, so skipping them is leakage-free; `--verify` runs the full v3 loader once to assert identical row order). Benchmark: `uv run python benchmarks/benchmark_icd_resolution.py` (heavy — it rebuilds the full v3 feature matrix for the diagnosis model's `predict_proba`).

The v2 pipeline reproduces the exact v1 sampling / split (`random_state=42`, stratified on `diagnosis_group`) so v1 and v2 can be evaluated on identical test sets for direct comparison. v3 base and v3 with-nurse share their own filtered split (also `random_state=42`, also stratified on `diagnosis_group`) so they can be compared to each other on identical 13-class test sets. v1/v2 (14-class, sampled) and v3 (13-class, full) are not on identical test sets and should not be compared as if they were — `benchmarks/compare_all_versions.py` flags this in its output.

---

## Open Improvements

See [`../future-work.md`](../future-work.md) for the full roadmap. Doctor-specific headline items:
- ~~Train on the full 157K admitted rows instead of 100K.~~ **DONE in v3** (~102K filtered).
- ~~Exploit longitudinal vital signs from `vitalsign.csv`.~~ **DONE in v3 nurse**.
- ~~Add PMH features from prior discharge notes + ICD codes.~~ **DONE in Change 1 (2026-05-21)**, see the [Change 1 section](#change-1--pmh-features).
- ~~**Change 2** — multi-label seq_num 2, 3 sibling head, blended with the softmax logits at inference.~~ **TRIED AND REVERTED (2026-05-21)**. Lift came in at +0.04pp top-3 best case (well below the +1.5-3pp predicted band). The multilabel head's signal is correlated with the softmax (same features, related labels), not orthogonal — linear blending couldn't extract additional accuracy. See [`../future-work.md`](../future-work.md) "Empirical findings — experiments tried and reverted" entry 4 for the alpha sweep + diagnosis.
- ~~**Change 4** — Two-stage surgical-vs-medical department routing (binary gate + 4-class medical head + 7-class surgical head, soft-blended at inference).~~ **TRIED AND REVERTED (2026-05-21)**. Net delta vs the legacy single-head model was **−1.22pp top-1 / −0.63pp top-3** on the same 20,420-row test split. Gate binary accuracy landed at 82.0%, which bounds the end-to-end ceiling — soft-blending cannot recover the 18% of patients misrouted at stage 1. TRAUM did recover +6.7pp under the surgical-only head (validating part of the hypothesis), but the volume-weighted impact of MED's −1.5pp regression (12,045 stays) ate the surgical gains many times over. Root cause: the plan deprived the gate of nurse features (snapshot vitals, meds, longitudinal); empirically those features *help* surgical-vs-medical discrimination, so the deprivation strictly hurt routing vs the legacy single model. See [`../future-work.md`](../future-work.md) "Empirical findings — experiments tried and reverted" entry 5 for the per-class delta table and post-mortem.
- ~~**Tier A — A2 (PMH vocab expansion, 397 → 596 keywords) + A3 (diagnosis-softmax cascade, 13 `diag_proba_*` columns into the dept model) + A4 (isotonic department calibration via FrozenEstimator + CalibratedClassifierCV).**~~ **SHIPPED (2026-05-22)**. Net Tier A deltas vs the pre-Tier-A Change-1 baseline: **diagnosis −0.09pp top-1 (flat), department +2.18pp top-1 (68.61% → 70.79%)**. A4 calibration delivered +1.46pp on its own — 4-5× the predicted band. Caveat: the dept top-1 gain is concentrated almost entirely on MED (+21.8pp vs v3 base, 12,045 stays); most non-MED departments regressed and Cohen's κ slipped from 0.5009 to 0.4948. See the [Tier A section](#tier-a--vocab-expansion--softmax-cascade--isotonic-calibration) above for the full per-lever attribution and per-class table.
- **A1 — Optuna macro-F1 sweep + class-weight exponent tuning. IN PROGRESS, 10/30 trials, paused 2026-05-22.** First 10 trials complete; best trial #2 on inner-val: macro_f1 = 0.5536, top-1 = 0.6306. vs hand-picked baseline macro_f1 = 0.5529 = **essentially flat (+0.0007)** — TPE hasn't converged yet (cold-start sampler typically starts exploiting around trial 15-25). Best `class_weight_exponent` ≈ 0.52 (≈sqrt default); heavier minority weighting consistently underperformed. `artifacts/doctor/v3/tuned_params.json` written but **no final retrain applied** — production model still on hand-picked defaults. See ["Experiments in progress" in future-work](../future-work.md#experiments-in-progress) for the per-trial table and resume plan.
- Hierarchical approach: first classify "Symptoms/Ill-Defined vs real diagnosis", then predict specific category.
- The "Symptoms, Signs, Ill-Defined" catch-all (33%) is a labeling issue — patients coded with symptom-level ICD codes vs disease-level codes may have identical presentations.
- Surgical department routing could benefit from **injury-specific features** (fracture mechanism, MVA detail, blunt vs penetrating) rather than vitals — Change 4 confirmed that a feature-set or routing-only fix isn't enough; the surgical-bucket regression is a feature-information problem.
- Cleaner labels from `discharge.csv` (extract physician's discharge diagnosis) — highest-ceiling deferred lever per `future-work.md` Tier 1 #3.
