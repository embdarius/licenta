# Doctor Agent (v1 + v2 + v3)

The Doctor Agent is the only agent in the pipeline that runs **twice**:
- **Phase 1 (v1):** Initial assessment using triage data only.
- **Phase 2 (v2):** Enhanced reassessment using triage data **plus** the vital signs and medication list collected by the Nurse Agent.

A **v3 tier** sits alongside v1/v2 (it does not replace them). v3 drops the "Symptoms, Signs, Ill-Defined" catch-all class so the model trains on a clinically meaningful 13-class label space, runs on the full filtered admitted-patient dataset (no 100K cap), and the v3 with-nurse variant adds longitudinal vitals + cardiac rhythm from `vitalsign.csv`. v1 and v2 stay in the repo as the 14-class baselines for direct thesis comparison. See the [v3 section](#phase-3--doctor-v3-13-class-label-space) below.

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
  - **Medications from medrecon.csv (11 features):** same as v2 — `n_medications`, `meds_unknown`, plus 9 binary category flags via the shared `med_vocab.flags_from_row` vocabulary.
  - **Longitudinal vital trajectory (24 features):** for each of the 6 vitals, `<vital>_min`, `<vital>_max`, `<vital>_last`, `<vital>_delta` aggregated over readings within `[intime, intime + 4h]` (the 4h window guards against late-stay disposition leakage).
  - **Abnormal-reading counts (7 features):** `n_fever_readings`, `n_tachycardia_readings`, `n_bradycardia_readings`, `n_tachypnea_readings`, `n_hypoxia_readings`, `n_hypertension_readings`, `n_hypotension_readings`.
  - **Cardiac rhythm (10 features):** one-hot over 8 normalized buckets (`sinus`, `sinus_tachy`, `sinus_brady`, `afib_flutter`, `paced`, `av_block`, `svt`, `other`) plus `rhythm_irregular` (any non-sinus reading) plus `has_longitudinal_vitals` (was the stay covered in `vitalsign.csv` at all).
  - **Total:** ~2097 features (~72 nurse-derived on top of the v3-base 2025).

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
| **v3 nurse**   | **13**           | **63.08%**     | **84.95%**     | **11**           | **67.11%**     | **93.38%**     | **81,680**  | **20,420** |

**Improvement deltas:**
- v3 base → v3 nurse (same 13-class space, isolates the value of nurse data): **+2.99pp diagnosis, +6.45pp department**. This is slightly larger than the v1→v2 nurse lift (+2.26pp / +5.91pp), suggesting the longitudinal vitals + rhythm features added on top of snapshot vitals are pulling their weight.
- v1 → v3 base (impact of catch-all exclusion + full dataset, both no-nurse): **+9.91pp diagnosis, +1.53pp department**. Diagnosis benefits massively because the catch-all class was absorbing 20–28% of misclassifications from every other category; department only ticks up because the catch-all was a diagnosis-side problem, not a department-side one.
- v2 → v3 nurse (full pipeline upgrade — but DIFFERENT label spaces, not apples-to-apples): **+10.64pp diagnosis, +2.07pp department**.

**Top-5 diagnosis accuracy on v3 nurse: 92.43%** — top-5 is essentially the entire useful prediction window, so for clinical decision support where the doctor sees a short ranked list of candidate categories, v3 nurse is near-ceiling on the 13-class task.

**Per-class diagnosis recall — v3 base → v3 nurse (where nurse data moves the needle):**

| Category | v3 base | v3 nurse | Delta | Notes |
|---|---|---|---|---|
| Circulatory | 59.2% | 67.8% | **+8.6pp** | Rhythm + HR trends help most |
| Genitourinary | 37.3% | 42.8% | **+5.5pp** | Often confused with Digestive; vitals help disambiguate |
| Respiratory | 61.9% | 66.8% | **+4.9pp** | O2 sat / resprate trends help |
| Endocrine | 35.9% | 39.6% | +3.7pp |  |
| Injury and Poisoning | 71.0% | 73.0% | +2.0pp |  |
| Digestive | 78.9% | 80.7% | +1.8pp |  |
| Infectious | 19.9% | 20.5% | +0.6pp | Still the weakest class |
| Mental | 80.3% | 79.3% | -1.0pp |  |
| Musculo | 58.0% | 57.0% | -1.0pp |  |
| Nervous | 54.8% | 53.7% | -1.1pp |  |
| Skin | 58.3% | 56.9% | -1.4pp |  |
| Blood | 40.4% | 36.5% | -3.9pp |  |
| Other | 23.2% | 18.7% | -4.5pp |  |

**Per-class department recall — v3 base → v3 nurse:**

| Department | v3 base | v3 nurse | Delta |
|---|---|---|---|
| MED | 65.4% | 77.2% | **+11.7pp** |
| OMED | 14.9% | 21.6% | **+6.8pp** |
| NMED | 69.5% | 73.1% | +3.6pp |
| CMED | 66.2% | 67.6% | +1.4pp |
| ORTHO | 65.4% | 64.4% | -1.0pp |
| NSURG | 41.9% | 40.6% | -1.3pp |
| TRAUM | 59.5% | 54.8% | -4.7pp |
| SURG | 56.1% | 51.0% | -5.0pp |
| OB_GYN | 46.2% | 38.7% | -7.6pp |
| OTHER | 23.9% | 10.7% | -13.2pp |
| OTHER_SURG | 36.5% | 23.1% | -13.3pp |

**The "surgical regression" pattern from v2 carries over to v3.** Nurse-collected vitals + medications + rhythm help medical departments (MED, OMED, NMED, CMED) where physiological state directly drives routing, and hurt surgical departments (SURG, TRAUM, OTHER_SURG, OB_GYN, OTHER) where routing depends mostly on injury type already captured in the chief-complaint TF-IDF. This is a structural property of the feature set, not a v3 bug — Phase C of the improvement plan covers a possible fix (specialty-conditional masking of nurse features).

**Top misclassification patterns on v3 base** (the catch-all is gone, so these are *real* category confusions, not labeling noise):

1. Circulatory → Respiratory (14.1% of true Circulatory) — shared symptoms (chest pain, shortness of breath)
2. Respiratory → Circulatory (12.8%) — same axis
3. Genitourinary → Digestive (14.8%) — abdominal-pain ambiguity
4. Skin → Musculoskeletal (14.4%) — limb complaints
5. Musculoskeletal → Injury (11.3%) — fracture vs musculoskeletal pain

These are the new top error pairs worth attacking next.

**Feature importance:** TF-IDF terms still dominate the top 20 in both v3 base and v3 with-nurse (specific complaint language like "sbo transfer", "pain pregnant", "motor vehicle collision", "sodium level", "sickle cell crisis"). No nurse-specific feature appeared in the top 50 of v3 nurse — yet the model is +3pp diagnosis, +6.5pp department better than v3 base. This means the nurse features contribute through many small interactions rather than as standalone heavy hitters.

**Run the benchmarks yourself:**

```
uv run python benchmarks/benchmark_doctor_v3.py    # full v3 base report
uv run python benchmarks/benchmark_nurse_v3.py     # v3 base vs v3 with-nurse
uv run python benchmarks/compare_all_versions.py   # four-way table
```

---

## Tools

- **`DoctorPredictionTool` (v1)** — `src/proiect_licenta/tools/doctor_tool.py`. Wraps the diagnosis v1 + department v1 models, rebuilds the 2025-feature vector from triage output.
- **`DoctorPredictionToolV2` (v2)** — `src/proiect_licenta/tools/doctor_tool_v2.py`. Wraps the diagnosis v2 + department v2 models, builds the 2056-feature vector, applies vital imputation and medication classification, and emits a JSON result that includes a `nurse_data_used` section so the output makes the comparison with v1 explicit.
- **`DoctorPredictionToolV3` (v3)** — `src/proiect_licenta/tools/doctor_tool_v3.py`. Wraps the v3 diagnosis + department models, accepts the nurse's `rhythm` field, rebuilds the ~2097-feature vector, and uses the snapshot-as-trajectory fallback described above. Output JSON includes `rhythm_raw`, `rhythm_bucket`, and `rhythm_irregular` under `nurse_data_used`, plus a `label_space` note flagging that catch-all is excluded.

All three tools lazy-load their artifacts at first use.

---

## Training

- **v1 pipeline:** `src/proiect_licenta/training/train_doctor.py`. Run with `uv run train_doctor` (~30 minutes for both v1 models on 80K samples; early stopping typically triggers around iteration 1400-1900).
- **v2 pipeline:** `src/proiect_licenta/training/train_nurse.py`. Run with `uv run train_nurse` (~45 minutes for both v2 models on 80K samples; medication aggregation over 3M rows from `medrecon.csv` adds ~5 minutes to data loading).
- **v3 base pipeline:** `src/proiect_licenta/training/train_doctor_v3.py`. Run with `uv run train_doctor_v3`. Same architecture as v1 but filters the catch-all class and trains on the full ~102K filtered admitted-patient set (no sub-sample). Expected runtime ~40 minutes.
- **v3 with-nurse pipeline:** `src/proiect_licenta/training/train_nurse_v3.py`. Run with `uv run train_nurse_v3`. v3 base + snapshot vitals + medications + longitudinal vitals/rhythm from `vitalsign.csv`. Expected runtime ~70 minutes (the chunked load of the 115 MB `vitalsign.csv` adds ~10 minutes; the 4-hour time-window filter is applied during aggregation).

The v2 pipeline reproduces the exact v1 sampling / split (`random_state=42`, stratified on `diagnosis_group`) so v1 and v2 can be evaluated on identical test sets for direct comparison. v3 base and v3 with-nurse share their own filtered split (also `random_state=42`, also stratified on `diagnosis_group`) so they can be compared to each other on identical 13-class test sets. v1/v2 (14-class, sampled) and v3 (13-class, full) are not on identical test sets and should not be compared as if they were — `benchmarks/compare_all_versions.py` flags this in its output.

---

## Open Improvements

See [`../future-work.md`](../future-work.md) for the full roadmap. Doctor-specific headline items:
- Train on the full 157K admitted rows instead of 100K.
- Add second/third diagnoses (seq_num 2, 3) as multi-label training signal.
- Hierarchical approach: first classify "Symptoms/Ill-Defined vs real diagnosis", then predict specific category.
- Exploit longitudinal vital signs from `vitalsign.csv` (1.4M readings during stay) vs the current single triage vitals.
- The "Symptoms, Signs, Ill-Defined" catch-all (33%) is a labeling issue — patients coded with symptom-level ICD codes vs disease-level codes may have identical presentations.
- Surgical department routing could benefit from injury-specific features rather than vitals.
