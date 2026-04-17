# MIMIC-IV Dataset Reference

This document catalogs every MIMIC-IV table the project uses, every table it has **inspected but not yet used**, and — where relevant — how those unused tables could be integrated. The "unused" section is important: it captures discussion that informed [`future-work.md`](future-work.md) but belongs here with the data itself.

Source root on disk: `src/proiect_licenta/datasets/datasets_mimic-iv/`.

---

## Tables currently used

### `mimic-iv-ed/triage.csv`  **[PRIMARY — used for triage training]**

Columns: `subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint`.

- `chiefcomplaint`: Free-text, comma-separated (e.g., "Abd pain, Abdominal distention").
- `pain`: 0-10 patient-reported pain score.
- `acuity`: ESI 1-5 (1 = most severe, 5 = least severe).
- Vital signs (`temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`): recorded by the triage nurse for virtually all patients. **Used by the triage v2 model** — but only for ambulance/helicopter patients (~36%), matching real inference behavior where only EMS patients have vitals at triage handoff. Walk-in patients have their vitals masked to missing during training. Also used by the **Doctor v2** pipeline as the nurse-data source at training time.

### `mimic-iv-ed/edstays.csv`

Columns: `subject_id, hadm_id, stay_id, intime, outtime, gender, race, arrival_transport, disposition`.

- `disposition`: `HOME`, `ADMITTED`, `LEFT WITHOUT BEING SEEN`, etc.
- `arrival_transport`: `AMBULANCE` (36%), `WALK IN` (60%), `HELICOPTER` (0.1%), `OTHER`, `UNKNOWN`.
- `hadm_id`: Links to `services.csv` for department assignment.
- `race`: currently **not used** — see "Inspected but not yet used" below.
- `outtime` - `intime` gives ED length of stay — currently **not used**.

### `mimic-iv-ed/diagnosis.csv` + `mimic-iv-ed/files_created/categorized_diagnosis.csv`

- ICD-9/10 codes per ED stay with categories.
- `seq_num=1` is the primary diagnosis (used for training).
- Grouping via `python/group_icd.py` maps ICD codes to broad medical categories.
- 22 raw categories grouped into 14 for the doctor model (see [`agents/doctor-agent.md`](agents/doctor-agent.md)).

### `mimic-iv-ed/medrecon.csv`  **[used for Doctor v2]**

~3M rows: medication reconciliation (pre-admission meds the patient was already taking when they arrived at the ED). Used to build the 9 binary medication category flags consumed by Doctor v2. The `etcdescription` field drives keyword matching against the 120+ entry drug-name map.

### `mimic-iv/hosp/patients.csv`

Columns: `subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod`.

- Age computed as: `anchor_age + (visit_year - anchor_year)`.

### `mimic-iv/hosp/services.csv`  **[used for department labels]**

- Maps `hadm_id` to hospital service/department.
- 19 unique services grouped into 11 for the doctor model.
- First service per admission used (initial department assignment).

### `mimic-iv/hosp/diagnoses_icd.csv` + `d_icd_diagnoses.csv`

Hospital-wide ICD diagnosis codes and the ICD code dictionary. Used indirectly through the categorization pipeline.

---

## Tables inspected but not yet used

These tables were opened and sampled during scoping. Each represents a concrete integration opportunity tracked in [`future-work.md`](future-work.md).

### `mimic-iv-ed/vitalsign.csv`  — 1.4M rows

**Longitudinal vital signs** recorded at multiple points during the ED stay (not just at triage). Columns include the same six vitals plus `rhythm` (cardiac rhythm, e.g., sinus, atrial fibrillation).

Why this matters:
- Current Doctor v2 uses a **single** vitals snapshot from triage. Trends (is heart rate rising? is O2 dropping?) carry much more diagnostic signal than a point-in-time value.
- The `rhythm` field is not in `triage.csv` at all — it's new categorical signal (likely very predictive for cardiac/neuro routing).
- Temporal-leakage caveat: only readings taken **before** the diagnosis/department decision point should be used for training. Measurements taken late in the stay would leak the outcome.

### `mimic-iv-ed/pyxis.csv`  — ED medications dispensed during stay

**Meds given during the ED visit** (not pre-admission). This is different from `medrecon.csv` which captures home meds.

Why this matters:
- Could be a very strong signal (e.g., "patient was given nitroglycerin in the ED" → cardiac).
- **Leakage risk:** meds dispensed reflect the working diagnosis the ED team already had. Using them as model input is essentially peeking at the answer. Would only be safe if restricted to the first N minutes after arrival (before assessment was complete) or used strictly as validation signal, not a feature.

### `mimic-iv/hosp/admissions.csv`

Hospital admission records. Contains potentially useful columns:
- **Insurance** type (may correlate with presentation patterns).
- **Marital status**, **language**.
- **Hospital expire flag** (in-hospital mortality) — a candidate target for a future "severity" model.
- Admission time / discharge time — gives length-of-stay labels.

### `mimic-iv-ed/edstays.csv` — `race` and `outtime` columns

Already loaded for other fields; `race` and `outtime` are not yet used. Adding race/ethnicity features is ethically sensitive (documented bias in ED presentation patterns) but could be explored as part of a fairness audit.

### `mimic-iv/note/discharge.csv`  — ~3.3 GB

**Discharge summaries** — the physician's final narrative for each admission. Contains:
- The actual clinical narrative (history of present illness, physical exam findings, working diagnosis, plan).
- The discharge diagnosis in natural language.

Why this matters:
- The discharge diagnosis is a **much cleaner label** than the ICD-coded primary diagnosis (which suffers from the 33% "Symptoms, Signs, Ill-Defined" catch-all). Extracting a normalized diagnosis from the discharge summary could produce higher-quality training labels.
- The notes also contain past medical history (PMH), allergies, and social history — a rich source for additional structured features.
- **Leakage caveat:** discharge summaries are written at the *end* of the stay. They must only be used to produce **labels** or **patient-history features from prior encounters**, never as input for predicting the current encounter's diagnosis.

### `mimic-iv/note/radiology.csv`  — ~2.7 GB

**Radiology reports** (CT, MRI, X-ray interpretations).

Why this matters:
- Imaging findings are often the decisive signal for department routing (e.g., "subdural hematoma" → NSURG).
- **Leakage caveat (same as discharge):** imaging ordered during this stay reflects the working diagnosis. Would only be safe to use if (a) restricted to imaging ordered very early, or (b) used as PMH signal from *prior* encounters.

### `mimic-iv/note/discharge_detail.csv` + `radiology_detail.csv`

Detail/metadata companions to the main note files (section splits, timestamps). Useful if you need to carve notes into usable sub-sections (e.g., pull only the "History of Present Illness" block for training).

### `mimic-iv/icu/`  — **explicitly not used**

The project focuses entirely on the Emergency Department pathway.

---

## How the data feeds each model

| Artifact | Training tables | Feature count |
|---|---|---|
| Triage acuity model (v4) | `triage.csv` (vitals for ambulance/helicopter only) + `edstays.csv` + `patients.csv` | 2051 |
| Triage disposition model (v4) | Same + `predicted_acuity` | 2052 |
| Doctor v1 diagnosis | Same + `services.csv` (labels via diagnoses) + `predicted_acuity` + `predicted_disposition` | 2025 |
| Doctor v1 department | Same + `predicted_diagnosis` | 2026 |
| Doctor v2 diagnosis | All of v1 + `triage.csv` vitals (20 features) + `medrecon.csv` (11 features) | 2056 |
| Doctor v2 department | Same + `predicted_diagnosis` | 2057 |

Row counts (after cleaning):
- Triage v4 training: 418K cleaned -> 334K train / 83K test (80/20 split, full dataset).
- Doctor training: 157K admitted -> 100K sampled -> 80K train / 20K test.
