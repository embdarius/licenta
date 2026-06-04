# Nurse Agent

The Nurse Agent is an interactive (non-ML) agent that collects vital signs and medication history from the patient between the Doctor's initial assessment (v1) and enhanced reassessment (v2).

- **Type:** CrewAI agent with `NurseDataCollectionTool` (interactive data collection via stdin).
- **Role:** Emergency Department Nurse.
- **Config:** `src/proiect_licenta/config/agents.yaml` (`nurse_agent`) and `tasks.yaml` (`nurse_data_collection_task`).
- **Placement in pipeline:** Runs after `doctor_assessment_task`, before `doctor_reassessment_task`.

---

## What it collects

**Vital signs (6 numeric fields):**
- `temperature` (F)
- `heartrate` (bpm)
- `resprate` (breaths/min)
- `o2sat` (%)
- `sbp` (systolic blood pressure, mmHg)
- `dbp` (diastolic blood pressure, mmHg)

Blood pressure accepts the familiar `"120/80"` format and is split into SBP / DBP.

**Medication list:** A free-text list of medications the patient is currently taking.

---

## Interactive flow

Implemented in `src/proiect_licenta/tools/nurse_tool.py` (`NurseDataCollectionTool`).

1. The tool receives `patient_context` summarizing the triage + initial assessment.
2. It collects **one or more chronological reading sets** (each: the 6 vitals + cardiac rhythm + an optional timestamp), looping with an "add another set?" prompt — see *Multi-reading collection* below.
3. Then it prompts for the medication list, past medical history (chronic conditions), and approximate prior-admission count (v3 additions).
4. It returns a JSON payload with:
   - `vital_signs` dict — the **first (arrival) reading set**, the single-snapshot fallback the doctor tools use for the triage cascade.
   - `vital_trajectory` dict — the full chronological series `{vital: [floats], "rhythm": [strs]}`; the v3 + disposition tools build real `min/max/last/delta` + abnormal-reading counts + the rhythm mode/`irregular` flags from it. (The v2 tool ignores it.)
   - `rhythm` — the first reading's cardiac rhythm string (v2 + display).
   - `medications_raw` — free-text list, classified downstream.
   - `prior_history` + `n_prior_admissions` — self-reported PMH (v3; overridden by the MRN/EHR lookup when available).

Helper utilities:
- `_parse_numeric` — tolerant float parser (strips units, handles commas).
- `_parse_bp` — splits `"120/80"` style input.
- `_parse_timestamp` — parses an optional reading time (clock time `14:30` → minutes since midnight, or relative minutes `15`/`+15`); blank/`now`/skip → `None`.
- `_collect_reading_round` — collects one chronological reading set (6 vitals + rhythm + optional timestamp).
- `SKIP_WORDS` — set of tokens treated as "I don't know" / skip (empty string, `skip`, `unknown`, `idk`, `i don't know`, etc.).

### Multi-reading collection (2026-06-03)

A real ED charts vitals repeatedly over a stay, and the v3/disposition models were trained on **all** `vitalsign.csv` readings within the first 4h — so a single snapshot under-represents the patient. The tool therefore collects a variable-length series:

- The **first set** is the primary/arrival reading (populates `vital_signs`).
- An **"add another set?"** loop appends further sets (capped at 24) — each with an **optional per-reading timestamp**.
- **Ordering:** if *every* set carries a timestamp, the series is **sorted by it** (so out-of-order entry is fine); if *any* set lacks one, the tool keeps **entry order** as the chronological proxy (the no-timestamp fallback — all-or-nothing, to avoid interleaving). No reading is ever dropped — the training 4h window exists to prevent *leakage*, which doesn't apply at live prediction time, so windowing is unnecessary at runtime.
- The wire format is unchanged (`vital_trajectory` value-lists), so the aggregator (`vital_trajectory.build_longitudinal_block`) and the doctor tools required **no** changes — timestamps are consumed at collection time and only the ordered values are emitted.
- This is **live-runtime only**; the offline benchmark already supplies the full in-window series via the case generator's `pull_vital_trajectory`.

---

## Partial data handling

Every field can be skipped ("I don't know" or just Enter). Missing values are handled downstream in a principled way:

- **Per vital:** A `<vital>_missing` binary flag is set, and the value is imputed with the population median computed from training data (`temp=98.1F, HR=84, RR=18, O2=98%, SBP=134, DBP=78`).
- **All vitals missing:** The model still has the `_missing` flags as signal and falls back entirely on the triage features.
- **Medications:** If the patient provides none, `meds_unknown=1` is set and all 9 category flags are zeroed. If the patient provides medications, they are classified via the 120+ entry drug-name map plus keyword matching (see [`doctor-agent.md`](doctor-agent.md)).

Physiological clipping is applied to any out-of-range values a patient reports (e.g., temperature below 90F or above 110F is clipped to that range).

---

## Why a dedicated agent?

Two reasons:
1. **Realism** — in a real ED, the nurse is a separate role from the triage clerk and the doctor. Modeling that as a distinct agent makes the pipeline's structure mirror clinical workflow.
2. **Demonstrable delta** — by collecting nurse data between the two Doctor phases, the system can directly compare v1 (without nurse data) and v2 (with nurse data) predictions on the same patient. This is the headline contribution of Phase 3.

---

## Related

- The output contract expected by the Doctor v2 tool (`DoctorPredictionToolV2`) lives in [`doctor-agent.md`](doctor-agent.md).
- See [`../future-work.md`](../future-work.md) for the analysis of why v2 gains were modest (+2.3pp diagnosis, +5.9pp department) and what nurse-related extensions could close that gap (longitudinal vitals, pyxis meds, past-medical-history from prior notes).
