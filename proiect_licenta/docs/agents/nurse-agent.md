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
2. It prompts the patient via stdin for each vital sign, one at a time.
3. Then it prompts for the medication list.
4. It returns a JSON payload with:
   - `vital_signs` dict (raw values plus any that were reported)
   - `medications_raw` list (free-text entries, classified downstream by `DoctorPredictionToolV2`)

Helper utilities:
- `_parse_numeric` — tolerant float parser (strips units, handles commas).
- `_parse_bp` — splits `"120/80"` style input.
- `SKIP_WORDS` — set of tokens treated as "I don't know" / skip (empty string, `skip`, `unknown`, `idk`, `i don't know`, etc.).

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
