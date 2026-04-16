# NLP Parser Agent

The first agent in the pipeline. It converts free-text patient descriptions into the structured JSON payload every downstream ML model expects.

---

## Role

- **Type:** LLM-based (Gemini 2.5 Flash via CrewAI)
- **Purpose:** Take free-text patient input and output a structured record that the Triage Agent can feed directly into XGBoost.
- **Config:** `src/proiect_licenta/config/agents.yaml` (`nlp_parser`) and `tasks.yaml` (`parse_symptoms_task`)

---

## Input / Output

**Input (example):**
> "I'm a 65 year old man, I have severe chest pain and difficulty breathing, I came by ambulance"

**Output (JSON):**
```json
{
  "chief_complaints": ["chest pain", "dyspnea"],
  "pain_score": 8,
  "age": 65,
  "gender": "male",
  "arrival_transport": "ambulance"
}
```

Fields:
- `chief_complaints` — list of normalized complaint phrases (clinical short-form; e.g. "dyspnea" rather than "difficulty breathing").
- `pain_score` — integer 0-10 patient-reported.
- `age` — integer.
- `gender` — `"male"` or `"female"` (MIMIC-IV's two categories).
- `arrival_transport` — one of `ambulance`, `walk_in`, `helicopter`, `other`, `unknown`.

---

## Tools

### `AskPatientTool`
Interactive follow-up tool. When information is missing from the free-text input (pain score, age, gender, arrival method), the agent calls this tool to ask the patient via stdin.

Implemented in `src/proiect_licenta/tools/ask_patient_tool.py`.

---

## Design Notes

- The NLP Parser is deliberately the *only* LLM agent in the pipeline. Every downstream decision (acuity, disposition, diagnosis, department) uses supervised ML on MIMIC-IV for reliable, auditable predictions. See [`../future-work.md`](../future-work.md) for why.
- Normalization is important: chief complaints must map to MIMIC-IV's vocabulary so TF-IDF and severity-prior lookups work. The Parser is prompted to produce clinical short-forms (e.g. "chest pain", "abd pain", "sob") that overlap with MIMIC-IV's training corpus.
- The chief-complaint preprocessing pipeline (lowercase, separator normalization, 45+ abbreviation expansions, TF-IDF, severity priors) is described in [`../architecture.md`](../architecture.md).
