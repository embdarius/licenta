# Case Generation Agent (Phase 4 — Text Generation)

The Case Generation Agent is an **offline, benchmark-only** agent. It does not
sit in the live patient pipeline. Its job is to turn a real MIMIC-IV tabular ED
row into a realistic natural-language patient description — the raw free-text
the NLP Parser would see at intake — plus every other input the crew collects
interactively, **grounded strictly in the row** (it invents nothing).

These synthetic cases let us measure something the per-model benchmarks can't:
**how much accuracy the pipeline loses when a patient is described in natural
language and parsed by the LLM, versus handed to the models as clean tabular
columns.**

- **Type:** CrewAI agent (Gemini), run as a dedicated offline crew.
- **Config:** `config/case_generation_agents.yaml` (`case_generator`) and
  `config/case_generation_tasks.yaml` (`generate_case_task`) — **separate** from
  the live crew's `agents.yaml`/`tasks.yaml` (see "Why separate config" below).
- **Code:** `src/proiect_licenta/case_generation.py`.
- **Entry point:** `uv run generate_cases [--limit N]`.
- **Benchmark:** `benchmarks/benchmark_pipeline_e2e.py`.
- **Roadmap:** this is Phase 4 in [`../future-work.md`](../future-work.md).

---

## What it generates

For each sampled `stay_id`, a case bundle (persisted to
`data/derived/synthetic_cases/cases.json`, gitignored — MIMIC DUA):

- **`narrative`** — the LLM's only free output: a 1-3 sentence first-person
  description of the chief complaints (lay paraphrase of the clinical shorthand)
  with age / arrival / pain woven in. This is what the NLP Parser receives.
- **`triage_inputs`** — chief complaints, pain, age, gender, arrival transport,
  EMS vitals (ambulance/helicopter only), `prior_history`, `n_prior_admissions`.
- **`nurse_inputs`** — the 6 vitals (measured snapshot), cardiac rhythm,
  medication names, `prior_history`, `n_prior_admissions`.
- **`ground_truth`** — `acuity`, `disposition`/`admitted`, and (admitted only)
  `diagnosis_group` + `service_group`.

**Only the narrative is LLM-generated.** Everything else is deterministic from
the tabular row, so invention risk is confined to the narrative and caught by
the grounding validator. The structured fields drive the harness's scripted
answers to the parser's follow-up questions and the nurse prompts.

### Tabular sources (every input the crew consumes at live inference)

| Field | Source |
|---|---|
| chief_complaints | `triage.csv:chiefcomplaint` (raw, comma-separated) |
| pain_score | `triage.csv:pain` |
| age | `patients.csv` (anchor_age + visit_year − anchor_year) |
| gender / arrival_transport | `edstays.csv` |
| vitals (temp/hr/rr/o2/sbp/dbp) | `triage.csv` snapshot (measured → value; missing → patient "doesn't know") |
| rhythm | `vitalsign.csv` (first reading in `[intime, +4h]`) |
| medications_raw | `medrecon.csv` home-med **names** (what a patient reports) |
| prior_history | reverse-mapped from the loader's real `pmh_*` flags to canonical condition phrases |
| n_prior_admissions | loader `n_prior_admissions` |

### PMH reverse-mapping

The loader produces 13 binary `pmh_<category>` flags per stay (from prior
discharge-note PMH sections + prior-ICD fallback — see Doctor Change 1).
`prior_history_text()` maps each *fired* category to a canonical patient-speak
phrase (e.g. Circulatory → "congestive heart failure", Endocrine → "diabetes",
Respiratory → "copd"). Each phrase is verified at import time to **round-trip**
through `pmh_vocab.flags_from_text` back to its category, so the triage /
disposition / v3 tools re-derive exactly the flag the loader set. This is
grounding (a deterministic lookup), not invention — it slightly idealizes a
patient's recall of their own history (a documented caveat).

---

## Grounding validator

`validate_grounding(narrative, fields)` enforces the "no invented clinical
facts" rule. It rejects (→ regenerate, up to 3 attempts, then flag):

- any stated **age** that doesn't match the row;
- any stated **pain** `X/10` that doesn't match the row;
- any **fabricated clinical measurement** in the opening narrative
  (blood-pressure-like `NNN/NN`, `bpm`, `mmHg`, temperature/O2 tokens) — those
  belong to the nurse stage, not the patient's opening description.

It deliberately does **not** score complaint wording: faithful re-extraction of
the complaint from the paraphrased narrative is exactly what the E2E benchmark
measures, so penalizing paraphrase here would defeat the purpose.

---

## Cohort sampling

Stratified **admit/discharge** (default 13 admitted + 7 discharged), drawn from
the held-out test splits:

- The disposition test split (`random_state=42`, stratified on `admitted`) —
  same split as `benchmark_doctor_disposition.py`.
- Admitted picks are additionally restricted to the **doctor-v3 nurse test
  split** intersection (and to non-catch-all rows), so diagnosis/department are
  genuinely held out *and* labeled.

**Caveat:** admitted picks are held out for the disposition + doctor-v3 models,
but their `acuity` rows are not guaranteed held out from the triage model. At a
20-case validation scale this is acceptable and documented.

---

## End-to-end benchmark (3 modes, same stay_ids)

`benchmarks/benchmark_pipeline_e2e.py` scores three prediction modes against
ground truth on the same cases:

1. **E2E (real crew)** — feeds only `narrative` into the full `ProiectLicenta`
   crew. The interactive tools are monkeypatched: `AskPatientTool` answers the
   parser's follow-ups from the structured fields (keyword-routed);
   `NurseDataCollectionTool` returns the case's deterministic nurse JSON. The
   four prediction tools are wrapped to **tee their exact JSON output** into a
   capture dict — predictions are read from the model output, never parsed from
   LLM prose. Crew gating is honored (diagnosis/department only fire when the
   refined disposition = admit).
2. **Tool-direct** — calls the same crew tools directly with the exact tabular
   field values (no LLM), replicating the gating in Python. Isolates the
   NL-translation layer (same models, cascade, PMH parsing as E2E).
3. **Feature-vector** — predicts from the cached `build_features` rows + trained
   models (the existing benchmark methodology), restricted to the sampled rows.

Targets: ESI acuity, refined disposition, diagnosis top-1/top-3, department
top-1/top-3. Output includes a per-target accuracy table across modes, a
per-case dump, and an **NL-fidelity report** (parser-extracted complaints/fields
vs the tabular truth).

### What the E2E test actually isolates

Because the harness answers the parser's follow-ups from the true fields,
demographics / vitals / PMH are always recoverable. The dominant measurable E2E
loss is therefore **chief-complaint text translation + inter-agent field
handoff** — which is the right thing to isolate.

---

## Why separate config files

CrewAI's `@CrewBase` maps **every** task in a `tasks.yaml` to an `@agent` method
on the crew class. If `generate_case_task` lived in the shared `tasks.yaml`, the
live `ProiectLicenta` crew (which has no `case_generator` agent) would fail with
`KeyError: 'case_generator'`, and vice-versa. The generation crew therefore uses
dedicated `case_generation_{agents,tasks}.yaml` files.

---

## Runtime / cost

`generate_cases` is heavy: it runs the disposition + doctor-v3-nurse loaders
once (incl. the `discharge.csv` PMH stream) to sample faithfully and capture
real PMH + ground truth, and caches the per-stay feature rows to
`data/derived/synthetic_cases/sampled_features.pkl`. The benchmark then runs the
fast modes off the cached JSON; only the E2E mode calls the LLM (~6 agent turns
× N cases). Use `--skip-feature-vector` / `--skip-e2e` and `--limit N` for quick
iteration.

---

## Results (first 20-case validation run)

Seed `20260529`, 13 admitted + 7 discharged, all from the held-out test splits.
All 20 narratives passed grounding on the first attempt (0 flagged); all 20 ran
through the crew with no errors.

**20 cases is a smoke/validation scale** — per-target deltas are noisy (one case
= 5pp on the 20-case targets, ≈ 7.7pp on the 13 admitted-GT diagnosis/department
cases). Read the deltas as directional, not precise.

Accuracy on the same stay_ids (acuity/disposition over all 20; diagnosis/
department over the 13 admitted-GT cases):

| Target | tool-direct | feature-vector | E2E (NL crew) |
|---|---|---|---|
| ESI acuity | 70.0% | 70.0% | **60.0%** |
| Disposition (refined) | 65.0% | 85.0% | 70.0% |
| Diagnosis @1 | 23.1% | 69.2% | 30.8% |
| Diagnosis @3 | 46.2% | 100.0% | 53.8% |
| Department @1 | 38.5% | 69.2% | 46.2% |
| Department @3 | 46.2% | 100.0% | 53.8% |

**The three columns do not measure the same thing — read them carefully:**

- **feature-vector** is the *unconditional per-model* methodology used by the
  existing benchmarks (`benchmark_nurse_v3.py`): every admitted-GT case is scored
  by the diagnosis/department models directly, regardless of what the disposition
  model would have decided. It produced a Dx for **13/13** admitted cases.
- **tool-direct** and **E2E** run the *gated pipeline*: diagnosis/department only
  fire when the pipeline's own refined disposition says admit. tool-direct
  produced a Dx for **6/13**, E2E for **7/13** — the other cases were routed to
  discharge and counted as diagnosis/department misses. This is why their
  diagnosis/department numbers are far below feature-vector's: most of that gap is
  the **disposition gate**, not the diagnosis model and not the LLM.

**What each comparison actually isolates:**

1. **Acuity is the clean NL-translation signal.** tool-direct and feature-vector
   are *identical* (70.0%) — the only target where the two tabular paths tie — so
   the E2E drop to 60.0% (−10pp = 2 cases) is attributable to the LLM paraphrase
   changing the chief-complaint text the acuity model keys on.

2. **The big diagnosis/department gap is the disposition gate + runtime vital
   degradation, not the LLM.** The disposition *tool* (which rebuilds longitudinal
   vitals from a single snapshot — `has_longitudinal_vitals = 0`) is markedly more
   discharge-prone than the feature-vector path (which uses the real
   `vitalsign.csv` trajectories): disposition accuracy 65–70% vs 85%. Because the
   gated pipeline under-admits, it never gets to diagnose ~half the admitted-GT
   cases. This is a genuine, documentable finding about **runtime feature
   degradation**, surfaced precisely because the harness compares the gated tool
   path against the unconditional feature path.

3. **The LLM layer adds little beyond what the gated runtime pipeline already
   costs.** E2E tracks tool-direct closely and is even *slightly higher* on
   diagnosis (30.8% vs 23.1%) and department (46.2% vs 38.5%) — driven by E2E
   admitting one more case (7 vs 6), i.e. noise at this scale. So once you hold the
   pipeline (gating + runtime vitals) fixed, swapping exact tabular values for the
   LLM-parsed narrative mostly costs the acuity −10pp above; diagnosis/department
   are dominated by the gate.

**NL-fidelity (what the parser re-extracted from the narrative vs the tabular truth):**

| Field | Correct |
|---|---|
| cases with parser output | 20/20 (no crew failures) |
| age | 18/20 |
| gender | 20/20 |
| arrival transport | 20/20 |
| mean chief-complaint token Jaccard | **0.663** |

Demographics survive the round-trip almost perfectly (the harness answers the
parser's follow-up questions from the true fields, so they are recoverable). The
**chief-complaint text is the lossy channel** — the LLM paraphrases clinical
shorthand into lay language, so only ~66% of normalized complaint tokens overlap
with the raw MIMIC text. That lexical drift is what drives the acuity dip.

**Follow-ups this run surfaced:**

1. **Scale up.** 20 cases is too few to separate signal from noise on the 13-case
   diagnosis/department targets — a few hundred cases would tighten every estimate.
2. **The runtime disposition tool under-admits vs its own training feature
   distribution** (snapshot-only longitudinal fallback). This is the single
   biggest driver of the gated-vs-unconditional gap and is worth its own
   investigation — it is a property of the *live pipeline*, independent of the
   Case Generation Agent.
3. Consider also reporting a *gated* feature-vector variant (apply the disposition
   model's own verdict before scoring diagnosis/department) so the feature-vector
   column is directly comparable to the gated tool/E2E columns.
