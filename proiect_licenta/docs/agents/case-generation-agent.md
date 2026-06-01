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

## End-to-end benchmark (4 modes, same stay_ids)

`benchmarks/benchmark_pipeline_e2e.py` scores four prediction modes against
ground truth on the same cases:

1. **E2E (real crew)** — feeds only `narrative` into the full `ProiectLicenta`
   crew. The interactive tools are monkeypatched: `AskPatientTool` answers the
   parser's follow-ups from the structured fields (keyword-routed);
   `NurseDataCollectionTool` returns the case's deterministic nurse JSON
   (including the multi-reading `vital_trajectory`). The four prediction tools
   are wrapped to **tee their exact JSON output** into a capture dict —
   predictions are read from the model output, never parsed from LLM prose. Crew
   gating is honored (diagnosis/department only fire when the refined disposition
   = admit).
2. **Tool-direct** — calls the same crew tools directly with the exact tabular
   field values (no LLM), replicating the gating in Python, and passes the real
   `vital_trajectory_json` so the runtime feature path matches training. Isolates
   the NL-translation layer.
3. **Feature-vector** — predicts from the cached `build_features` rows + trained
   models (the existing `benchmark_nurse_v3` methodology), restricted to the
   sampled rows. Diagnosis/department scored **unconditionally** (no disposition
   gate) — the upper reference.
4. **Feature-vector-gated** — feature-vector, but diagnosis/department only count
   when the disposition model's own real-feature verdict is admit. This makes the
   feature-vector path apples-to-apples with the gated tool/E2E columns, so the
   table cleanly separates **gate cost** (feature-vector → gated), **runtime
   feature-degradation cost** (gated → tool-direct), and **LLM cost**
   (tool-direct → E2E).

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

## Results (20-case validation run)

Seed `20260529`, 13 admitted + 7 discharged, all from the held-out test splits.
All 20 narratives passed grounding on the first attempt (0 flagged); all 20 ran
through the crew with no errors.

**20 cases is a smoke/validation scale** — per-target deltas are noisy (one case
= 5pp on the 20-case targets, ≈ 7.7pp on the 13 admitted-GT diagnosis/department
cases). The *tabular* columns (tool-direct, feature-vector) are deterministic and
are the trustworthy signal; the **E2E column also carries LLM run-to-run jitter**
(the parser/triage agent calls are nondeterministic — E2E acuity moved 60%↔70%
across reruns of the *same* narratives). Read E2E as "tracks tool-direct, ±1–2
cases of LLM noise."

### The four columns (what each isolates)

| Column | Disposition gate? | Vital features | NL layer | Purpose |
|---|---|---|---|---|
| **feature-vector** | no (scores all admitted-GT) | real longitudinal | no | the existing `benchmark_nurse_v3` methodology — the upper reference |
| **feature-vector-gated** | yes (disposition model's own verdict) | real longitudinal | no | isolates the cost of the **gate** |
| **tool-direct** | yes (runtime gate) | runtime (trajectory or snapshot) | no | the real pipeline minus the LLM |
| **E2E** | yes (runtime gate) | runtime | yes | the full crew from free text |

`feature-vector → feature-vector-gated` = cost of the **disposition gate**;
`feature-vector-gated → tool-direct` = cost of **runtime feature degradation**;
`tool-direct → E2E` = cost of the **LLM/NLP layer**.

### Headline (after the multi-reading-vitals fix)

Accuracy on the same stay_ids (acuity/disposition over all 20; diagnosis/
department over the 13 admitted-GT cases):

| Target | tool-direct | feature-vector-gated | feature-vector | E2E |
|---|---|---|---|---|
| ESI acuity | 70.0% | 70.0% | 70.0% | 70.0% |
| Disposition (refined) | **80.0%** | 85.0% | 85.0% | 75.0% |
| Diagnosis @1 | **46.2%** | 53.8% | 69.2% | 38.5% |
| Diagnosis @3 | **69.2%** | 76.9% | 100.0% | 69.2% |
| Department @1 | 46.2% | 46.2% | 69.2% | 53.8% |
| Department @3 | **69.2%** | 76.9% | 100.0% | 69.2% |

Dx coverage (cases where the gated pipeline actually produced a diagnosis):
tool-direct **9/13**, E2E 9/13, feature-vector-gated 10/13, feature-vector 13/13.

### The runtime multi-reading-vitals fix (the big lever)

The original runtime collapsed the longitudinal-vitals features from a **single
snapshot** (`min=max=last=snapshot`, `delta=0`, `has_longitudinal_vitals=0`),
because the live nurse only took one reading. Measured effect on the 13 admitted
cases: mean disposition P(admit) **0.624 → 0.486**, pushing 4 true-admits below
the 0.5 gate (so the gated pipeline diagnosed only 6/13).

The fix (Stage B): the nurse tool now optionally collects a **second set of
readings**; a shared `build_longitudinal_block` (`src/proiect_licenta/vital_trajectory.py`,
the same aggregation as `train_nurse_v3`) builds **real** min/max/last/delta +
abnormal-reading counts when ≥1 trajectory is present, and `has_longitudinal_vitals=1`.
The doctor disposition + v3 tools accept it via a `vital_trajectory_json`
argument (empty → snapshot fallback, fully backward-compatible). The benchmark
and case generator feed the real `vitalsign.csv` trajectory.

**tool-direct, snapshot → trajectory:**

| Target | snapshot | trajectory | gated ceiling |
|---|---|---|---|
| Disposition | 65.0% | **80.0%** | 85.0% |
| Diagnosis @1 | 23.1% | **46.2%** | 53.8% |
| Diagnosis @3 | 46.2% | **69.2%** | 76.9% |
| Dx coverage | 6/13 | **9/13** | 10/13 |

The trajectory recovers almost the entire feature-degradation gap: tool-direct is
now within ~1 case of its true (gated) ceiling on every target. 16/20 cases had
real `vitalsign.csv` coverage; the other 4 fall back to snapshot (matching the
~5–7% of training stays with no longitudinal coverage).

### What each comparison says now

1. **Acuity is the clean NL signal.** All three tabular columns tie at 70.0%, so
   any E2E acuity movement is purely the LLM paraphrase + run-to-run jitter.
2. **Runtime vital degradation — mostly fixed.** `feature-vector-gated →
   tool-direct` on diagnosis@1 is now 53.8% → 46.2% (one case) vs the 53.8% →
   23.1% chasm before the fix. Disposition closed 65% → 80% against the 85%
   gated reference.
3. **The residual gap to the *ungated* feature-vector is the disposition gate
   itself**, by design: tool-direct/E2E can't diagnose a case the pipeline routes
   to discharge. tool-direct diagnosis@1 (46.2%) is close to its real ceiling
   (feature-vector-gated 53.8%); the remaining lift requires either **threshold
   tuning** (3 true-admits sit at P(admit) ≈ 0.47, just under the 0.5 gate) or a
   stronger disposition model — not anything the NL layer or the Case Generation
   Agent controls.

**NL-fidelity (parser-extracted vs tabular truth):** cases with parser output
20/20; age 19/20; gender 20/20; arrival transport 20/20; mean chief-complaint
token Jaccard **0.663**. Demographics survive the round-trip (recoverable via the
parser's follow-up questions); the chief-complaint text is the lossy channel
(~66% token overlap after lay paraphrase).

### Remaining levers (to push tool-direct/E2E toward the *ungated* feature-vector)

1. **Disposition threshold tuning (kept at 0.5 for now, by decision).** Lowering
   to ~0.45 recovers the 3 true-admits parked at P(admit) ≈ 0.47, lifting Dx
   coverage 9/13 → 12/13 at a small specificity cost on the discharge cases.
   The honest reporting at 0.5 stands; a threshold sweep is the documented next step.
2. **Scale beyond 20 cases.** Needed to make the 13-case diagnosis/department
   numbers statistically meaningful and to average out the LLM jitter on E2E.
3. **A stronger / recalibrated disposition model** would raise the
   feature-vector-gated ceiling itself (currently 10/13 admit recall on real
   features) — the one lever that lifts the gated diagnosis ceiling for *every*
   downstream column.
