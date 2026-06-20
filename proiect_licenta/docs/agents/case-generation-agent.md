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

- **Type:** CrewAI agent (Gemini by default; **switchable to self-hosted MedGemma** via `LLM_BACKEND` — see [`../llm-backend.md`](../llm-backend.md), incl. `generate_cases --backend` and the Experiment B grounding comparison).
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
- **(added 2026-06-20) clinical-jargon leakage** — the narrative must stay lay
  (no `dyspnea`/`syncope`/`hematuria`/`n/v`/`s/p`/…); guards the risk that a
  medical-domain generator (MedGemma) pre-clinicalizes the input and unfairly
  inflates parser accuracy (the §7 anti-pattern in `../llm-backend.md`);
- **(added 2026-06-20) completeness** — every comma-separated complaint must be
  voiced (no dropped secondary complaints). Uses a separate `_COMPLETENESS_ANCHORS`
  set, **deliberately independent of the parser-side `clinicalize_complaint` map**,
  so generation isn't biased toward the eval-time vocabulary. Conservative: only
  common families, 0/150 false positives on the existing set.

It deliberately does **not** score complaint *wording* (lay paraphrase is the
point — faithful re-extraction is what the E2E benchmark measures); it only
enforces that no complaint is *dropped* and no clinical jargon leaks in.

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

## Results (20-case validation run, disposition threshold 0.40)

Seed `20260529`, 13 admitted + 7 discharged, all from the held-out test splits.
All 20 narratives passed grounding on the first attempt (0 flagged); all 20 ran
through the crew with no errors. **All gated columns use the live pipeline's
`DECISION_THRESHOLD = 0.40`** (tuned from 0.50 — see
[doctor-agent.md](doctor-agent.md#whats-next)).

**20 cases is a smoke/validation scale** — per-target deltas are noisy (one case
= 5pp on the 20-case targets, ≈ 7.7pp on the 13 admitted-GT diagnosis/department
cases). The *tabular* columns (tool-direct, feature-vector, feature-vector-gated)
are deterministic and are the trustworthy signal; the **E2E column also carries
LLM run-to-run jitter** (the parser/triage agent calls are nondeterministic —
E2E acuity moved 60%↔70% across reruns of the *same* narratives). Read E2E as
"tracks tool-direct, ±1–2 cases of LLM noise."

### The four columns (what each isolates)

| Column | Disposition gate? | Vital features | NL layer | Purpose |
|---|---|---|---|---|
| **feature-vector** | no (scores all admitted-GT) | real longitudinal | no | the existing `benchmark_nurse_v3` methodology — the upper reference |
| **feature-vector-gated** | yes (disposition model's verdict @0.40) | real longitudinal | no | isolates the cost of the **gate** |
| **tool-direct** | yes (runtime gate @0.40) | runtime (trajectory or snapshot) | no | the real pipeline minus the LLM |
| **E2E** | yes (runtime gate @0.40) | runtime | yes | the full crew from free text |

`feature-vector → feature-vector-gated` = cost of the **disposition gate**;
`feature-vector-gated → tool-direct` = cost of **runtime feature degradation**;
`tool-direct → E2E` = cost of the **LLM/NLP layer**.

### Headline (multi-reading-vitals fix + 0.40 threshold)

Accuracy on the same stay_ids (acuity/disposition over all 20; diagnosis/
department over the 13 admitted-GT cases):

| Target | tool-direct | feature-vector-gated | feature-vector | E2E |
|---|---|---|---|---|
| ESI acuity | 70.0% | 70.0% | 70.0% | 65.0% |
| Disposition (refined) | **90.0%** | 95.0% | 95.0% | 85.0% |
| Diagnosis @1 | **61.5%** | 69.2% | 69.2% | 53.8% |
| Diagnosis @3 | **84.6%** | 92.3% | 100.0% | 76.9% |
| Department @1 | 53.8% | 61.5% | 69.2% | 61.5% |
| Department @3 | **84.6%** | 92.3% | 100.0% | 76.9% |
| ICD exact @1 | 23.1% | 23.1% | 23.1% | 7.7% |
| ICD exact @5 | **61.5%** | 46.2% | 46.2% | 53.8% |
| ICD flat@10 | **69.2%** | 69.2% | 69.2% | 61.5% |
| ICD union | 69.2% | 69.2% | 69.2% | 69.2% |

Dx coverage (cases where the gated pipeline produced a diagnosis): tool-direct
**11/13**, E2E 11/13, feature-vector-gated 12/13, feature-vector 13/13.

The last four rows score the **Stage-2 exact-ICD resolver** (doctor+nurse / v3,
3-char rollup) through the same modes — added 2026-06-15; see
[Stage-2 exact-ICD end-to-end](#stage-2-exact-icd-end-to-end-doctornurse) below.
The runtime tool resolves over top-5 categories / k_per_cat=5 / k_flat=10, the same
parameters as `benchmark_icd_resolution.py`, so `ICD union` ≈ that benchmark's *E2E
union* and `ICD flat@10` ≈ its *E2E flat-10*. `tool_direct_lookup` (omitted from
this table) scored @1/@5/flat@10/union = 23.1% / 53.8% / 69.2% / 69.2%.

### Two fixes drove this — runtime vitals + threshold

**(a) Multi-reading-vitals fix (Stage B).** The original runtime collapsed the
longitudinal-vitals features from a **single snapshot** (`min=max=last=snapshot`,
`delta=0`, `has_longitudinal_vitals=0`), because the live nurse only took one
reading. Measured effect on the 13 admitted cases: mean disposition P(admit)
**0.624 → 0.486** in the snapshot path. The fix: the nurse tool collects **one or
more additional reading sets** (an "add another set?" loop, each optionally
timestamped and sorted chronologically; 2026-06-03); a shared `build_longitudinal_block`
(`src/proiect_licenta/vital_trajectory.py`, the same aggregation as
`train_nurse_v3`) builds **real** min/max/last/delta + abnormal-reading counts
when ≥1 trajectory is present (`has_longitudinal_vitals=1`). The doctor
disposition + v3 tools accept it via a `vital_trajectory_json` arg (empty →
snapshot fallback, fully backward-compatible). 16/20 cases had real
`vitalsign.csv` coverage; the other 4 fall back to snapshot.

**(b) Disposition threshold 0.50 → 0.40.** The sweep
(`benchmarks/sweep_disposition_threshold.py`, max F1 / max Youden on the 83K test
split) lowered the admit cutoff, which recovers true-admits the 0.5 gate was
dropping. Combined effect on **tool-direct**, across both fixes:

| Target | snapshot @0.50 | trajectory @0.50 | trajectory @0.40 |
|---|---|---|---|
| Disposition | 65.0% | 80.0% | **90.0%** |
| Diagnosis @1 | 23.1% | 46.2% | **61.5%** |
| Diagnosis @3 | 46.2% | 69.2% | **84.6%** |
| Dx coverage | 6/13 | 9/13 | **11/13** |

tool-direct is now within **1 case** of its true (gated) ceiling (12/13) on
coverage, and within ~8pp on diagnosis@1.

### What each comparison says now

1. **Acuity is the clean NL signal.** All three tabular columns tie at 70.0%, so
   the E2E acuity drop (65.0%) is purely LLM paraphrase + run-to-run jitter.
2. **Runtime vital degradation — essentially closed.** `feature-vector-gated →
   tool-direct` on diagnosis@1 is now 69.2% → 61.5% (one case) vs the 53.8% →
   23.1% chasm before the vitals fix. Disposition: tool-direct 90.0% vs the 95.0%
   gated reference (a single discharge case the runtime over-admits).
3. **The residual gap to the *ungated* feature-vector is the disposition gate
   itself**, by design: tool-direct/E2E can't diagnose a case the pipeline routes
   to discharge (11/13 covered vs 13/13). Closing it further requires a stronger
   disposition model (raising the gated ceiling), not the NL layer.

### Stage-2 exact-ICD end-to-end (doctor+nurse)

Added 2026-06-15: the benchmark now also scores the [Stage-2 exact-ICD resolver](doctor-agent.md#stage-2--exact-icd-resolution-within-categories-shipped-2026-06-11)
(doctor+nurse model, 3-char rollup) on the same 13 admitted cases. For the live E2E
column it reads the v3 tool's `diagnosis_prediction.exact_diagnoses` block; for
`feature_vector` it runs the resolver on the cached features
(`_fv_exact_icd` reuses `icd_resolution.physio_matrix` — all 28 physio + 14 PMH
columns are present in `nurse_v3_features`). Metrics mirror
`benchmark_icd_resolution.py`: **@1/@5** from the flat list, **flat@10** (≙ that
benchmark's *E2E flat-10*) and **union** (≙ *E2E union*, the true code anywhere in
top-5 cats × top-5 codes). Coverage: exact-ICD produced for 11–13/13 per mode.

**Caveat first: n=13, so 1 case = 7.7pp** — read these as ±1–2 case confirmations,
not precise estimates; the 20,420-row `benchmark_icd_resolution.py` is the
authoritative source.

1. **`union` is mode-invariant (9/13 = 69.2% everywhere, incl. E2E).** Union is a
   set-membership test over the candidate pool — it ignores the `P(cat)` ordering and
   depends only on the true category being in the top-5 (forgiving) and the within-
   category top-5, which is driven by **complaint text** (identical in tabular modes,
   only paraphrased in E2E). So the resolver's value **survives the full free-text →
   crew → nurse pipeline** — the headline result. It matches `benchmark_icd_resolution.py`
   (full-set v3-nurse E2E union 62.85%; the 13-case sample reads a touch higher).
2. **`@1`/`@5`/`flat@10` wobble because the flat list ranks by `P(category)·score`** —
   the category-softmax *shape* is the lever that differs across modes:
   - **`feature_vector` @5 is the *lowest* (6/13) despite the best Stage-1**
     (diagnosis@1 69.2%, @3 100%). A cleaner feature vector → **peakier softmax** →
     `P(cat)` suppresses codes from correctly-but-secondarily ranked categories, so a
     true code in the #2/#3 category drops below flat rank 5. The degraded
     (`tool_direct`) and NL-parsed (`e2e`) vectors give flatter softmaxes that surface
     those codes — hence `tool_direct` 8/13 and `e2e` 7/13 *beat* it on @5. Proof it's
     pure re-ranking: **union is identical (9/13)** for all three. A better category
     model can mean worse flat exact-code recall.
   - **`tool_direct_lookup` @5 (7/13) < `tool_direct` (8/13).** The EHR lookup lifts
     the coarse targets (diagnosis@1 +7.7pp, coverage 11→12) but its added PMH/prior
     signal re-weighted the flat order, sliding one true code from rank ≤5 to 6–10
     (flat@10 unchanged at 9/13). The lookup pays off on disposition/category/coverage,
     not on the fine exact-code ordering — the resolver ranks on complaint text, which
     the lookup doesn't change.
3. **E2E `@1` collapses 3/13 → 1/13.** The flat #1 is dominated by complaint text
   cosine; the parser's paraphrase perturbs the TF-IDF query, hitting the most precise
   metric hardest. E2E recovers toward the tabular modes as the window widens
   (@5 → flat@10 → union).
4. **Some "misses" are ICD-9/10 splits of the same disease** (e.g. truth `518` ICD-9
   vs E2E-predicted `J18` ICD-10, both pneumonia), so true clinical recall is somewhat
   higher than measured — the deferred CCSR-unification item.

**Takeaway:** the resolver is an **advisory differential** (a short candidate list),
not a single-code prediction — so `union`/`flat@10` are the honest headline, and they
hold up end-to-end; `@1` is expectedly weak and degrades most under paraphrase. The
**base-doctor (v3_base)** column is not yet a live-E2E number — its tool doesn't wire
the resolver (deferred); see [doctor-agent.md](doctor-agent.md#whats-next).

### Why tool-direct ≠ feature-vector-gated even without the LLM

tool-direct has **no LLM** — so why isn't it identical to feature-vector-gated,
which runs the *same models at the same 0.40 threshold*? Because the two paths
**build the feature vector differently**. feature-vector(-gated) calls the
training pipeline's `build_features()` on the raw MIMIC row, so it has every
column the models were trained on, straight from the tables. tool-direct
**reconstructs** that ~2116-column vector inside the tool, from only the fields a
live patient/nurse can actually report — and some columns cannot be reconstructed
from a bedside interview. On a borderline case a small feature difference flips a
gate decision (P(admit) crossing 0.40) or a top-1 argmax; at 13 cases one flip =
7.7pp. The runtime divergences (all verified in the tools/loaders):

1. **Prior-encounter numerics cannot be reconstructed.** The loader computes
   `days_since_last_admission`, `days_since_last_ed`, `n_prior_ed_visits`, and
   `same_complaint_as_prior` (Jaccard vs the patient's last visit) from prior
   MIMIC encounters. A patient at the bedside can't report these, so the
   **self-report path** zero-fills to the first-time-patient sentinel
   (`days_since = 9999`, `same_complaint = 0`). **This is unrecoverable by
   asking — only by an EHR lookup**, now wired into the disposition,
   reassessment, **and (as of 2026-06-03) the triage** tools for returning
   patients with an MRN (see the
   [EHR-integration section](#patient-history-lookup-for-returning-patients-ehr-integration--implemented) below).
2. **PMH flags round-trip through text.** feature-vector reads `pmh_<group>`
   flags directly from prior discharge-note parsing + ICD codes; tool-direct
   reverse-maps those flags to phrases ("congestive heart failure") and re-parses
   them through `pmh_vocab`. Asymmetric vocab coverage can drop/add a flag.
3. **Rhythm: one reading vs many — CLOSED.** The loader aggregates rhythm across
   all readings in the 4h window (one-hot = mode bucket, `irregular=1` if *any*
   reading is non-sinus). The runtime previously bucketed only the first reported
   string. Now the nurse collects a rhythm reading per chronological reading set, the readings ride
   inside the `vital_trajectory` blob under a `"rhythm"` key (parsed by
   `parse_rhythm_readings`), and `build_longitudinal_block` aggregates them with
   the *exact* training logic (mode + any-non-sinus). A single reading is the
   degenerate one-element case, so behavior is unchanged when only one is
   available. In the benchmark, `pull_rhythm_readings` supplies the full
   in-window sequence, so `tool-direct` now matches the feature-vector's rhythm
   features.
4. **Medications: different vocab coverage.** feature-vector flags come from the
   real `medrecon` aggregation (drug name + pharmacy-class `etcdescription`);
   the self-report path re-parses only patient-reported drug names through
   `med_vocab`. **Closeable for returning patients (added 2026-06-03):** the
   MRN EHR lookup now also returns a `med_block` — the patient's reconciled
   home-med list from their most recent PRIOR stay (leakage-safe, `< intime`) —
   which the disposition + v3 tools accept via `med_lookup_json` and use to
   OVERRIDE self-report. (Prior-visit, not current-visit, medrecon: a live
   patient's current list is what the nurse is collecting, so the EHR can only
   supply the last documented one — an honest proxy that won't perfectly match
   the current-visit `medrecon` the feature-vector column uses if meds changed.)
5. **Cascade computation.** The triage acuity/disposition cascade columns are
   computed slightly differently between the two paths (a few columns of ~2116).
6. **Rounding** of trajectory readings to 1 decimal in the tool path (trivial).

**Evidence — the entire tool-direct↔gated gap on this run is one patient.** A
per-case diff (`feature-vector` vs the tool's internal vector) found exactly
**one** flip: stay `37744212`, a 63yo walk-in with abdominal pain. feature-vector
admits her (P=0.474); tool-direct discharges her (P=0.266). The two vectors
differ in **10 of ~2116 columns and 0 TF-IDF columns** — the complaint text
round-trips perfectly. The dominant 4 are the prior-encounter numerics above:
she was admitted **2.7 days ago for the same complaint** (`days_since_last_admission`
2.71→9999, `same_complaint_as_prior` 1→0). feature-vector sees "frequent recent
admit, same problem → admit"; tool-direct is structurally blind to it. The other
6 differing columns are the cascade probabilities (#5).

**Interpretation for the thesis:** tool-direct is arguably the *more honest*
number and feature-vector-gated is a mildly optimistic upper bound — the gated
path uses prior-encounter features a real ED does **not** have at triage for a
walk-in. The ~1-case gap is the cost of *information the live system cannot
observe*, a permanent property of deployment, not an engineering defect. (The
one lever that *would* recover it — an EHR lookup for returning patients — is in
Future directions.)

**NL-fidelity (parser-extracted vs tabular truth):** cases with parser output
20/20; age 18/20; gender 20/20; arrival transport 20/20; mean chief-complaint
token Jaccard **0.657**. Demographics survive the round-trip (recoverable via the
parser's follow-up questions); the chief-complaint text is the lossy channel
(~66% token overlap after lay paraphrase).

### Remaining levers

1. **Scale beyond 20 cases — PLANNED (near-future), not yet done.** Needed to
   make the 13-case diagnosis/department numbers statistically meaningful, to
   average out the LLM jitter on E2E, and — now the most pressing reason — to
   give the **triage-PMH and prior-visit-medication lookups enough
   returning-patient volume to register** (the 2026-06-03 fast re-run showed
   both are net-neutral at n=20: acuity flipped nothing and only 5/20 had a
   prior medrecon). Unblock = regenerate a larger stratified case set + rebuild
   the index (`--all-subjects` or a bigger cohort), then re-run.
2. **A stronger / recalibrated disposition model** would raise the
   feature-vector-gated ceiling itself (12/13 admit recall at 0.40) — the one
   lever that lifts the gated diagnosis ceiling for *every* downstream column.
   Asymmetric `scale_pos_weight` retrain is the documented candidate.
3. **Threshold is now 0.40** (done); **0.30** is a future option if missed admits
   are judged substantially costlier than false admits — but see the over-triage
   caveat in [doctor-agent.md](doctor-agent.md#whats-next) (the 25–35% ED figure
   is from trauma/acuity triage, not admit/discharge).

## Patient-history lookup for returning patients (EHR integration) — IMPLEMENTED

**Motivation:** the per-case diagnostic above shows the *entire* residual
tool-direct↔feature-vector-gated gap on this run is **one returning patient**
whose prior-encounter numerics (`days_since_last_admission`,
`same_complaint_as_prior`, …) the runtime cannot reconstruct from a bedside
interview. The runtime handles history in *one* way today: the NLP parser/nurse
**asks** the patient (`prior_history`, `n_prior_admissions`) — which recovers the
coarse PMH flags but not the fine recent-visit numerics. This is the
complementary half: **fetch** prior data for patients already in the system.

**What shipped:**

- **`pmh_features.py` refactor** — `aggregate_pmh` was split into
  `build_pmh_index(subjects, …)` (the heavy half: parses admissions / edstays /
  ICD / the 3.3 GB discharge.csv into four per-subject structures) and
  `assemble_pmh_for_stay(subject_id, intime, complaint_norm, index)` (the cheap
  per-stay loop body, with the `< intime` leakage filter + an explicit assertion).
  `aggregate_pmh` now calls both, so training and inference can't drift.
- **Offline index builder** — `scripts/build_history_index.py` (`uv run
  build_history_index`, `--from-cases` default / `--all-subjects`) persists the
  index to `HISTORY_INDEX_PKL` (`artifacts/history/pmh_index.joblib`, gitignored).
- **`PatientHistoryLookupTool`** (`src/proiect_licenta/tools/patient_history_lookup_tool.py`) —
  keyed on a **`subject_id` / simulated MRN** (deliberately *not* fuzzy name/age
  matching — record linkage is its own error-prone subsystem). Loads the index
  (per-process cache) and returns the real 19-column `pmh_block` for a known
  patient, or `known_patient=false` for a first-time/unknown subject (or when the
  index isn't built) so the pipeline keeps its ask-the-patient zero-fill path.
- **Medication lookup (added 2026-06-03)** — the same tool now ALSO returns a
  `med_block` (the 11-col `MED_FEATURE_COLS`) from the patient's most recent
  PRIOR stay's reconciled `medrecon` home-med list, via
  `pmh_features.build_pmh_index` (new `med_by_stay`) + `assemble_meds_for_stay`
  (leakage-safe, `< intime`, reuses the PMH discipline). `med_vocab.py` hosts
  the shared `MED_FEATURE_COLS`, `med_block_from_rows`, `parse_med_lookup`, and
  `med_self_report_discrepancy`. The disposition + v3 tools gained a parallel
  `med_lookup_json` override (triage has no med features, so it's untouched).
  Prior-visit semantics (not current-visit) keep it honest about live deployment.
- **Doctor-tool wiring** — `doctor_disposition_tool.py` and `doctor_tool_v3.py`
  gained an optional `pmh_lookup_json` arg (mirroring `vital_trajectory_json`):
  when present the real PMH block **overrides** the text-derived one; when empty
  they fall back to self-report. `doctor_tool_v3_base.py` uses no PMH, so it's
  unaffected. The doctor agent in the live crew carries the lookup tool, and
  `tasks.yaml` instructs it to call the lookup (if an MRN is present) and forward
  `pmh_lookup_json` into the disposition + reassessment tools.
- **Triage-tool wiring (added 2026-06-03)** — `triage_tool.py` gained the same
  optional `pmh_lookup_json` arg + override, and `PatientHistoryLookupTool` is
  now registered on the **triage agent**. `triage_assessment_task` step "1b"
  calls the lookup when the patient gives an MRN and passes the `pmh_block` to
  the triage tool — so a returning patient's real prior-encounter numerics reach
  the triage acuity/disposition models too, not just the doctor stage. The
  shared `parse_pmh_lookup` + new `pmh_self_report_discrepancy` helpers were
  hoisted into `pmh_features.py` (single source of truth); triage and both
  doctor tools import them and all three emit a read-only `self_report_not_in_record`
  reconciliation list (record still wins the feature vector).
- **Benchmark column** — `benchmark_pipeline_e2e.py` adds a **`tool_direct_lookup`**
  mode (tool-direct + the EHR lookup by the case's real `subject_id`+`intime`)
  alongside the no-lookup `tool_direct`. The case bundle now carries `intime`
  (the leakage cutoff). The gap between the two columns = **the value of EHR
  access at triage**.
  - **Triage now wired into the lookup columns (2026-06-03):** `run_tool_direct`
    forwards `pmh_lookup_json` to the triage `_run` call as well (not just the
    disposition + v3 tools), so `tool_direct_lookup` now exercises triage's
    lookup; `tool_direct` passes `""` and stays on self-report. The **E2E
    (real-crew) column** needs no benchmark change — it runs the actual crew,
    which calls the lookup at triage, and the harness's class-level
    `PatientHistoryLookupTool` patch (forcing the case's historical `intime`)
    already covers that call.
  - **Medication lookup also wired (2026-06-03):** `_lookup_blocks` now returns
    `(pmh_json, med_json)` and `run_tool_direct` forwards `med_lookup_json` to
    the disposition + v3 tools, so `tool_direct_lookup` exercises the prior-visit
    med override too (E2E gets it via the crew + the disposition/reassessment
    tasks forwarding `med_block`).
  - **Fast re-run after the 2026-06-03 wiring (done; `--skip-e2e`):** the four
    tabular columns are **unchanged** from the table below — the triage-side PMH
    and medication wiring are **net-neutral at n=20** (no regression, no
    measurable gain beyond the PMH→disposition flip already captured).
    Specifically: **ESI acuity stayed 70.0 → 70.0** (the triage PMH lookup
    flipped zero acuity predictions), and **only 5/20 patients even had a prior
    medrecon** to look up (the rest are first-in-window visits where
    `assemble_meds_for_stay` correctly returns `None`), so the med override could
    not register a scored change. The effect of both is simply below the n=20
    noise floor — see "Remaining levers" (scaling the cohort is the unblock).
- **Live intake path (wired)** — the NLP parser (`parse_symptoms_task`) now asks,
  *first*, whether the patient has been treated here before and for their MRN,
  emitting `subject_id` (an integer, `-1` if new/unknown) in its JSON. That
  `subject_id` is threaded through every `---STRUCTURED_DATA---` block
  (triage → initial doctor → disposition). **`triage_assessment_task` (step 1b,
  added 2026-06-03)** and `doctor_disposition_task` each call
  `patient_history_lookup_tool` with it (and `current_intime="now"`) before
  their prediction tool, forwarding the resulting `pmh_block` as
  `pmh_lookup_json`. (Each stage re-calls the deterministic, process-cached
  lookup rather than threading the block through the LLM text — chosen for
  robustness over a fire-once-and-propagate variant.)
  So entering a known `subject_id` at intake (e.g. `17287581`) flips the case to
  a returning patient with real prior history; `-1` keeps the ask-the-patient
  path. **MIMIC date caveat:** MIMIC timestamps are de-identified into the future,
  so a real wall-clock "now" would precede every encounter (→ "no prior
  history"). The tool therefore resolves the `"now"` sentinel to the subject's
  **most recent recorded encounter** (treated as today's visit), so the `< intime`
  filter returns everything strictly before it. (An explicit ISO timestamp — the
  benchmark path — is used verbatim, which is why the benchmark sees stay
  `37744212`'s exact 3-prior-admissions / 2.71-day numbers while a live `"now"`
  lookup for the same subject sees their full pre-latest-visit history.) This is
  inherently a **simulation**: a live MRN only matches if it equals a real MIMIC
  `subject_id`, so casual testing with a made-up number correctly shows a
  first-time patient.

**Leakage discipline (the critical risk):** `assemble_pmh_for_stay` filters to
encounters strictly *before* the current `intime` and asserts no surviving prior
encounter is dated at/after it. The current encounter's own discharge summary can
never leak because its admit/intime is not strictly before `intime`. The
`build_pmh_index` notes-by-prior-hadm guard is preserved from `aggregate_pmh`.

**Status / what's verified:** the refactor, builder, tool, doctor-tool override,
crew/task wiring, and benchmark column are implemented; `assemble_pmh_for_stay`
is unit-tested (returning patient → real `days_since`/`same_complaint`; future-
only encounters → excluded; unknown subject → first-timer; ISO-string intime).

**Single-case real-data check (done — the headline mechanism is confirmed).**
Built the index over the 20-case cohort (`uv run build_history_index --from-cases`,
~44 s discharge parse) and queried the tool for stay `37744212` (subject
`17287581`) at its real arrival time: it returns `days_since_last_admission =
2.71` and `same_complaint_as_prior = 1.0` — **exactly** the divergent features
the per-case diagnostic identified. Feeding that block into the disposition tool:

| | P(admit) | decision |
|---|---|---|
| tool-direct, **no** lookup | 0.266 | DISCHARGE |
| tool-direct, **with** lookup | **0.474** | **ADMIT** |

0.474 matches the feature-vector's value for this case to 3 dp — the lookup
closes the gap on the one patient that *was* the gap, flipping it back to the
correct admit (and thereby restoring its downstream diagnosis/department).

**Update (2026-06-03) — post triage-PMH + medication wiring, fast re-run
(`--skip-e2e`).** After registering the lookup on the triage agent (PMH) and
adding the prior-visit medication lookup, the four tabular columns below are
**unchanged**: both additions are **net-neutral at n=20** — no regression, no
measurable gain beyond the PMH→disposition flip already shown. Why: **ESI acuity
held at 70.0** (the triage PMH lookup flipped no acuity prediction), and **only
5/20 patients had a prior medrecon** to fetch (the rest are first-in-window
visits → `assemble_meds_for_stay` returns `None`), so the med override had almost
no cases to act on and changed no scored outcome. Both effects sit below the
n=20 noise floor; measuring them needs a larger returning-patient cohort
(**scaling — planned, near-future; see "Remaining levers"**). The E2E column was
not re-run in this pass.

**Full 20-case benchmark re-run (done — lookup + rhythm in place, MRN wired
through the E2E crew, regenerated bundle with `intime` + in-window rhythm).**
`tool_direct_lookup` equals `feature_vector_gated` on **every** gated target, and
**E2E now matches them too** — the lookup closes the entire runtime↔gated-
feature-vector gap both in tool-direct and end-to-end through the LLM (previously
a single-case anecdote, now the full-cohort result):

| Target | tool_direct | tool_direct_lookup | feature_vector_gated | feature_vector (ungated) | E2E |
|---|---|---|---|---|---|
| ESI acuity | 70.0% | 70.0% | 70.0% | 70.0% | 60.0% |
| Disposition | 90.0% | **95.0%** | 95.0% | 95.0% | **95.0%** |
| Diagnosis @1 | 61.5% | **69.2%** | 69.2% | 69.2% | **69.2%** |
| Diagnosis @3 | 84.6% | **92.3%** | 92.3% | 100.0% | **92.3%** |
| Department @1 | 53.8% | **61.5%** | 61.5% | 69.2% | **69.2%** |
| Department @3 | 84.6% | **92.3%** | 92.3% | 100.0% | **92.3%** |
| Dx coverage | 11/13 | **12/13** | 12/13 | 13/13 | **12/13** |

The per-case dump confirms the mechanism: stay `37744212` is exactly the case
that flips — `tool_direct` discharges it (no Dx), while `tool_direct_lookup`,
`feature_vector_gated`, **and E2E** all admit it and recover the correct
**Digestive / MED** (= ground truth). E2E now does so because the parser collects
the MRN (`_answer_ask` MRN branch), threads `subject_id` through the structured-
data blocks, and the disposition task calls the lookup — the full live chain. To
keep the E2E column apples-to-apples with the gated reference, the harness forces
the lookup's `current_intime` to the case's real stay intime (Option A; otherwise
the live `"now"` sentinel would anchor to the subject's *latest* encounter, a
different leakage cutoff). **Rhythm (#3) caused no regression**: `tool_direct` is
identical to the pre-rhythm baseline — only 2 cases had in-window rhythm coverage
and none flipped at n=20 (expected below-noise behavior; rhythm's value is only
measurable at scale). The residual `gated → ungated` gap (dx@3 92.3 vs 100,
coverage 12/13 vs 13/13) is the **disposition gate** — one admitted case it routes
to discharge. The only remaining E2E-specific loss is **ESI acuity (60.0%)**,
pure complaint-paraphrase + triage-agent jitter (the docs note acuity swinging
60↔70 across reruns of the same narratives); nothing here touches triage. E2E
department@1 (69.2%) even edged the gated column this run — a favorable LLM draw,
i.e. noise, not a systematic effect.

**Magnitude caveat (unchanged):** at 20 cases this is still a **mechanism
demonstration** — the gated-axis gap *was* one patient, so the lookup deltas above
ride on single cases, and the E2E column carries LLM run-to-run jitter. A scaled
benchmark (a few hundred cases) is the next step, now de-risked: the run above
confirms the wiring closes the gap (tool-direct AND E2E) with no regressions, so
scaling is about statistical weight, not correctness.
