# LLM Backend — Gemini Flash 2.5 vs MedGemma

This document is the canonical reference for the **switchable LLM backend**: why it
exists, how it's implemented, how to serve MedGemma, how it's benchmarked, and what
Experiment A found. It is cross-cutting — it touches the [NLP Parser](agents/nlp-parser-agent.md),
the [Case Generation Agent](agents/case-generation-agent.md), and the
[end-to-end benchmark](agents/case-generation-agent.md#benchmark).

---

## 1. Motivation

The project uses an LLM for two jobs only: the **NLP Parser** (live crew) and the
**Case Generation Agent** (offline). Everything that *predicts* is XGBoost. The thesis
question: does a **medical-domain** LLM (MedGemma) change pipeline results vs a strong
**general** model (Gemini Flash 2.5)?

MedGemma has **no managed API** — it is open weights (gated on Hugging Face), so it must
be **self-hosted**. The switch therefore has to (a) select the backend with zero changes
to the ML pipeline, and (b) be host-agnostic so the GPU can live anywhere (Colab, Kaytus,
Vertex, local).

### Key fact: every CrewAI agent is LLM-backed
In CrewAI the LLM does not only parse — it decides which tool each agent calls and writes
each task's output. So the triage/doctor/nurse agents are *not* LLM-free at the
orchestration layer (only their XGBoost **tools** are ML). A faithful Flash-vs-MedGemma
comparison therefore swaps the LLM for **all four live agents + the case generator**, and
MedGemma's tool-calling reliability is a whole-crew concern, not just a parser one.

---

## 2. Implementation

### `src/proiect_licenta/llm_config.py` — single source of truth
Selected by the `LLM_BACKEND` env var (mirrors the `paths.py` centralization pattern).

- **`get_llm()`** — used by the live agents + case generator.
  - `flash` (default) → returns **`None`**. `None` is the CrewAI Agent default, so the
    Gemini path is **byte-for-byte unchanged** — the live pipeline only changes by
    *choosing not to override it*.
  - `medgemma` → returns a `crewai.LLM` with **`provider="openai"`** (native OpenAI
    client) pointed at the vLLM endpoint. See the "Wiring gotcha" below for why
    `provider="openai"` is required.
- **`get_parse_llm()`** — used **only** by the benchmark's `parser-llm` mode. Unlike
  `get_llm()` it is **never `None`**: for `flash` it returns an explicit Gemini
  `crewai.LLM`. This is a benchmark-only direct completion (not the agentic crew), so it
  does not touch the live pipeline — it exists so Flash can also run the `parser-llm` mode
  and give a **same-mode** comparison against MedGemma.

Both LLMs are built with `temperature=0` for reproducible benchmark runs.
`load_dotenv()` runs at import so `.env` values are available.

### Wiring points
- `src/proiect_licenta/crew.py` — `llm=get_llm()` on all four `@agent` methods
  (`nlp_parser`, `triage_agent`, `doctor_agent`, `nurse_agent`). For `flash` this is
  `None` → unchanged.
- `src/proiect_licenta/case_generation.py` — `llm=get_llm()` on `case_generator`; plus an
  `out_dir` param so MedGemma narratives can be written to a separate directory.

### Wiring gotcha (why `provider="openai"`)
`crewai[google-genai]==1.9.3` ships the native Google provider but **not LiteLLM**.
Passing `model="openai/<name>"` makes CrewAI route to its LiteLLM fallback for unknown
model names → `ImportError: Fallback to LiteLLM is not available`. Passing
`provider="openai"` + the bare served model name forces CrewAI's **native**
`OpenAICompletion` client, which accepts a custom `base_url`/`api_key` and sends the model
name straight to vLLM — no LiteLLM needed.

### `.env`
```
LLM_BACKEND=flash                 # flash | medgemma
MEDGEMMA_BASE_URL=                # e.g. https://<tunnel>/v1  (only used when medgemma)
MEDGEMMA_API_KEY=EMPTY            # any non-empty string for vanilla vLLM
MEDGEMMA_MODEL=google/medgemma-4b-it
```
`.env` is gitignored.

---

## 3. Serving MedGemma (self-hosted vLLM)

The GPU host is interchangeable — the code only needs `MEDGEMMA_BASE_URL` + `MEDGEMMA_API_KEY`,
so Colab / Kaytus / Vertex / local all work by changing env values. Full walk-through and a
ready-to-run notebook are in [`../colab/`](../colab/README.md).

**Serving stack:** vLLM's OpenAI-compatible server, exposed with a Cloudflare quick-tunnel,
consumed by CrewAI's native OpenAI client.

**Colab gotchas that were solved (documented so they don't recur):**
- Colab's GPU **driver is CUDA 13.0**, but its PyTorch is cu12.8. The newest vLLM wheel is
  CUDA-13-only → `ImportError: libcudart.so.13`. Pinning an **old cu12 vLLM** instead is too
  old for MedGemma's Gemma-3 config → pydantic `rope_scaling should have a 'rope_type' key`.
  **Fix:** install with uv, all-cu13:
  `uv pip install --system -U vllm --torch-backend=auto` (auto detects the 13.0 driver).
- **Sessions are ephemeral** (~12–24 h, idle-disconnect ~90 min) and the tunnel URL changes
  per session — run each benchmark in one sitting and update `MEDGEMMA_BASE_URL` if it drops.

**Tool-calling limitation (important).** Vanilla vLLM rejects CrewAI's `tool_choice="auto"`
with `400 "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`.
Even with those flags, MedGemma-4B is not function-call tuned and Gemma-3 has no dedicated
vLLM tool-call template, so it cannot reliably drive the **full agentic crew**. This is why
MedGemma is evaluated via the `parser-llm` bypass (below). Flash drives the agentic crew fine.

---

## 4. How it's benchmarked

`benchmarks/benchmark_pipeline_e2e.py` runs the cached 20 synthetic cases
(`data/derived/synthetic_cases/cases.json`) through several modes on the same `stay_id`s and
scores each against MIMIC truth. Modes relevant to the backend comparison:

| mode | LLM? | what it isolates |
|---|---|---|
| `tool_direct` | no | exact tabular fields → tools. The no-LLM reference ceiling. |
| `tool_direct_lookup` | no | + EHR history lookup by `subject_id`. |
| `feature_vector` / `_gated` | no | cached `build_features` rows → models (± disposition gate). |
| **`parser_llm`** | **yes** | one **direct** LLM parse (narrative → JSON) → the deterministic `tool_direct` pipeline. Isolates **clinical NL parsing** with no agentic tool-calling. Runs for **both** backends (`get_parse_llm()`). |
| `e2e` | yes | the **full agentic crew** on the narrative. Flash only — MedGemma can't (tool-calling). |

CLI flags: `--llm-backend {flash,medgemma}`, `--parser-llm`, `--skip-e2e`,
`--dump-json <path>` (writes per-mode metrics **and** the NL-fidelity block),
`--plain-prompt` (original parser prompt vs the clinical-term default — §5c Track 1),
`--clinicalize` (apply the lay→clinical map to parser output — §5c Track 2; re-runs free
from the parse cache). The dump records `parse_prompt` + `clinicalize` for auditability.

### The fair cross-backend anchors
1. **Same-mode `parser_llm`** — identical deterministic downstream, differing only in which
   LLM parsed. This is the apples-to-apples accuracy comparison.
2. **NL-FIDELITY** — what the parser extracted (age / gender / arrival transport / complaint
   token Jaccard) vs tabular truth. Orchestration-independent, so it's valid even when Flash
   also ran agentic `e2e` and MedGemma did not.

### Run commands
```bash
# MedGemma (tunnel up):
uv run python benchmarks/benchmark_pipeline_e2e.py --llm-backend medgemma --parser-llm --skip-e2e \
    --dump-json artifacts/benchmarks/e2e_medgemma.json
# Flash (no tunnel needed; add --skip-e2e for the same-mode comparison, drop it for agentic numbers):
uv run python benchmarks/benchmark_pipeline_e2e.py --llm-backend flash --parser-llm --skip-e2e \
    --dump-json artifacts/benchmarks/e2e_flash.json
```

---

## 5. Experiment A results (20-case, same-mode `parser_llm`, 2026-06-16)

Cases: 20 (13 admitted / 7 discharged). One case ≈ 7.7pp on the 13-admitted metrics.
**Methodology check passed:** the four LLM-free modes were byte-identical across both
backends' dumps, confirming only the parser LLM varied.

| target | `tool_direct` (no-LLM ref) | **Flash** parse | **MedGemma** parse |
|---|---|---|---|
| ESI acuity | .700 | **.750** | .700 |
| disposition | .900 | .800 | .800 |
| diagnosis @1 | .615 | .462 | .462 |
| diagnosis @3 | .846 | .692 | .692 |
| department @1 | .538 | .538 | .538 |
| department @3 | .846 | **.769** | .692 |
| ICD exact @5 | .615 | .385 | .385 |
| ICD union | .692 | .615 | .615 |
| Dx coverage | 11/13 | 10/13 | 10/13 |

**NL-FIDELITY** (parser extraction vs truth):

| | age | gender | arrival transport | complaint Jaccard |
|---|---|---|---|---|
| Flash | 20/20 | 19/20 | 19/20 | .558 |
| MedGemma | 20/20 | 19/20 | 19/20 | **.602** |

### Interpretation
- **MedGemma-4B and Flash 2.5 are statistically indistinguishable as the parser.** They tie
  exactly on disposition, diagnosis @1/@3, department @1, and exact-ICD. Flash is +1 case on
  acuity and department @3; MedGemma is marginally ahead on complaint fidelity. Every delta is
  within the n=20 noise floor.
- **The free-text cost is real and equal for both.** Both LLM parsers drop ~15pp on
  diagnosis@1 vs the exact-field `tool_direct` (.615 → .462) — paraphrased complaints overlap
  only ~56–60% with the tabular `chief_complaint`, shifting the TF-IDF features. This is the
  irreducible NL-layer cost, identical across models.
- **Operational asymmetry.** Flash drives the full agentic crew; MedGemma-4B cannot out of the
  box. So a small open medical model is a viable **parser** but not an agentic **orchestrator**
  without extra work.
- **Headline.** A 4B open medical model matches a strong proprietary general model at clinical
  NL parsing in this pipeline — with the honest caveat that n=20 is small.

## 5b. Scaled to 150 cases — the headline result (2026-06-17)

The n=20 pilot couldn't separate the models. We scaled to **150 cases (98 admitted / 52
discharged)** — 1 case now ≈ 1.0pp on diagnosis/department/ICD (~0.67pp on acuity/dispo).
Both backends ran the **identical lookup-enabled `parser_llm` bypass** (one direct LLM parse →
deterministic gated tool-direct, with the prior-encounter PMH + reconciled-med EHR lookup fed
to the tools). **Methodology check passed** — the four LLM-free modes are byte-identical across
both dumps. Machine-readable results: `artifacts/benchmarks/comparison_n150.json` (+ the raw
`parser_flash_n150.json` / `parser_medgemma_n150.json`).

| target | `feature_vector_gated` (ceiling) | **Flash** parse | **MedGemma** parse | Δ (MG−Flash) |
|---|---|---|---|---|
| ESI acuity | .720 | **.660** | .620 | **−4.0pp** |
| disposition | .873 | .747 | **.760** | +1.3pp |
| diagnosis @1 | .500 | .367 | **.398** | **+3.1pp** |
| diagnosis @3 | .714 | .520 | **.561** | **+4.1pp** |
| department @1 | .622 | .541 | **.551** | +1.0pp |
| department @3 | .806 | .653 | .653 | 0.0pp |
| ICD exact @5 | .327 | .255 | **.276** | +2.1pp |
| ICD union | .541 | .418 | **.439** | +2.1pp |
| complaint Jaccard | — | **.425** | .413 | −1.2pp |

**Finding.** MedGemma-4B has a **small but directionally consistent edge on every
clinical-content target** (diagnosis, department, disposition, ICD) — *despite* a marginally
lower complaint-token Jaccard. It paraphrases the complaint into more **clinically-aligned
terminology** (better TF-IDF→model signal) even when it overlaps the raw tabular tokens slightly
less. Flash wins **ESI acuity** (+4.0pp). Each delta is 1–4 cases (within the per-metric ~±7–10pp
95% CI), so the signal is the *consistency of direction across five clinical targets*, not any
single number — a meaningful sharpening of the n=20 tie, but still "modest, consistent edge for
the medical model," not a decisive win. MedGemma produced parseable JSON for all 150 parses
(0 errors), confirming it is a competent **parser** even though it can't drive the agentic crew.

### Why the LLM parse costs ~10–13pp, and how to read it fairly

The four LLM-free columns are designed to separate three losses, so the parser gap is isolated
cleanly. For diagnosis@1 (n=150):

| stage | diag@1 | loss isolated |
|---|---|---|
| `feature_vector` (ungated) | .633 | — model on exact features, **upper bound** (= the tabular `ceiling.json` 64.1%) |
| `feature_vector_gated` | .500 | **−13.3pp disposition gate** (admit-only diagnosis; dispo sensitivity ~76.5% discards true admits) |
| `tool_direct_lookup` | .500 | **~0 runtime feature-degradation** — the input-matched runtime path equals the model-path ceiling |
| `parser_llm` (Flash / MedGemma) | .367 / .398 | **−13.3 / −10.2pp NL parse** — the only remaining gap is the LLM-parsed complaint |

So with **equal features and equal gating**, the residual is attributable *solely* to the chief
complaint being parsed from free text rather than read exactly from the chart. The mechanism is a
**clinical-shorthand ↔ lay-language gap**: the synthetic patient says "really bad **stomach**
pain" for the tabular `Abd pain`, "**brain bleed**" for `SDH`, "bright red blood in my stool" for
`BRBPR`. These are semantic paraphrases that abbreviation-expansion can't bridge, so the
complaint's TF-IDF vector shifts and the (complaint-dominated) diagnosis/department/ICD models
move with it. Complaint Jaccard vs the tabular `chief_complaint` is only ~0.42 — and the parser
is *not* at fault (it extracts ~0.42 vs a raw-narrative ceiling of ~0.06, and parses age/gender/
transport near-perfectly: 150/137/144). It is the irreducible cost of free-text intake vs a
perfectly-charted complaint, and it is **equal in structure for both backends**, so the
Flash-vs-MedGemma delta above is unaffected by it. `feature_vector` *ungated* (.633) is **not** the
fair comparator for the gated `parser_llm`; `feature_vector_gated` (.500) is.

Roadmap for *shrinking* this gap (system-side, not by changing the benchmark — see §7): (1) prompt
the LLM to emit the **clinical** chief-complaint term, not a verbatim paraphrase — **DONE, §5c**;
(2) a deterministic lay→clinical map — **DONE (`clinicalize_complaint`), §5c**; (3) semantic
complaint embeddings (re-openable: the prior Bio_ClinicalBERT revert was on *exact* complaints,
never under paraphrase); (4) paraphrase data-augmentation at training time. (1)+(2) each recover
~4–6pp of the diagnosis@1 gap on the frozen 250-case Flash headline (§5c) — about half the gap,
and exactly where a medical model's instruction-following may change the balance (MedGemma pending).

### Benchmark bug found and fixed during interpretation
The parser prompt offered an `"unknown"` arrival-transport option the **live** parser
(`tasks.yaml`: `ambulance`/`helicopter`/`walk_in`, walk-in default) does not have. Of 20 cases,
11 narratives state no transport (walk-in is the absence of an ambulance mention). Flash,
being literal, answered `"unknown"` for those → scored wrong vs the `walk_in` ground truth
(**10/20**); MedGemma defaulted to `walk_in` → matched (**18/20**). The gap was a prompt
artifact rewarding default-guessing, not a parser-quality difference. Aligning the prompt to
the live 3-value contract (walk-in default, no "unknown") brought **both to 19/20** (the single
genuine `unknown`-truth PICC case is an inherent miss for both). This is why the bypass must
replicate the live parser's contract exactly.

---

## 5c. Shrinking the NL-parse gap — clinical-term prompt + lay→clinical map (2026-06-20)

§5b isolated the ~10–13pp diagnosis@1 NL-parse cost to the **chief-complaint being parsed
from lay free text**. Two system-side levers attack it directly (both fair — the lay→clinical
bridge stays *inside* the system-under-test; we do **not** clinicalize the narrative input):

- **Track 1 — clinical-term parser prompt.** The parser is steered to emit the **standard
  ED complaint term** instead of a verbatim paraphrase ("brain bleed" → "intracranial
  hemorrhage", "bright red blood in my stool" → "BRBPR"). Applied symmetrically to the
  benchmark `_PARSE_PROMPT` and the live `tasks.yaml` `parse_symptoms_task`. The benchmark
  keeps both a `--plain-prompt` variant (reproduces the documented baseline) and the
  clinical-term default, so the prompt's effect is measurable in isolation.
- **Track 2 — deterministic lay→clinical map.** `preprocessing.clinicalize_complaint()`
  (`LAY_TO_CLINICAL`) rewrites lay phrases to the clinical terms the models were trained on.
  **Applied to parser output ONLY** — never to the tabular complaint, never at training.

**Design note — why `clinicalize_complaint` is separate from `normalize_complaint_text`.**
The obvious approach (edit `normalize_complaint_text` in place) would change the **shared
TF-IDF vocabulary and force retraining** every dependent model, because MIMIC triage staff
already chart many semi-lay terms (`difficulty breathing` ×268, `dizzy` ×734, `lightheaded`
×2016, `weak` ×271, `chest tightness` ×145, `allergic reaction` ×2554, `head injury` ×3265,
`vaginal bleeding` ×2540, `migraine` ×207, `passed out` ×65, …). Keeping the map as a
**parser-output-only** step leaves the vectorizer and all trained models **byte-identical**
(verified: `clinicalize_complaint` is a 0-mismatch no-op on the top clinical complaints), and
leaves the benchmark's `tool_direct`/`feature_vector` reference modes uncontaminated — so the
accuracy delta from enabling it cleanly isolates the map's value. Map targets are anchored to
the MIMIC training `chiefcomplaint` vocabulary; lay keys come from general/consumer-health
wording + the inverse of the case-generator's documented paraphrase rules — **not** from
inspecting eval cases (non-circular). A useful by-product finding: the chart vocabulary is
already semi-lay, so the gap is **not purely vocabulary**.

**Cost infra.** The benchmark now caches each LLM parse keyed by (prompt, narrative) under
`artifacts/benchmarks/parse_cache_{backend}.json`. The deterministic map is applied *after*
the parse, so toggling `--clinicalize` re-runs from cache with **zero LLM calls**; a prompt
change alters the key and self-invalidates. This makes the 2×2 below cost only the two
prompt variants' parses.

### Flash 2×2 — HEADLINE (fresh 250-case completeness cohort, 2026-06-21)
Cohort: `data/derived/synthetic_cases/cases.json` regenerated with the Track-3 completeness +
stay-lay prompt (Flash, seed 20260620, 163 admitted / 87 discharged, **250/250 grounding-clean**,
81 multi-complaint cases). The whole 250 is the frozen test — the map/prompt are non-circular
(built from training vocab + general knowledge, never tuned on these cases). **Methodology check
PASS**: the LLM-free `tool_direct` / `tool_direct_lookup` modes are byte-identical across all four
cells, so only the parser column varied. Audit: `artifacts/benchmarks/clinicalize_2x2_flash_n250.json`
(+ raw dumps `artifacts/benchmarks/headline_n250/`).

Full table (all targets; Δ = pp vs base):

| target | ceiling | base | +map | +prompt | +both | map Δ | prompt Δ |
|---|---|---|---|---|---|---|---|
| ESI acuity | .684 | .628 | .628 | .648 | .644 | +0.0 | +2.0 |
| disposition | .820 | .760 | .776 | .768 | .768 | +1.6 | +0.8 |
| diagnosis @1 | .472 | .356 | .399 | **.417** | .417 | +4.3 | **+6.1** |
| diagnosis @3 | .663 | .564 | .595 | .601 | .601 | +3.1 | +3.7 |
| department @1 | .589 | .503 | **.528** | .478 | .478 | +2.4 | **−2.5** |
| department @3 | .736 | .669 | .687 | .669 | .669 | +1.8 | +0.0 |
| ICD exact @1 | .233 | .092 | .166 | .178 | .178 | +7.4 | +8.6 |
| ICD exact @5 | .393 | .252 | .301 | .307 | .301 | +4.9 | +5.5 |
| ICD flat @10 | .448 | .350 | .368 | .387 | .380 | +1.8 | +3.7 |
| ICD union | .534 | .472 | .491 | .497 | .491 | +1.8 | +2.5 |
| complaint Jaccard | — | .393 | .523 (rewrote 76/250) | **.607** | .612 (rewrote 11/250) |  |  |

**Flash findings.**
- **Both levers give real but modest gains** — diag@1: map **+4.3pp**, prompt **+6.1pp**; ICD@1
  +7.4/+8.6, ICD@5 +4.9/+5.5. They close roughly **half** of the ~11.6pp base→ceiling diag@1 gap.
  Consistent with the "chart vocab is already semi-lay → gap not *purely* vocabulary" finding.
- **Prompt vs map diverge on department.** The clinical-term prompt is best for diagnosis/ICD
  (and helps acuity +2.0) but **slightly regresses department (−2.5pp)**; the deterministic map is
  the **more balanced** lever (helps diagnosis *and* department +2.4). Department deltas are ~4
  cases at n=163 admitted (near the noise floor), so the robust signal is **diagnosis@1 + ICD**.
- **Largely redundant, with a real trade-off.** `+both` == `+prompt` (the clinical prompt already
  emits clinical terms → the map rewrites only 11/250 on top). Choosing the prompt forgoes the
  map's department gain; the map alone keeps it but recovers a bit less on diagnosis.

**n150 quick-read (older/easier cohort, for the record).** The first read on the *pre-completeness*
150-case set (`clinicalize_2x2_flash_n150.json`, ceiling .500) showed **larger, optimistic** gains —
map +10.2pp, prompt +12.2pp diag@1. The smaller, easier, pre-completeness cohort overstated the
effect; the frozen 250 above is the headline. (This is itself a lesson: validate on the larger,
deployment-realistic cohort, not a quick read.)

### MedGemma-4B vs Flash 2.5 — same frozen 250 cohort (2026-06-21)
Same 2×2, only the parser LLM differs (downstream deterministic + frozen models identical; ceiling
is LLM-free and shared). MedGemma parses were produced via the self-hosted vLLM tunnel.
Audit: `clinicalize_2x2_medgemma_n250.json`; combined `comparison_flash_vs_medgemma_n250.json`.

| target | base | +map | +prompt | +both | map Δ | prompt Δ |
|---|---|---|---|---|---|---|
| ESI acuity | .572 | .552 | .576 | .580 | −2.0 | +0.4 |
| disposition | .740 | .756 | .768 | .768 | +1.6 | +2.8 |
| diagnosis @1 | .344 | .399 | **.423** | .423 | +5.5 | **+8.0** |
| diagnosis @3 | .534 | .583 | .595 | .595 | +4.9 | +6.1 |
| department @1 | .448 | .466 | **.503** | .503 | +1.8 | **+5.5** |
| department @3 | .620 | .650 | .650 | .650 | +3.1 | +3.1 |
| ICD exact @1 | .080 | .141 | .172 | .172 | +6.1 | +9.2 |
| ICD exact @5 | .239 | .294 | .325 | .319 | +5.5 | +8.6 |
| ICD flat @10 | .337 | .368 | .374 | .368 | +3.1 | +3.7 |
| ICD union | .466 | .491 | .491 | .485 | +2.5 | +2.5 |
| complaint Jaccard | .394 | — | .529 | — |  |  |

**MedGemma findings (hypotheses confirmed, with a sharper story).**
- **Weaker on *plain* parsing, but responds better to clinical steering.** MedGemma's plain base is
  behind Flash on every clinical target (diag@1 .344 vs .356; dept@1 .448 vs .503; acuity .572 vs
  .628), yet the clinical-term prompt lifts it **more** (diag@1 +8.0 vs Flash +6.1; ICD@5 +8.6 vs
  +5.5), so at the best config it ends **level-or-ahead** (diag@1 .423 vs .417 — tie within noise;
  dept@1 .503 vs .478 and ICD@5 .325 vs .307 — MedGemma ahead). The documented "small medical-domain
  edge" thus **materializes through the clinical-term prompt**, not plain parsing, on this cohort.
- **The prompt's department regression is Flash-specific.** Flash's clinical prompt *hurts*
  department (−2.5pp); MedGemma's *helps* it (**+5.5pp**). So clinicalizing isn't inherently a
  diagnosis-vs-department trade-off — that was a Flash artifact; the medical model handles the
  clinical-term instruction more coherently. ✓ (the original hypothesis)
- **The map helps MedGemma slightly more** (+5.5 vs +4.3 diag@1) ✓, and despite a **lower**
  clinical-term Jaccard (.529 vs Flash .607) MedGemma's downstream accuracy is higher — it
  paraphrases into terms the TF-IDF models recognize even when raw token-overlap is lower (same
  dynamic as the n150 finding).
- **Redundancy holds for both** (`+both` ≈ `+prompt`); for MedGemma the map on top of the clinical
  prompt even slightly dents ICD (.319 vs .325 @5).
- **Flash still wins ESI acuity decisively** (.648 vs .576 at +prompt, ~18 cases on n=250 — robust),
  matching §5b. So the net picture: **Flash → acuity; MedGemma → department + ICD; diagnosis a
  tie** (all at the best, clinical-term config).

### Run mechanics & robustness findings (lessons for re-running)
- **OOM, not sleep, killed the long heavy runs.** Full-benchmark runs doing 250 *fresh* parses while
  holding models + features + prediction dicts hard-died (no traceback) on the 14 GB machine after
  ~60–145 parses. Fix: **decouple** the tunnel work into (A) a lightweight clinical **cache-fill**
  (parse-only, minimal memory, per-case `try/except`, incremental save) then (B) **tunnel-free
  scoring** from the cache. (Cache-hit cells like `base`/`+map` never OOM'd — only fresh-parse loops.)
- **Incremental + atomic parse-cache flush** (every 20 parses, temp-file rename) added so a killed
  run loses ≤20 parses and the next resumes from cache. This is what made the repeated MedGemma
  tunnel/OOM interruptions cheap to recover from.
- **Self-hosted tunnel reality:** Cloudflare quick-tunnels crash (`530 origin unregistered`) and
  hand out a **new URL per session** — update `MEDGEMMA_BASE_URL` and re-probe `/v1/models` before
  each run. The per-backend cache (`parse_cache_medgemma.json`) survives across tunnel restarts.
- **Validator false-positives caught on the 250 regen:** 6 narratives were flagged for "dropped"
  SI/palpitations complaints, but the narratives were correct ("harming myself", "heart feels like
  it's racing") — the `_COMPLETENESS_ANCHORS` were too narrow. Broadening them → **250/250 clean**,
  backup-150 still 0 false-positives. (Generation QA only; independent of the eval map.)

### Track 3 — case completeness + stay-lay guard (fair)
The case-gen prompt (`case_generation_{agents,tasks}.yaml`) now requires **every** complaint
to be voiced in lay words (no dropped secondary complaints; sentence cap relaxed to fit
them). `case_generation.validate_grounding` gained two checks, both **independent of the
Track-2 eval map** (so generation isn't biased toward it): a **clinical-jargon leakage
guard** (keeps narratives lay — guards the medical-model over-clinicalization risk in §7) and
a **completeness check** using a separate `_COMPLETENESS_ANCHORS` set (flags dropped
complaints for common families; conservative). On the 250-case regen the cohort is **250/250
grounding-clean** after a one-time broadening of the SI/palpitations anchors — the initial 6
flags were validator false-positives on correct lay phrasings ("harming myself", "heart feels
like it's racing"), not bad narratives; the backup-150 stays at 0 false-positives.

---

## 6. Experiment B — case-generation quality (designed, not yet run)

Separate from Experiment A. Regenerate the narratives with each backend and compare the
deterministic grounding-validator pass rate (stated age/pain match the row; no fabricated
vitals in the opening narrative) — measures the **Case Generation Agent** itself.
```bash
uv run generate_cases --backend medgemma      # writes data/derived/synthetic_cases_medgemma/
uv run python benchmarks/compare_case_generation.py
```
`--backend flash` writes the canonical `synthetic_cases/`; `medgemma` writes
`synthetic_cases_medgemma/`, so the two sets never collide.

---

## 7. Future work
- ~~**Scale 20 → ~100 cases**~~ **DONE (2026-06-17)** — scaled to **150** (`generate_cases
  --backend flash --n-admitted 98 --n-discharged 52`); see §5b. Required a memory fix to
  `case_generation.py` (`sample_and_extract` freed the 418K disposition frame before the nurse
  loader — two full MIMIC frames at once OOM'd a 14 GB machine) and forcing UTF-8 I/O
  (`PYTHONUTF8=1`) so the `≥`/`κ` prints don't crash under cp1252 when stdout is redirected.
- ~~**250-case completeness regen + Flash & MedGemma 2×2 headline**~~ **DONE (2026-06-21, §5c).**
  Frozen 250-case cohort (seed 20260620); both backends' full 2×2 (clinical-term prompt × map).
  Each backend closes ~half the diag@1 gap; the prompt's department regression is Flash-specific
  (MedGemma gains there); Flash wins acuity, MedGemma edges department/ICD, diagnosis a tie. This is
  Experiment-A-style (parser comparison on a shared, Flash-generated cohort).
- **Run Experiment B** (case-generation *grounding quality* — regenerate narratives with MedGemma
  and compare validator pass rates via `compare_case_generation.py`) — **still pending**; distinct
  from the parser comparison above. Needs a live MedGemma tunnel for a ~50-min generation run.
- ~~**Make the generated narratives more complaint-faithful (Experiment-B-adjacent), carefully.**~~
  **DONE (2026-06-20, §5c Track 3)** — the case-gen prompt now requires every complaint voiced in
  lay words, and `validate_grounding` gained a completeness check + a stay-lay clinical-jargon
  guard (both independent of the eval map). The clean headline still needs the 250-case regen.
  Original rationale retained for the record:
  The ~10–13pp NL gap is partly that narratives use lay paraphrase and sometimes drop secondary
  tabular terms. **Fair to fix:** ensure the narrative *completely and faithfully* conveys what a
  patient could plausibly state, in natural lay language (removes the dropped-info artifact;
  improves realism). **NOT fair:** make the narrative use the tabular *clinical* terminology
  (`subdural hematoma` instead of `brain bleed`) so tokens match — that is "teaching to the test":
  real patients speak lay language, so pre-clinicalizing the input moves the lay→clinical bridge
  *out* of the system-under-test and inflates the parser numbers vs deployment. `feature_vector`
  with the exact clinical complaint is an **upper bound by design**; the gap is the real cost of
  free-text intake and should be closed on the **system** side (§5b roadmap), not the input side.
  A clearly-labeled *clinical-phrasing* narrative variant is fine as an **upper-bound diagnostic**
  (decomposing vocabulary-vs-information loss) — just never as the headline number.
- **MedGemma 27B** (A100 + 4-bit AWQ) for a stronger comparison point.
- **Optional agentic retry** with vLLM `--enable-auto-tool-choice --tool-call-parser pythonic`
  — expected to stay unreliable for a non-function-tuned 4B model (a reportable finding either way).

---

## 8. File reference
| Path | Role |
|---|---|
| `src/proiect_licenta/llm_config.py` | `get_llm()` / `get_parse_llm()` backend selection |
| `src/proiect_licenta/preprocessing.py` | `clinicalize_complaint()` + `LAY_TO_CLINICAL` (§5c Track 2, parser-output only) |
| `artifacts/benchmarks/clinicalize_2x2_flash_n250.json` | curated audit of the §5c Flash 2×2 **headline** (+ raw dumps in `artifacts/benchmarks/headline_n250/`) |
| `artifacts/benchmarks/clinicalize_2x2_medgemma_n250.json` | curated audit of the §5c MedGemma 2×2 (+ raw dumps in `artifacts/benchmarks/headline_n250_medgemma/`) |
| `artifacts/benchmarks/comparison_flash_vs_medgemma_n250.json` | combined Flash-vs-MedGemma 2×2 (all targets, both backends, ceiling, deltas) |
| `artifacts/benchmarks/clinicalize_2x2_flash_n150.json` | the earlier n150 quick-read (+ raw dumps in `artifacts/benchmarks/quickread/`) |
| `src/proiect_licenta/crew.py` | `llm=get_llm()` on the 4 live agents |
| `src/proiect_licenta/case_generation.py` | `llm=get_llm()` on case generator; `out_dir` param |
| `src/proiect_licenta/main.py` | `generate_cases --backend` |
| `benchmarks/benchmark_pipeline_e2e.py` | `--llm-backend`, `parser-llm` mode, NL-fidelity in `--dump-json` |
| `benchmarks/compare_case_generation.py` | Experiment B grounding comparison |
| `colab/serve_medgemma_colab.ipynb` + `colab/README.md` | self-hosting walk-through |
| `.env` | `LLM_BACKEND` + `MEDGEMMA_*` |
