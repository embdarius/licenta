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
`--dump-json <path>` (writes per-mode metrics **and** the NL-fidelity block).

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
the LLM to emit the **clinical** chief-complaint term, not a verbatim paraphrase; (2) a deterministic
lay→clinical synonym map in `normalize_complaint_text`; (3) semantic complaint embeddings
(re-openable: the prior Bio_ClinicalBERT revert was on *exact* complaints, never under paraphrase);
(4) paraphrase data-augmentation at training time. (1)+(2) are an afternoon and are exactly where a
medical-domain model could extend its edge.

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
- **Run Experiment B** (case-generation grounding quality, MedGemma vs Flash) — still pending;
  needs a live MedGemma tunnel for a ~50-min run.
- **Make the generated narratives more complaint-faithful (Experiment-B-adjacent), carefully.**
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
| `src/proiect_licenta/crew.py` | `llm=get_llm()` on the 4 live agents |
| `src/proiect_licenta/case_generation.py` | `llm=get_llm()` on case generator; `out_dir` param |
| `src/proiect_licenta/main.py` | `generate_cases --backend` |
| `benchmarks/benchmark_pipeline_e2e.py` | `--llm-backend`, `parser-llm` mode, NL-fidelity in `--dump-json` |
| `benchmarks/compare_case_generation.py` | Experiment B grounding comparison |
| `colab/serve_medgemma_colab.ipynb` + `colab/README.md` | self-hosting walk-through |
| `.env` | `LLM_BACKEND` + `MEDGEMMA_*` |
