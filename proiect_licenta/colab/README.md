# Running the pipeline on MedGemma (Colab Pro)

MedGemma has no managed API — it's open weights, so you self-host it. This folder
serves MedGemma on a Colab Pro GPU behind a vLLM OpenAI-compatible API and tunnels
it to your machine, so the local pipeline can call it by setting `LLM_BACKEND=medgemma`.
Nothing about the Gemini (`flash`) path changes; you flip a switch in `.env`.

The GPU host is interchangeable — these same steps work on Kaytus or any rented GPU;
only the public URL changes. See [`serve_medgemma_colab.ipynb`](serve_medgemma_colab.ipynb).

---

## One-time setup

1. **Accept the license** (free) on Hugging Face: open
   <https://huggingface.co/google/medgemma-4b-it> and click *Agree and access*.
2. **Create a read token**: <https://huggingface.co/settings/tokens>.

## On Colab

1. Upload `serve_medgemma_colab.ipynb` to <https://colab.research.google.com> (or
   File → Upload notebook).
2. **Runtime → Change runtime type → GPU** (L4 recommended; T4 also works for 4B).
3. Run the cells top to bottom:
   - installs vLLM + cloudflared,
   - logs in to Hugging Face (paste your token),
   - starts the vLLM server and waits until it's ready,
   - opens a Cloudflare quick-tunnel and **prints a public URL**,
   - runs a sanity JSON-extraction prompt.
4. **Leave the tab open.** Run the last "keep-alive" cell so Colab is less likely to
   idle-disconnect while you run the benchmark locally. The URL changes every session.

## On your machine (this repo)

Edit `.env` with the values the notebook printed (note the `/v1` suffix):

```
LLM_BACKEND=medgemma
MEDGEMMA_BASE_URL=https://<your>.trycloudflare.com/v1
MEDGEMMA_MODEL=google/medgemma-4b-it
MEDGEMMA_API_KEY=EMPTY
```

Then:

```bash
# 1) Smoke test — 1 case, proves the plumbing + parser bypass end to end:
uv run python benchmarks/benchmark_pipeline_e2e.py --limit 1 --llm-backend medgemma --parser-llm --skip-feature-vector

# 2) Experiment A — full 20-case parser/pipeline comparison:
uv run python benchmarks/benchmark_pipeline_e2e.py --llm-backend medgemma --parser-llm \
    --dump-json artifacts/benchmarks/e2e_medgemma.json
uv run python benchmarks/benchmark_pipeline_e2e.py \
    --dump-json artifacts/benchmarks/e2e_flash.json          # Flash baseline (no server needed)

# 3) Experiment B — case-generation narrative quality:
uv run generate_cases --backend medgemma --limit 5          # writes data/derived/synthetic_cases_medgemma/
uv run python benchmarks/compare_case_generation.py
```

When finished: set `LLM_BACKEND=flash` in `.env` and stop the Colab runtime.

---

## Reading the results

- **Experiment A** prints an accuracy table per mode and per target. The fair
  cross-backend comparison is the **NL-FIDELITY** section (parser extraction vs tabular
  truth) — it's orchestration-independent, so it's valid even though MedGemma runs in
  `--parser-llm` mode while Flash runs the full agentic crew. The `--dump-json` files let
  you diff the two runs.
- **Experiment B** prints the grounding-validator pass rate per backend plus a few
  side-by-side narratives.

## Notes & gotchas

- **`--parser-llm`** is MedGemma's evaluation path: one direct LLM completion does the
  parse step, then the deterministic (LLM-free) tool-direct pipeline scores it. It does
  **not** need tool-calling. To try MedGemma driving the *full agentic crew* instead,
  drop `--parser-llm` and enable tool-calling in the notebook (see its last cell). If the
  crew can't call tools reliably, that's itself a reportable finding — fall back to
  `--parser-llm`.
- **Session limits:** Colab sessions are ephemeral (idle-disconnect ~90 min; max ~12–24 h).
  Run each full benchmark in one sitting; re-tunnel and update `.env` if the session drops.
- **27B variant:** needs an A100 + a 4-bit checkpoint — see the optional cell in the
  notebook. Start with 4B to prove the pipeline; attempt 27B only if an A100 is free.
- **Quick-tunnel privacy:** `trycloudflare.com` URLs are public but unguessable and
  short-lived. Don't paste sensitive patient text through a backend you don't trust;
  the benchmark sends only the synthetic generated narratives.
