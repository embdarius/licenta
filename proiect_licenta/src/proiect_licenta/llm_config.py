"""Central LLM backend selection (Gemini Flash 2.5 vs self-hosted MedGemma).

Single source of truth for which LLM the live CrewAI agents and the offline
case generator use, mirroring the ``paths.py`` centralization pattern. The
backend is chosen with the ``LLM_BACKEND`` environment variable:

    LLM_BACKEND=flash      -> Gemini 2.5 Flash (DEFAULT). ``get_llm()`` returns
                              ``None``, which is the CrewAI Agent default, so the
                              Gemini path is left byte-for-byte unchanged (same
                              baseline benchmark numbers).
    LLM_BACKEND=medgemma   -> MedGemma served behind a vLLM OpenAI-compatible
                              endpoint. The GPU host is irrelevant to this code
                              (Colab / Kaytus / Vertex / local all work) — only
                              the URL + key + served model name matter:
                                MEDGEMMA_BASE_URL  (e.g. https://<tunnel>/v1)
                                MEDGEMMA_API_KEY   (vLLM ignores; "EMPTY" is fine)
                                MEDGEMMA_MODEL     (e.g. google/medgemma-4b-it)

MedGemma has no public managed API — it is open weights (gated on Hugging Face),
so it must be self-hosted. See the project plan for serving instructions.

Only the ``medgemma`` backend injects an explicit ``crewai.LLM``; ``flash``
never does. This guarantees the experiment touches the Gemini pipeline only by
choosing not to override it.
"""
import os

try:  # populate MEDGEMMA_* / LLM_BACKEND from .env without overriding os.environ
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv missing or .env absent — env vars still work
    pass


def llm_backend() -> str:
    """Return the active backend name, normalized."""
    return os.getenv("LLM_BACKEND", "flash").strip().lower()


def get_llm():
    """Return the ``crewai.LLM`` for the active backend, or ``None`` for flash.

    ``None`` == CrewAI Agent default == today's Gemini routing, so wiring this
    into every agent is a no-op while ``LLM_BACKEND`` is unset/``flash``.
    """
    backend = llm_backend()
    if backend in ("", "flash", "gemini"):
        return None
    if backend == "medgemma":
        from crewai import LLM

        base_url = os.getenv("MEDGEMMA_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "LLM_BACKEND=medgemma but MEDGEMMA_BASE_URL is unset. Point it at "
                "your vLLM OpenAI-compatible endpoint, e.g. https://<tunnel>/v1"
            )
        # Force CrewAI's NATIVE OpenAI client (provider="openai") pointed at the vLLM
        # endpoint. We avoid the "openai/<model>" model-string form on purpose: CrewAI
        # only routes that to its native client for *known* OpenAI model names, otherwise
        # it falls back to LiteLLM — which this install (crewai[google-genai]) doesn't
        # ship, raising "Fallback to LiteLLM is not available". Passing provider="openai"
        # explicitly + the bare served model name uses the native OpenAI SDK with our
        # base_url, no LiteLLM needed.
        return LLM(
            model=os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it"),
            provider="openai",
            base_url=base_url,
            api_key=os.getenv("MEDGEMMA_API_KEY", "EMPTY"),
            temperature=0,  # deterministic for reproducible benchmark runs
        )
    raise ValueError(
        f"Unknown LLM_BACKEND={backend!r} (expected 'flash' or 'medgemma')."
    )


def get_parse_llm():
    """Explicit LLM for the benchmark's direct parse-only call (parser_llm mode),
    for BOTH backends. Unlike :func:`get_llm`, this never returns ``None``.

    The flash branch needs an explicit Gemini LLM here because parser_llm is a
    single direct completion (narrative -> JSON), NOT the agentic crew — so using
    an explicit LLM for it does not touch the live pipeline (the crew's flash path
    still goes through ``get_llm() -> None -> CrewAI default``, unchanged). This
    enables a same-mode Flash-vs-MedGemma parser comparison.
    """
    backend = llm_backend()
    if backend in ("", "flash", "gemini"):
        from crewai import LLM

        return LLM(
            model=os.getenv("MODEL", "gemini/gemini-2.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    return get_llm()
