"""LLM backend selection: Gemini Flash 2.5 (default) or self-hosted MedGemma.

Chosen with the LLM_BACKEND environment variable. flash makes get_llm() return
None (CrewAI's Agent default), leaving the Gemini path unchanged. medgemma routes
to a vLLM OpenAI-compatible endpoint configured with MEDGEMMA_BASE_URL,
MEDGEMMA_API_KEY, and MEDGEMMA_MODEL. Only the medgemma backend injects an
explicit crewai.LLM.
"""
import os

try:  # load MEDGEMMA_* / LLM_BACKEND from .env without overriding os.environ
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv missing or .env absent; env vars still work
    pass


def llm_backend() -> str:
    """Return the active backend name, normalized."""
    return os.getenv("LLM_BACKEND", "flash").strip().lower()


def get_llm():
    """Return the crewai.LLM for the active backend, or None for flash.

    None is the CrewAI Agent default (Gemini routing), so wiring this into every
    agent is a no-op while LLM_BACKEND is unset or flash.
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
        # Use CrewAI's native OpenAI client via provider="openai". The
        # "openai/<model>" string form only routes to the native client for known
        # OpenAI model names; anything else falls back to LiteLLM, which this
        # install doesn't ship ("Fallback to LiteLLM is not available"). Passing
        # provider="openai" with the bare served model name avoids that.
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
    """Explicit LLM for the direct parse-only call, for both backends.

    Unlike get_llm, this never returns None. The flash branch needs an explicit
    Gemini LLM because the parse is a single completion, not the agentic crew, so
    it does not touch the live pipeline (the crew's flash path still goes through
    get_llm -> None -> CrewAI default).
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
