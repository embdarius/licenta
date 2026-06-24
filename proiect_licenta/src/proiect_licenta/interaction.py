"""Interactive I/O indirection for the agent tools.

The live runtime tools (``ask_patient``, ``nurse_data_collection``,
``confirm_intake``) need to ask the patient questions. At the terminal they use
``input()``; behind the web UI they must instead emit the question to the browser
and block until an answer is POSTed back. This module decouples the tools from
*how* the asking happens:

- A tool that has no channel (the default, e.g. ``uv run run_crew``) falls back
  to ``print()/input()`` — the terminal behavior is preserved.
- A tool given a ``channel`` (an object implementing :class:`InteractiveChannel`)
  routes every question through it; the web backend's ``SessionChannel`` emits a
  ``question`` event and blocks the crew thread until the user answers.

Tools call ``ask = make_ask(self.channel)`` once, then ``ask(prompt, kind=...)``
for each question. ``kind`` is a UI hint (``"text"``, ``"number"``, ``"bp"``,
``"rhythm"``, ``"integer"``, ``"yesno"``, ``"intake_form"``) so the web client can
render the right input widget; the terminal ignores it.
"""
from __future__ import annotations

import contextvars
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Fallback channel resolution. The live backend sets this ContextVar at the top
# of each crew thread; ``make_ask`` uses it when a tool instance wasn't given a
# channel directly (e.g. if CrewAI ever clones tool instances). Constructor
# injection is the primary path; this is belt-and-suspenders.
_current_channel: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "current_interactive_channel", default=None
)


def set_current_channel(channel: Any) -> contextvars.Token:
    """Bind the active session channel for the current context (crew thread)."""
    return _current_channel.set(channel)


def reset_current_channel(token: contextvars.Token) -> None:
    _current_channel.reset(token)


def resolve_channel(channel: Any = None) -> Any:
    """Return the effective channel: the explicit one, else the context one."""
    return channel if channel is not None else _current_channel.get()


@runtime_checkable
class InteractiveChannel(Protocol):
    """What a tool needs from a web session channel to be interactive."""

    def ask(self, kind: str, prompt: str, meta: Optional[dict] = None) -> str:
        """Emit a question and block until the user's answer string arrives."""
        ...

    def emit(self, event: dict) -> None:
        """Emit a non-blocking status/info event (optional)."""
        ...


# ask(prompt, kind="text", role=None, meta=None) -> raw answer string
AskFn = Callable[..., str]


def make_ask(channel: Any = None, *, default_role: str = "Nurse") -> AskFn:
    """Return an ``ask`` callable bound to ``channel`` (or stdin when ``None``).

    The returned string is the user's RAW answer; callers apply their own
    parsing (``_parse_numeric`` / ``_parse_bp`` / skip-word checks), exactly as
    they did around the old ``input()`` calls — so behavior is unchanged.
    """

    def ask(prompt: str, kind: str = "text", *,
            role: Optional[str] = None, meta: Optional[dict] = None) -> str:
        ch = channel if channel is not None else _current_channel.get()
        if ch is not None:
            m = dict(meta or {})
            m.setdefault("role", role or default_role)
            return ch.ask(kind, prompt, m)
        # Terminal fallback — preserve the existing CLI framing.
        print(f"\n  [{role or default_role}]: {prompt}")
        return input("  [You]:   ")

    return ask
