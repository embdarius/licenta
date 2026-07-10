"""Interactive I/O indirection so the agent tools can ask the patient questions.

At the terminal the tools use input(); behind the web UI they route each question
through a channel that emits an event and blocks the crew thread until an answer
is POSTed back. A tool with no channel falls back to print()/input(), preserving
terminal behavior. `kind` is a UI hint (text, number, bp, rhythm, integer, yesno,
intake_form) the web client uses to pick a widget; the terminal ignores it.
"""
from __future__ import annotations

import contextvars
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# The backend sets this ContextVar at the top of each crew thread. make_ask falls
# back to it when a tool instance wasn't given a channel directly (e.g. if CrewAI
# clones tool instances). Constructor injection is the primary path.
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

    The returned string is the user's raw answer; callers do their own parsing.
    """

    def ask(prompt: str, kind: str = "text", *,
            role: Optional[str] = None, meta: Optional[dict] = None) -> str:
        ch = channel if channel is not None else _current_channel.get()
        if ch is not None:
            m = dict(meta or {})
            m.setdefault("role", role or default_role)
            return ch.ask(kind, prompt, m)
        print(f"\n  [{role or default_role}]: {prompt}")
        return input("  [You]:   ")

    return ask
