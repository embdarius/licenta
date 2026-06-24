"""Live conversational runtime over SSE.

Drives the REAL agentic CrewAI crew (the same one as ``uv run run_crew``) per
session, in a background thread, and bridges its interactive ``input()`` calls
and its agent/tool/task telemetry to the browser:

- Each session gets a :class:`SessionChannel`. Its ``ask`` is injected into the
  interactive tools (ask_patient / nurse / confirm_intake); when a tool asks a
  question the crew thread BLOCKS until the browser POSTs an answer.
- CrewAI event-bus listeners (registered once) emit agent/tool/task activity to
  the active session's channel (attributed via a ContextVar that CrewAI's bus
  propagates through ``copy_context``).
- ``step_callback`` / ``task_callback`` capture per-step reasoning and task
  outputs.

Transport: SSE for server->client events, plain POST for client->server answers.
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from proiect_licenta import interaction

router = APIRouter(prefix="/api/live", tags=["live"])

# Sentinel pushed onto the answer queue to abort a pending ask (cancel/teardown).
_ABORT = object()


# ---------------------------------------------------------------------------
# Per-session channel
# ---------------------------------------------------------------------------
class SessionChannel:
    """Thread-safe bridge between the crew thread and the browser."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.events: "queue.Queue[dict]" = queue.Queue()
        self._answer: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self.finished = threading.Event()
        self._seq = 0

    # -- crew thread -> browser ------------------------------------------------
    def emit(self, event: dict) -> None:
        self._seq += 1
        event = {"seq": self._seq, **event}
        self.events.put(event)

    # -- tool asks a question, blocks until the browser answers ---------------
    def ask(self, kind: str, prompt: str, meta: Optional[dict] = None) -> str:
        self.emit({"type": "question", "kind": kind, "prompt": prompt,
                   "meta": meta or {}})
        answer = self._answer.get()  # blocks the crew thread
        if answer is _ABORT:
            raise RuntimeError("Session cancelled")
        return answer

    # -- browser -> crew thread -----------------------------------------------
    def answer(self, text: str) -> None:
        try:
            self._answer.put_nowait(text)
        except queue.Full:
            # An answer is already pending/unconsumed; replace it.
            try:
                self._answer.get_nowait()
            except queue.Empty:
                pass
            self._answer.put_nowait(text)

    def abort(self) -> None:
        try:
            self._answer.put_nowait(_ABORT)
        except queue.Full:
            pass


_SESSIONS: dict[str, SessionChannel] = {}


# ---------------------------------------------------------------------------
# Telemetry — CrewAI event-bus listeners (registered once)
# ---------------------------------------------------------------------------
_LISTENERS_REGISTERED = False


def _active() -> Optional[SessionChannel]:
    ch = interaction.resolve_channel(None)
    return ch if isinstance(ch, SessionChannel) else None


def _task_label(task: Any) -> str:
    name = getattr(task, "name", None)
    if name:
        return str(name)
    desc = getattr(task, "description", "") or ""
    return desc.strip().split("\n")[0][:80] or "task"


def _agent_role(agent: Any) -> str:
    role = getattr(agent, "role", None)
    return (str(role).strip() if role else "Agent")


def register_listeners() -> None:
    """Register global event-bus handlers once. They no-op when no SessionChannel
    is bound to the current context (e.g. CLI runs, other crews)."""
    global _LISTENERS_REGISTERED
    if _LISTENERS_REGISTERED:
        return
    _LISTENERS_REGISTERED = True

    from crewai.events import crewai_event_bus
    from crewai.events.types.task_events import TaskStartedEvent, TaskCompletedEvent
    from crewai.events.types.agent_events import (
        AgentExecutionStartedEvent, AgentExecutionCompletedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageStartedEvent, ToolUsageFinishedEvent,
    )

    @crewai_event_bus.on(TaskStartedEvent)
    def _on_task_started(source, event):  # noqa: ANN001
        ch = _active()
        if ch:
            ch.emit({"type": "task_started", "task": _task_label(event.task)})

    @crewai_event_bus.on(TaskCompletedEvent)
    def _on_task_completed(source, event):  # noqa: ANN001
        ch = _active()
        if ch:
            raw = getattr(event.output, "raw", None) or str(event.output)
            ch.emit({"type": "task_completed", "task": _task_label(event.task),
                     "output": raw})

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def _on_agent_started(source, event):  # noqa: ANN001
        ch = _active()
        if ch:
            ch.emit({"type": "agent_started", "agent": _agent_role(event.agent)})

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def _on_agent_completed(source, event):  # noqa: ANN001
        ch = _active()
        if ch:
            ch.emit({"type": "agent_message", "agent": _agent_role(event.agent),
                     "text": str(getattr(event, "output", "") or "")})

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def _on_tool_started(source, event):  # noqa: ANN001
        ch = _active()
        if ch:
            ch.emit({"type": "tool_started", "tool": event.tool_name,
                     "agent": (event.agent_role or "").strip(),
                     "args": _coerce(event.tool_args)})

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def _on_tool_finished(source, event):  # noqa: ANN001
        ch = _active()
        if ch:
            ch.emit({"type": "tool_finished", "tool": event.tool_name,
                     "agent": (event.agent_role or "").strip(),
                     "output": _coerce(getattr(event, "output", None))})


def _coerce(value: Any) -> Any:
    """Best-effort JSON-friendly view of tool args/output (parse JSON strings)."""
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    s = str(value)
    t = s.strip()
    if t and t[0] in "{[":
        try:
            return json.loads(t)
        except (json.JSONDecodeError, ValueError):
            return s
    return s


# ---------------------------------------------------------------------------
# Reasoning / task callbacks (closures over the channel)
# ---------------------------------------------------------------------------
def _make_step_callback(ch: SessionChannel):
    def step_callback(step: Any) -> None:
        thought = getattr(step, "thought", None)
        text = getattr(step, "text", None)
        tool = getattr(step, "tool", None)
        payload = {"type": "reasoning"}
        if thought:
            payload["thought"] = str(thought).strip()
        if tool:
            payload["tool"] = str(tool)
        if not thought and text:
            payload["thought"] = str(text).strip()[:2000]
        if payload.get("thought") or payload.get("tool"):
            ch.emit(payload)
    return step_callback


def _make_task_callback(ch: SessionChannel):
    def task_callback(task_output: Any) -> None:
        # task_completed is already emitted by the event listener; this is a
        # backstop carrying the final raw output if the listener didn't fire.
        raw = getattr(task_output, "raw", None) or str(task_output)
        ch.emit({"type": "task_output", "output": raw})
    return task_callback


# ---------------------------------------------------------------------------
# Crew runner thread
# ---------------------------------------------------------------------------
def _run_crew(ch: SessionChannel, narrative: str) -> None:
    token = interaction.set_current_channel(ch)
    try:
        from proiect_licenta.crew import ProiectLicenta

        crew = ProiectLicenta().crew()
        # Primary injection: set the channel on the interactive tool instances.
        for agent in crew.agents:
            for tool in getattr(agent, "tools", []) or []:
                if hasattr(tool, "channel"):
                    try:
                        tool.channel = ch
                    except Exception:
                        pass
        crew.step_callback = _make_step_callback(ch)
        crew.task_callback = _make_task_callback(ch)

        ch.emit({"type": "started"})
        result = crew.kickoff(inputs={"user_message": narrative})
        ch.emit({"type": "final", "result": str(result)})
    except Exception as exc:  # noqa: BLE001
        ch.emit({"type": "error", "message": str(exc)})
    finally:
        interaction.reset_current_channel(token)
        ch.emit({"type": "done"})
        ch.finished.set()


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    narrative: str


class AnswerRequest(BaseModel):
    text: str


@router.post("/start")
def start(req: StartRequest) -> dict:
    if not req.narrative.strip():
        raise HTTPException(status_code=400, detail="Empty narrative.")
    register_listeners()
    sid = uuid.uuid4().hex
    ch = SessionChannel(sid)
    _SESSIONS[sid] = ch
    threading.Thread(target=_run_crew, args=(ch, req.narrative),
                     name=f"crew-{sid[:8]}", daemon=True).start()
    return {"session_id": sid}


@router.post("/answer/{session_id}")
def answer(session_id: str, req: AnswerRequest) -> dict:
    ch = _SESSIONS.get(session_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Unknown session.")
    ch.answer(req.text)
    return {"ok": True}


@router.post("/cancel/{session_id}")
def cancel(session_id: str) -> dict:
    ch = _SESSIONS.get(session_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Unknown session.")
    ch.abort()
    return {"ok": True}


@router.get("/stream/{session_id}")
def stream(session_id: str) -> StreamingResponse:
    ch = _SESSIONS.get(session_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Unknown session.")

    import asyncio

    async def gen():
        loop = asyncio.get_event_loop()
        yield ": connected\n\n"
        while True:
            try:
                ev = await loop.run_in_executor(None, ch.events.get, True, 15)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev.get("type") == "done":
                break
        # Allow the session to be GC'd shortly after it ends.
        _SESSIONS.pop(session_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
