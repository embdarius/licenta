"""
Ask Patient Tool — CrewAI Tool

Allows an agent to ask the patient a follow-up question and receive their
response. At the terminal the answer is read from stdin; behind the web UI a
``channel`` is injected and the question is routed to the browser instead (see
``proiect_licenta.interaction``). Behavior with no channel is unchanged.
"""

from typing import Any, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from proiect_licenta.interaction import make_ask


class AskPatientInput(BaseModel):
    """Input schema for asking the patient a question."""
    question: str = Field(
        ...,
        description="The question to ask the patient."
    )


class AskPatientTool(BaseTool):
    name: str = "ask_patient"
    description: str = (
        "Ask the patient a follow-up question to gather missing information. "
        "Use this when the patient's initial description is missing important "
        "details such as: pain level (0-10), age, gender, or how they arrived "
        "at the emergency department (ambulance, walk-in, helicopter). "
        "Ask one question at a time. Be friendly and empathetic."
    )
    args_schema: Type[BaseModel] = AskPatientInput
    # Web session channel (set by the live backend). None -> stdin (terminal).
    # Excluded from the LLM-facing args_schema; it is a tool-instance field only.
    channel: Any = None

    def _run(self, question: str) -> str:
        """Ask the patient a question and return their answer."""
        ask = make_ask(self.channel, default_role="Agent")
        answer = ask(question, "text")
        answer = (answer or "").strip()
        return answer if answer else "No answer provided"
