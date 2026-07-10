"""Ask-patient tool: lets an agent ask a follow-up question and read the answer.

At the terminal the answer comes from stdin; behind the web UI an injected
channel routes the question to the browser (see proiect_licenta.interaction).
Behavior with no channel is unchanged.
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
