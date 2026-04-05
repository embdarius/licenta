"""
Ask Patient Tool — CrewAI Tool

Allows an agent to ask the patient a follow-up question
and receive their response via stdin.
"""

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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

    def _run(self, question: str) -> str:
        """Ask the patient a question and return their answer."""
        print(f"\n  [Agent]: {question}")
        answer = input("  [You]:   ")
        return answer.strip() if answer.strip() else "No answer provided"
