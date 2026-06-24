"""
Confirm Intake Tool — CrewAI Tool

The NLP Parser calls this as its final step with the structured intake JSON it
assembled. Behind the web UI an injected ``channel`` surfaces the parsed fields
to the user as an editable form; the user can correct any extraction error, and
the (possibly edited) JSON is returned for the parser to emit verbatim — a
human-in-the-loop checkpoint before triage runs.

At the terminal (no channel) it is a no-op: it returns the JSON unchanged, so
the CLI flow (`uv run run_crew`) is unaffected.
"""

import json
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from proiect_licenta.interaction import make_ask, resolve_channel


class ConfirmIntakeInput(BaseModel):
    """Input schema for the confirm-intake tool."""
    intake_json: str = Field(
        ...,
        description="The structured intake as a JSON string (the same object you "
                    "are about to output): subject_id, chief_complaints, "
                    "pain_score, age, gender, arrival_transport, prior_history, "
                    "n_prior_admissions, and vitals if ambulance/helicopter.",
    )


class ConfirmIntakeTool(BaseTool):
    name: str = "confirm_intake"
    description: str = (
        "Present the structured intake you have assembled to the patient/clinician "
        "for confirmation and correction BEFORE triage. Call this once, as your "
        "final step, passing your assembled intake as a JSON string. It returns "
        "the confirmed (possibly user-corrected) JSON — output exactly what it "
        "returns as your final answer, with no further changes."
    )
    args_schema: Type[BaseModel] = ConfirmIntakeInput
    # Web session channel (set by the live backend). None -> no-op (terminal).
    channel: Any = None

    def _run(self, intake_json: str) -> str:
        # Terminal / no channel: pass through unchanged so the CLI flow is intact.
        channel = resolve_channel(self.channel)
        if channel is None:
            return intake_json

        # Parse what the LLM assembled so the UI can pre-fill an editable form.
        try:
            parsed = json.loads(intake_json)
        except (json.JSONDecodeError, TypeError):
            # Best-effort: still let the user see/edit the raw text.
            parsed = {}

        ask = make_ask(channel, default_role="Intake")
        answer = ask(
            "Please review the information I extracted and correct anything that is wrong.",
            "intake_form",
            meta={"parsed": parsed, "raw": intake_json},
        )

        # The UI returns the confirmed JSON string. If it's valid JSON, return it;
        # otherwise fall back to the original so the pipeline never breaks.
        answer = (answer or "").strip()
        if not answer:
            return intake_json
        try:
            json.loads(answer)
            return answer
        except (json.JSONDecodeError, TypeError):
            return intake_json
