from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from proiect_licenta.tools.triage_tool import TriagePredictionTool
from proiect_licenta.tools.ask_patient_tool import AskPatientTool
from proiect_licenta.tools.nurse_tool import NurseDataCollectionTool

# Doctor v3 tier — full swap from v1/v2 (2026-05-28).
# - v3_base for the pre-nurse initial assessment (13-class label space).
# - v3 (nurse) for the post-nurse diagnosis/department reassessment.
# - new disposition tool for the post-nurse admit/discharge refinement
#   (plan section 3, Option B). The reassessment task gates on THIS tool's
#   is_admitted flag rather than the triage one.
# v1/v2 tool files stay on disk for thesis benchmarks but are no longer
# registered with the live crew.
from proiect_licenta.tools.doctor_tool_v3_base import DoctorPredictionToolV3Base
from proiect_licenta.tools.doctor_tool_v3 import DoctorPredictionToolV3
from proiect_licenta.tools.doctor_disposition_tool import DoctorDispositionTool


@CrewBase
class ProiectLicenta():
    """ProiectLicenta crew — Multi-Agent Medical Decision Support System"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ── Agents ────────────────────────────────────────────────

    @agent
    def nlp_parser(self) -> Agent:
        """NLP Parser Agent — conducts patient intake interview."""
        return Agent(
            config=self.agents_config['nlp_parser'],  # type: ignore[index]
            tools=[AskPatientTool()],
            verbose=True,
        )

    @agent
    def triage_agent(self) -> Agent:
        """Triage Agent — predicts ESI acuity + admission using triage v3."""
        return Agent(
            config=self.agents_config['triage_agent'],  # type: ignore[index]
            tools=[TriagePredictionTool()],
            verbose=True,
        )

    @agent
    def doctor_agent(self) -> Agent:
        """Doctor Agent — runs three tasks: initial (v3_base), disposition
        refinement (new), reassessment (v3 with nurse data)."""
        return Agent(
            config=self.agents_config['doctor_agent'],  # type: ignore[index]
            tools=[
                DoctorPredictionToolV3Base(),
                DoctorDispositionTool(),
                DoctorPredictionToolV3(),
            ],
            verbose=True,
        )

    @agent
    def nurse_agent(self) -> Agent:
        """Nurse Agent — collects vital signs and medication history."""
        return Agent(
            config=self.agents_config['nurse_agent'],  # type: ignore[index]
            tools=[NurseDataCollectionTool()],
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────

    @task
    def parse_symptoms_task(self) -> Task:
        """Parse patient's free-text input into structured complaints + pain."""
        return Task(
            config=self.tasks_config['parse_symptoms_task'],  # type: ignore[index]
        )

    @task
    def triage_assessment_task(self) -> Task:
        """Predict acuity (with hedged top-2 when borderline) and disposition."""
        return Task(
            config=self.tasks_config['triage_assessment_task'],  # type: ignore[index]
        )

    @task
    def doctor_assessment_task(self) -> Task:
        """Initial doctor assessment (v3_base) — diagnosis + department + top-3."""
        return Task(
            config=self.tasks_config['doctor_assessment_task'],  # type: ignore[index]
        )

    @task
    def nurse_data_collection_task(self) -> Task:
        """Nurse collects vital signs, medications, rhythm, and PMH."""
        return Task(
            config=self.tasks_config['nurse_data_collection_task'],  # type: ignore[index]
        )

    @task
    def doctor_disposition_task(self) -> Task:
        """NEW — Doctor disposition refinement (plan section 3, Option B).

        Runs after nurse data is collected, BEFORE the diagnosis/department
        reassessment. Uses all signals (triage softmax + vitals + meds +
        longitudinal + PMH) to produce a calibrated admit probability and
        binary decision. The reassessment task gates on this task's
        is_admitted, not triage's.
        """
        return Task(
            config=self.tasks_config['doctor_disposition_task'],  # type: ignore[index]
        )

    @task
    def doctor_reassessment_task(self) -> Task:
        """Enhanced doctor assessment (v3 nurse) — gated on the NEW disposition's verdict."""
        return Task(
            config=self.tasks_config['doctor_reassessment_task'],  # type: ignore[index]
        )

    # ── Crew ──────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Creates the ProiectLicenta crew (4 agents, 6 tasks)."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
