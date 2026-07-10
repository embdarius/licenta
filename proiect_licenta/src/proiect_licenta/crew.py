from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from proiect_licenta.tools.triage_tool import TriagePredictionTool
from proiect_licenta.tools.ask_patient_tool import AskPatientTool
from proiect_licenta.tools.nurse_tool import NurseDataCollectionTool
from proiect_licenta.tools.confirm_intake_tool import ConfirmIntakeTool

# get_llm() returns None for the default `flash` backend (CrewAI's own default),
# so the Gemini path is unchanged unless LLM_BACKEND=medgemma is set.
from proiect_licenta.llm_config import get_llm

# Live doctor tier is v3: v3_base for the pre-nurse initial assessment, the
# disposition tool for the post-nurse admit/discharge refinement, and v3 (nurse)
# for the reassessment. The reassessment gates on the disposition tool's
# is_admitted, not the triage one. v1/v2 tools stay on disk as baselines.
from proiect_licenta.tools.doctor_tool_v3_base import DoctorPredictionToolV3Base
from proiect_licenta.tools.doctor_tool_v3 import DoctorPredictionToolV3
from proiect_licenta.tools.doctor_disposition_tool import DoctorDispositionTool

# Returning-patient lookup: fetches the real prior-encounter PMH block keyed on
# subject_id, recovering the days-since-last-visit / same-complaint numerics the
# bedside interview can't. No-op for first-time/unknown patients.
from proiect_licenta.tools.patient_history_lookup_tool import PatientHistoryLookupTool


@CrewBase
class ProiectLicenta():
    """Multi-agent medical decision support crew (4 agents, 6 tasks)."""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def nlp_parser(self) -> Agent:
        """NLP parser agent that conducts the patient intake interview."""
        return Agent(
            config=self.agents_config['nlp_parser'],  # type: ignore[index]
            tools=[AskPatientTool(), ConfirmIntakeTool()],
            llm=get_llm(),
            verbose=True,
        )

    @agent
    def triage_agent(self) -> Agent:
        """Triage agent that predicts ESI acuity and admission using triage v3.

        Also carries PatientHistoryLookupTool so a returning patient's real
        prior-encounter PMH block reaches the triage model, the same way the
        doctor disposition and reassessment tasks do.
        """
        return Agent(
            config=self.agents_config['triage_agent'],  # type: ignore[index]
            tools=[TriagePredictionTool(), PatientHistoryLookupTool()],
            llm=get_llm(),
            verbose=True,
        )

    @agent
    def doctor_agent(self) -> Agent:
        """Doctor agent that runs the initial, disposition, and reassessment tasks."""
        return Agent(
            config=self.agents_config['doctor_agent'],  # type: ignore[index]
            tools=[
                DoctorPredictionToolV3Base(),
                PatientHistoryLookupTool(),
                DoctorDispositionTool(),
                DoctorPredictionToolV3(),
            ],
            llm=get_llm(),
            verbose=True,
        )

    @agent
    def nurse_agent(self) -> Agent:
        """Nurse agent that collects vital signs and medication history."""
        return Agent(
            config=self.agents_config['nurse_agent'],  # type: ignore[index]
            tools=[NurseDataCollectionTool()],
            llm=get_llm(),
            verbose=True,
        )

    @task
    def parse_symptoms_task(self) -> Task:
        """Parse the patient's free text into structured complaints and pain."""
        return Task(
            config=self.tasks_config['parse_symptoms_task'],  # type: ignore[index]
        )

    @task
    def triage_assessment_task(self) -> Task:
        """Predict acuity and disposition."""
        return Task(
            config=self.tasks_config['triage_assessment_task'],  # type: ignore[index]
        )

    @task
    def doctor_assessment_task(self) -> Task:
        """Initial doctor assessment: diagnosis, department, top-3."""
        return Task(
            config=self.tasks_config['doctor_assessment_task'],  # type: ignore[index]
        )

    @task
    def nurse_data_collection_task(self) -> Task:
        """Collect vital signs, medications, rhythm, and PMH."""
        return Task(
            config=self.tasks_config['nurse_data_collection_task'],  # type: ignore[index]
        )

    @task
    def doctor_disposition_task(self) -> Task:
        """Disposition refinement after nurse data, before the reassessment.

        Uses all signals (triage softmax, vitals, meds, longitudinal, PMH) to
        produce a calibrated admit probability. The reassessment task gates on
        this task's is_admitted, not triage's.
        """
        return Task(
            config=self.tasks_config['doctor_disposition_task'],  # type: ignore[index]
        )

    @task
    def doctor_reassessment_task(self) -> Task:
        """Enhanced doctor assessment, gated on the disposition task's verdict."""
        return Task(
            config=self.tasks_config['doctor_reassessment_task'],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Build the crew (4 agents, 6 tasks)."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
