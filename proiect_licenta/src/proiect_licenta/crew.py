from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from proiect_licenta.tools.triage_tool import TriagePredictionTool
from proiect_licenta.tools.ask_patient_tool import AskPatientTool
from proiect_licenta.tools.doctor_tool import DoctorPredictionTool


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
        """Triage Agent — predicts ESI acuity + admission using ML model."""
        return Agent(
            config=self.agents_config['triage_agent'],  # type: ignore[index]
            tools=[TriagePredictionTool()],
            verbose=True,
        )

    @agent
    def doctor_agent(self) -> Agent:
        """Doctor Agent — predicts diagnosis + department for admitted patients."""
        return Agent(
            config=self.agents_config['doctor_agent'],  # type: ignore[index]
            tools=[DoctorPredictionTool()],
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
        """Predict acuity and disposition using the ML triage model."""
        return Task(
            config=self.tasks_config['triage_assessment_task'],  # type: ignore[index]
        )

    @task
    def doctor_assessment_task(self) -> Task:
        """Predict diagnosis and department for admitted patients."""
        return Task(
            config=self.tasks_config['doctor_assessment_task'],  # type: ignore[index]
        )

    # ── Crew ──────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Creates the ProiectLicenta crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
