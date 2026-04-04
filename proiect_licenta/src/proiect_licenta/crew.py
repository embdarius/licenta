from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from proiect_licenta.tools.triage_tool import TriagePredictionTool


@CrewBase
class ProiectLicenta():
    """ProiectLicenta crew — Multi-Agent Medical Triage System"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ── Agents ────────────────────────────────────────────────

    @agent
    def nlp_parser(self) -> Agent:
        """NLP Parser Agent — converts free-text symptoms to structured data."""
        return Agent(
            config=self.agents_config['nlp_parser'],  # type: ignore[index]
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
