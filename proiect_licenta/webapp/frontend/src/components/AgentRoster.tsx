// Live pipeline rail: the 6 tasks (4 agents) with active/done highlighting,
// driven by task_started / task_completed events.

export const TASK_ORDER = [
  "parse_symptoms_task",
  "triage_assessment_task",
  "doctor_assessment_task",
  "nurse_data_collection_task",
  "doctor_disposition_task",
  "doctor_reassessment_task",
];

export const TASK_META: Record<string, { label: string; agent: string }> = {
  parse_symptoms_task: { label: "Intake parsing", agent: "Intake Specialist" },
  triage_assessment_task: { label: "Triage", agent: "Triage Specialist" },
  doctor_assessment_task: { label: "Initial assessment", agent: "Physician" },
  nurse_data_collection_task: { label: "Nurse data", agent: "Nurse" },
  doctor_disposition_task: { label: "Disposition", agent: "Physician" },
  doctor_reassessment_task: { label: "Reassessment", agent: "Physician" },
};

export default function AgentRoster({
  doneTasks, activeTask,
}: {
  doneTasks: Set<string>;
  activeTask: string | null;
}) {
  return (
    <nav aria-label="Pipeline">
      <div className="label mb-3 px-1">Workflow</div>
      <ol>
        {TASK_ORDER.map((key, i) => {
          const meta = TASK_META[key];
          const done = doneTasks.has(key);
          const active = key === activeTask;
          return (
            <li key={key} className="flex items-stretch gap-3">
              <div className="flex flex-col items-center">
                <span className={`flex h-6 w-6 items-center justify-center rounded-full border text-[11px] font-semibold ${
                  done ? "border-clinical bg-clinical text-white"
                    : active ? "border-clinical bg-white text-clinical"
                      : "border-slate-300 bg-white text-slate-400"
                }`}>
                  {done ? "✓" : i + 1}
                </span>
                {i < TASK_ORDER.length - 1 && (
                  <span className={`my-1 w-px flex-1 ${done ? "bg-clinical/40" : "bg-slate-200"}`} />
                )}
              </div>
              <div className="pb-4 pt-0.5">
                <div className={`flex items-center gap-1.5 text-sm leading-tight ${
                  active ? "font-semibold text-slate-900" : done ? "text-slate-700" : "text-slate-400"
                }`}>
                  {meta.label}
                  {active && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-clinical" />}
                </div>
                <div className="text-[11px] text-slate-400">{meta.agent}</div>
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
