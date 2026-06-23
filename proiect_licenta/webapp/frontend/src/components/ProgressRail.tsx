import type { StageKey } from "../types";

const STAGES: { key: StageKey; label: string; sub: string }[] = [
  { key: "intake", label: "Intake", sub: "NLP parse" },
  { key: "triage", label: "Triage", sub: "ESI acuity + disposition" },
  { key: "initial", label: "Initial assessment", sub: "Diagnosis v3-base" },
  { key: "nurse", label: "Nurse data", sub: "Vitals, medications, history" },
  { key: "disposition", label: "Disposition", sub: "Calibrated admit/discharge" },
  { key: "reassessment", label: "Reassessment", sub: "Diagnosis v3 + ICD" },
];

export default function ProgressRail({ done, active }: {
  done: Set<StageKey>;
  active: StageKey;
}) {
  return (
    <nav aria-label="Pipeline stages">
      <div className="label mb-3 px-1">Workflow</div>
      <ol>
        {STAGES.map((s, i) => {
          const isDone = done.has(s.key);
          const isActive = s.key === active;
          return (
            <li key={s.key} className="flex items-stretch gap-3">
              <div className="flex flex-col items-center">
                <span
                  className={`flex h-6 w-6 items-center justify-center rounded-full border text-[11px] font-semibold ${
                    isDone
                      ? "border-clinical bg-clinical text-white"
                      : isActive
                        ? "border-clinical bg-white text-clinical"
                        : "border-slate-300 bg-white text-slate-400"
                  }`}
                >
                  {isDone ? "✓" : i + 1}
                </span>
                {i < STAGES.length - 1 && (
                  <span className={`my-1 w-px flex-1 ${isDone ? "bg-clinical/40" : "bg-slate-200"}`} />
                )}
              </div>
              <div className="pb-4 pt-0.5">
                <div className={`text-sm leading-tight ${
                  isActive ? "font-semibold text-slate-900" : isDone ? "text-slate-700" : "text-slate-400"
                }`}>
                  {s.label}
                </div>
                <div className="text-[11px] text-slate-400">{s.sub}</div>
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
