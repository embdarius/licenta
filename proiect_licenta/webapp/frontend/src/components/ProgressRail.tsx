import type { StageKey } from "../types";

const STAGES: { key: StageKey; label: string; sub: string }[] = [
  { key: "intake", label: "Intake", sub: "LLM · NLP parse" },
  { key: "triage", label: "Triage", sub: "ESI acuity + disposition" },
  { key: "initial", label: "Initial Dx", sub: "Doctor v3_base" },
  { key: "nurse", label: "Nurse", sub: "Vitals · meds · PMH" },
  { key: "disposition", label: "Disposition", sub: "Calibrated refine" },
  { key: "reassessment", label: "Reassessment", sub: "Doctor v3 + ICD" },
];

export default function ProgressRail({ done, active }: {
  done: Set<StageKey>;
  active: StageKey;
}) {
  return (
    <nav className="space-y-1">
      {STAGES.map((s, i) => {
        const isDone = done.has(s.key);
        const isActive = s.key === active;
        return (
          <div key={s.key} className="flex items-start gap-3">
            <div className="flex flex-col items-center">
              <div
                className={`flex h-7 w-7 items-center justify-center rounded-full border text-xs font-bold transition ${
                  isDone
                    ? "border-emerald-400/50 bg-emerald-500/20 text-emerald-300"
                    : isActive
                      ? "border-sky-400 bg-sky-500/20 text-sky-200"
                      : "border-white/15 bg-white/5 text-slate-500"
                }`}
              >
                {isDone ? "✓" : i + 1}
              </div>
              {i < STAGES.length - 1 && (
                <div className={`my-0.5 h-6 w-px ${isDone ? "bg-emerald-400/40" : "bg-white/10"}`} />
              )}
            </div>
            <div className="pt-0.5">
              <div className={`text-sm font-medium ${isActive ? "text-white" : isDone ? "text-slate-200" : "text-slate-500"}`}>
                {s.label}
              </div>
              <div className="text-[11px] text-slate-500">{s.sub}</div>
            </div>
          </div>
        );
      })}
    </nav>
  );
}
