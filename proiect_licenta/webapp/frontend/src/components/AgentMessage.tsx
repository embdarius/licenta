// One agent's message in the transcript. Long task reports are collapsed to a
// few lines with a "Show full report" toggle to keep the conversation readable.
import { useState } from "react";

const AGENT_INITIALS: Record<string, string> = {
  "Medical Intake Specialist": "IN",
  "Emergency Department Triage Specialist": "TR",
  "Emergency Department Physician": "MD",
  "Emergency Department Nurse": "RN",
};

export default function AgentMessage({ agent, text }: { agent: string; text: string }) {
  const [open, setOpen] = useState(false);
  const clean = (text || "").trim();
  const long = clean.length > 320;
  const shown = open || !long ? clean : clean.slice(0, 320) + "…";
  return (
    <div className="flex gap-3">
      <div className="flex h-7 w-7 flex-none items-center justify-center rounded-full bg-clinical text-[11px] font-bold text-white">
        {AGENT_INITIALS[agent] ?? "AI"}
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-0.5 text-xs font-semibold text-slate-700">{agent}</div>
        <div className="whitespace-pre-wrap rounded-lg rounded-tl-none border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
          {shown}
          {long && (
            <button className="ml-1 align-baseline text-xs font-medium text-clinical hover:underline"
              onClick={() => setOpen((v) => !v)}>
              {open ? "Show less" : "Show full report"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
