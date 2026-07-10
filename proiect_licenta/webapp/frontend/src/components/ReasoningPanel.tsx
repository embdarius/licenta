// Collapsible per-task reasoning, available after the fact (not streamed live).
// Populated from the step_callback reasoning captured server-side.
import { useState } from "react";

export default function ReasoningPanel({ steps }: { steps: string[] }) {
  const [open, setOpen] = useState(false);
  if (!steps.length) return null;
  return (
    <div className="ml-10">
      <button
        className="text-xs font-medium text-slate-500 hover:text-slate-700 hover:underline"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Hide reasoning" : `Show reasoning (${steps.length} steps)`}
      </button>
      {open && (
        <ol className="mt-1 space-y-1 border-l-2 border-slate-200 pl-3 text-xs text-slate-500">
          {steps.map((s, i) => (
            <li key={i} className="whitespace-pre-wrap">{s}</li>
          ))}
        </ol>
      )}
    </div>
  );
}
