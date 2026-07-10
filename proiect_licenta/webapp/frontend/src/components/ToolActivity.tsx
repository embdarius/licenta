// A compact chip showing a tool call in real time: a spinner while running, a
// check when done. Expandable to reveal the raw args/output.
import { useState } from "react";

const TOOL_LABELS: Record<string, string> = {
  ask_patient: "Asking the patient",
  confirm_intake: "Confirming intake",
  triage_prediction_tool: "Triage model",
  patient_history_lookup_tool: "EHR history lookup",
  doctor_prediction_tool_v3_base: "Diagnosis model (initial)",
  doctor_disposition_tool: "Disposition model",
  doctor_prediction_tool_v3: "Diagnosis model (reassessment)",
  nurse_data_collection: "Nurse data collection",
};

export default function ToolActivity({
  tool, status, args, output,
}: {
  tool: string;
  status: "running" | "done";
  args?: any;
  output?: any;
}) {
  const [open, setOpen] = useState(false);
  const label = TOOL_LABELS[tool] ?? tool;
  const hasDetail = args != null || output != null;
  return (
    <div className="ml-10">
      <button
        className="inline-flex items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-600 hover:bg-slate-100"
        onClick={() => hasDetail && setOpen((v) => !v)}
      >
        {status === "running" ? (
          <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-300 border-t-clinical" />
        ) : (
          <span className="text-emerald-600">✓</span>
        )}
        <span className="font-mono">{label}</span>
        <span className="text-slate-400">{status === "running" ? "running…" : "done"}</span>
        {hasDetail && <span className="text-slate-400">{open ? "▾" : "▸"}</span>}
      </button>
      {open && hasDetail && (
        <pre className="mt-1 max-h-56 overflow-auto rounded-md border border-slate-200 bg-white p-2 text-[11px] leading-snug text-slate-600">
          {JSON.stringify(output ?? args, null, 2)}
        </pre>
      )}
    </div>
  );
}
