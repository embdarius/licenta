import { pct } from "../util";

// Initial (v3_base, triage-only) vs Enhanced (v3-nurse) top-1 diagnosis +
// department, with the probability delta — shows what the nurse data changed.
export default function ComparisonPanel({
  initial, enhanced,
}: {
  initial: any;
  enhanced: any;
}) {
  if (!initial || initial.status === "NOT_ADMITTED") return null;

  const row = (
    label: string,
    a: any,
    b: any,
    aLabelKey: string,
    bLabelKey: string,
  ) => {
    const aLab = a?.[aLabelKey] ?? "—";
    const bLab = b?.[bLabelKey] ?? "—";
    const aP = pct(a?.confidence);
    const bP = pct(b?.confidence);
    const delta = bP - aP;
    const changed = aLab !== bLab;
    return (
      <div className="grid grid-cols-[80px,1fr,16px,1fr] items-center gap-2 text-sm">
        <span className="label">{label}</span>
        <span className="text-slate-400">
          {aLab} <span className="text-xs text-slate-500">{aP.toFixed(1)}%</span>
        </span>
        <span className="text-slate-500">→</span>
        <span className={changed ? "font-semibold text-amber-200" : "text-white"}>
          {bLab} <span className="text-xs text-slate-500">{bP.toFixed(1)}%</span>
          <span
            className={`ml-1 text-xs ${delta >= 0 ? "text-emerald-300" : "text-rose-300"}`}
          >
            {delta >= 0 ? "▲" : "▼"}
            {Math.abs(delta).toFixed(1)}
          </span>
        </span>
      </div>
    );
  };

  return (
    <div className="mt-4 space-y-2 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="label mb-1">Initial (triage-only) → Enhanced (with nurse data)</div>
      {row("Diagnosis", initial.diagnosis_prediction, enhanced.diagnosis_prediction,
        "predicted_category", "predicted_category")}
      {row("Department", initial.department_prediction, enhanced.department_prediction,
        "predicted_department", "predicted_department")}
    </div>
  );
}
