import { pct } from "../util";

// Initial (v3-base, triage-only) vs Enhanced (v3, with nurse data) top-1
// diagnosis + department, with the probability change.
export default function ComparisonPanel({
  initial, enhanced,
}: {
  initial: any;
  enhanced: any;
}) {
  if (!initial || initial.status === "NOT_ADMITTED") return null;

  const row = (label: string, a: any, b: any, key: string) => {
    const aLab = a?.[key] ?? "—";
    const bLab = b?.[key] ?? "—";
    const aP = pct(a?.confidence);
    const bP = pct(b?.confidence);
    const delta = bP - aP;
    const changed = aLab !== bLab;
    return (
      <tr className="border-t border-slate-100">
        <td className="py-1.5 pr-3 text-slate-500">{label}</td>
        <td className="py-1.5 pr-3 text-slate-600">
          {aLab} <span className="text-slate-400">{aP.toFixed(1)}%</span>
        </td>
        <td className={`py-1.5 pr-3 ${changed ? "font-semibold text-amber-700" : "text-slate-800"}`}>
          {bLab} <span className="font-normal text-slate-400">{bP.toFixed(1)}%</span>
        </td>
        <td className={`py-1.5 text-right tabular-nums ${delta >= 0 ? "text-emerald-700" : "text-red-700"}`}>
          {delta >= 0 ? "+" : "−"}{Math.abs(delta).toFixed(1)} pp
        </td>
      </tr>
    );
  };

  return (
    <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 px-4 py-3">
      <div className="label mb-1">Initial (triage only) vs reassessed (with nurse data)</div>
      <table className="w-full text-sm">
        <tbody>
          {row("Diagnosis", initial.diagnosis_prediction, enhanced.diagnosis_prediction, "predicted_category")}
          {row("Department", initial.department_prediction, enhanced.department_prediction, "predicted_department")}
        </tbody>
      </table>
    </div>
  );
}
