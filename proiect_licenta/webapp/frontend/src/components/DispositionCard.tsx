import { pct } from "../util";

// Calibrated admit/discharge with the decision threshold marked, the
// override-vs-triage status, and the model's reasoning.
export default function DispositionCard({ payload }: { payload: any }) {
  const d = payload.disposition_prediction;
  const base = payload.triage_v3_baseline ?? {};
  const pAdmit = pct(d.p_admit);
  const threshold = pct(d.decision_threshold);
  const admit = d.is_admitted;
  const reasoning: string[] = payload.reasoning ?? [];

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className={admit ? "status-admit" : "status-discharge"}>
          {admit ? "Admit" : "Discharge"}
        </span>
        {base.doctor_overrode_triage ? (
          <span className="chip border border-amber-200 bg-amber-50 text-amber-800">
            Overrides triage ({base.override_direction})
          </span>
        ) : (
          <span className="chip border border-slate-200 bg-slate-50 text-slate-600">
            Confirms triage
          </span>
        )}
      </div>

      <div className="label mb-1">Calibrated probability</div>
      <div className="relative h-7 overflow-hidden rounded border border-slate-200 bg-emerald-50">
        <div className="bar-fill absolute inset-y-0 left-0 bg-red-100" style={{ width: `${pAdmit}%` }} />
        <div className="absolute inset-y-0 w-px bg-slate-700"
          style={{ left: `${threshold}%` }} title={`threshold ${threshold.toFixed(0)}%`} />
        <div className="absolute inset-0 flex items-center justify-between px-2.5 text-xs font-medium">
          <span className="text-red-800">Admit {pAdmit.toFixed(1)}%</span>
          <span className="text-emerald-800">Discharge {(100 - pAdmit).toFixed(1)}%</span>
        </div>
      </div>
      <p className="mt-1.5 text-[11px] leading-snug text-slate-500">
        Decision threshold {threshold.toFixed(0)}% (lowered from 50% to reduce missed
        admissions). {d.calibration_note}
      </p>

      {reasoning.length > 0 && (
        <div className="mt-3">
          <div className="label mb-1.5">Contributing factors</div>
          <ul className="space-y-1 text-xs text-slate-600">
            {reasoning.map((r, i) => (
              <li key={i} className="flex gap-2">
                <span className="mt-1.5 h-1 w-1 flex-none rounded-full bg-slate-400" />
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
