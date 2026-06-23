import { pct } from "../util";

// The calibrated admit/discharge bar with the decision threshold marked, plus
// the override-vs-triage badge and the model's reasoning bullets.
export default function DispositionCard({ payload }: { payload: any }) {
  const d = payload.disposition_prediction;
  const base = payload.triage_v3_baseline ?? {};
  const pAdmit = pct(d.p_admit);
  const threshold = pct(d.decision_threshold);
  const admit = d.is_admitted;
  const reasoning: string[] = payload.reasoning ?? [];

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span
          className={`chip ${admit ? "bg-rose-500/20 text-rose-200" : "bg-emerald-500/20 text-emerald-200"}`}
        >
          {admit ? "● ADMIT" : "● DISCHARGE"}
        </span>
        {base.doctor_overrode_triage ? (
          <span className="chip bg-amber-400/20 text-amber-200">
            ⚑ Overrode triage ({base.override_direction})
          </span>
        ) : (
          <span className="chip bg-white/10 text-slate-300">Confirms triage</span>
        )}
      </div>

      {/* Admit vs discharge probability bar with the threshold tick */}
      <div className="relative mt-3 h-8 overflow-hidden rounded-lg bg-emerald-500/15">
        <div
          className="bar-fill absolute inset-y-0 left-0 bg-rose-500/40"
          style={{ width: `${pAdmit}%` }}
        />
        <div
          className="absolute inset-y-0 w-0.5 bg-white/80"
          style={{ left: `${threshold}%` }}
          title={`decision threshold ${threshold.toFixed(0)}%`}
        />
        <div className="absolute inset-0 flex items-center justify-between px-3 text-xs font-medium">
          <span className="text-rose-100">P(admit) {pAdmit.toFixed(1)}%</span>
          <span className="text-emerald-100">P(discharge) {(100 - pAdmit).toFixed(1)}%</span>
        </div>
      </div>
      <p className="mt-1 text-[11px] text-slate-500">
        White tick = decision threshold ({threshold.toFixed(0)}%, tuned down from 50% to
        reduce missed admissions). {d.calibration_note}
      </p>

      {reasoning.length > 0 && (
        <ul className="mt-3 space-y-1 text-xs text-slate-300">
          {reasoning.map((r, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-sky-400">›</span>
              <span>{r}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
