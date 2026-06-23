import { ESI_COLORS, esiColor, pct } from "../util";

// ESI 1-5 acuity prediction: a level tag, the per-level probability bars, and —
// when the top-2 are within 0.10 — a borderline note ("ESI X-Y"), matching the
// live triage task's hedging rule.
export default function AcuityGauge({ acuity }: { acuity: any }) {
  const level = acuity.predicted_esi_level as number;
  const breakdown: Record<string, string> = acuity.probability_breakdown ?? {};
  const rows = Object.entries(breakdown).map(([k, v]) => ({
    lvl: parseInt(k.replace(/\D/g, ""), 10),
    p: pct(v),
  }));
  const sorted = [...rows].sort((a, b) => b.p - a.p);
  const hedge = sorted.length >= 2 && Math.abs(sorted[0].p - sorted[1].p) <= 10;
  const c = ESI_COLORS[level];

  return (
    <div className="grid gap-6 sm:grid-cols-[180px,1fr]">
      <div>
        <div className="label mb-1">Acuity (ESI)</div>
        <div
          className="flex items-center gap-3 rounded-md border px-3 py-2.5"
          style={{ background: c?.bg, borderColor: c?.border }}
        >
          <span className="text-3xl font-bold tabular-nums" style={{ color: esiColor(level) }}>
            {level}
          </span>
          <div className="leading-tight">
            <div className={`text-sm font-semibold ${c?.text ?? "text-slate-700"}`}>{c?.name}</div>
            <div className="text-xs text-slate-500">{acuity.confidence} confidence</div>
          </div>
        </div>
        <p className="mt-2 text-xs leading-snug text-slate-500">{acuity.description}</p>
      </div>

      <div>
        <div className="label mb-2">Level probabilities</div>
        {hedge && (
          <div className="mb-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            <span className="font-semibold">Borderline — ESI {sorted[0].lvl}–{sorted[1].lvl}.</span>{" "}
            Top two levels within 0.10 ({sorted[0].p.toFixed(1)}% vs {sorted[1].p.toFixed(1)}%);
            flagged for clinician review rather than hard-classified.
          </div>
        )}
        <div className="space-y-1.5">
          {rows.map((r) => (
            <div key={r.lvl} className="flex items-center gap-3 text-xs">
              <span className="w-12 text-slate-500">ESI {r.lvl}</span>
              <div className="h-2.5 flex-1 overflow-hidden rounded bg-slate-100">
                <div className="bar-fill h-full rounded"
                  style={{ width: `${Math.max(1, r.p)}%`, background: esiColor(r.lvl) }} />
              </div>
              <span className="w-12 text-right tabular-nums text-slate-500">{r.p.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
