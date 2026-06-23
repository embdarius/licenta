import { ESI_COLORS, esiColor, pct } from "../util";

// Renders the ESI 1-5 acuity prediction: a level dial, the per-level
// probability bars, and — when the top-2 are within 0.10 — a borderline hedge
// banner ("ESI X-Y"), matching the live triage task's hedging rule.
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
    <div className="grid gap-5 sm:grid-cols-[auto,1fr] sm:items-center">
      <div className="flex flex-col items-center justify-center">
        <div
          className="flex h-24 w-24 flex-col items-center justify-center rounded-2xl border-2"
          style={{ borderColor: esiColor(level), boxShadow: `0 0 28px ${esiColor(level)}55` }}
        >
          <span className="label">ESI</span>
          <span className="text-4xl font-black" style={{ color: esiColor(level) }}>
            {level}
          </span>
        </div>
        <span className={`mt-2 text-sm font-medium ${c?.text ?? "text-slate-300"}`}>
          {c?.name}
        </span>
        <span className="text-xs text-slate-500">{acuity.confidence} conf.</span>
      </div>

      <div>
        {hedge && (
          <div className="mb-3 rounded-lg border border-amber-400/30 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">
            <b>Borderline case — ESI {sorted[0].lvl}–{sorted[1].lvl}.</b> Top two
            acuity levels are within 0.10 ({sorted[0].p.toFixed(1)}% vs{" "}
            {sorted[1].p.toFixed(1)}%); flagged rather than hard-classified.
          </div>
        )}
        <div className="space-y-1.5">
          {rows.map((r) => (
            <div key={r.lvl} className="flex items-center gap-2 text-xs">
              <span className="w-10 text-slate-400">ESI {r.lvl}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/5">
                <div
                  className="bar-fill h-full rounded-full"
                  style={{ width: `${Math.max(1, r.p)}%`, background: esiColor(r.lvl) }}
                />
              </div>
              <span className="w-12 text-right tabular-nums text-slate-400">
                {r.p.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-slate-400">{acuity.description}</p>
      </div>
    </div>
  );
}
