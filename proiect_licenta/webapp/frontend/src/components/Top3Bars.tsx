import type { Top3Entry } from "../types";
import { pct } from "../util";

export default function Top3Bars({
  entries, color = "#1b5e9b", showCode = false,
}: {
  entries: Top3Entry[];
  color?: string;
  showCode?: boolean;
}) {
  if (!entries?.length) return null;
  return (
    <ul className="space-y-2">
      {entries.map((e, i) => {
        const label = e.name ?? e.category ?? e.code ?? "—";
        const code = showCode ? e.code : undefined;
        const p = pct(e.probability ?? e.probability_raw);
        return (
          <li key={i}>
            <div className="mb-1 flex items-baseline justify-between gap-3 text-sm">
              <span className={i === 0 ? "font-semibold text-slate-900" : "text-slate-600"}>
                {label}
                {code && code !== label && (
                  <span className="ml-1.5 text-xs text-slate-400">{code}</span>
                )}
              </span>
              <span className="tabular-nums text-slate-500">{p.toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded bg-slate-100">
              <div className="bar-fill h-full rounded"
                style={{ width: `${Math.max(2, p)}%`, background: color, opacity: i === 0 ? 1 : 0.5 }} />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
