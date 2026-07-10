// Stage-2 exact-ICD differential (retrieval cascade). Advisory only; renders
// nothing when the resolver produced no block.
export default function IcdDifferential({ exact }: { exact: any }) {
  const ranking: any[] = exact?.flat_ranking ?? [];
  if (!ranking.length) return null;
  return (
    <div className="mt-4 rounded-md border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2">
        <span className="label">Likely exact diagnoses (advisory)</span>
        <span className="chip border border-slate-200 bg-slate-50 text-slate-500">
          Retrieval cascade - does not alter category/department
        </span>
      </div>
      <ul className="divide-y divide-slate-100">
        {ranking.slice(0, 6).map((d, i) => (
          <li key={i} className="flex items-baseline gap-3 px-4 py-1.5 text-sm">
            <span className="w-12 font-mono text-xs text-slate-500">{d.icd_3char}</span>
            <span className="flex-1 text-slate-700">{d.title}</span>
            {d.category && <span className="text-xs text-slate-400">{d.category}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
