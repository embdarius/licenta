// Advisory Stage-2 exact-ICD differential (the retrieval cascade). Clearly
// labeled experimental; renders nothing when the resolver produced no block.
export default function IcdDifferential({ exact }: { exact: any }) {
  const ranking: any[] = exact?.flat_ranking ?? [];
  if (!ranking.length) return null;
  return (
    <div className="mt-4 rounded-xl border border-violet-400/25 bg-violet-500/5 p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="chip bg-violet-500/20 text-violet-200">Advisory · experimental</span>
        <span className="text-xs text-slate-400">
          Likely exact diagnoses within the predicted categories (retrieval cascade)
        </span>
      </div>
      <ul className="grid gap-1.5 sm:grid-cols-2">
        {ranking.slice(0, 6).map((d, i) => (
          <li key={i} className="flex items-baseline gap-2 text-sm">
            <span className="rounded bg-violet-500/20 px-1.5 py-0.5 font-mono text-xs text-violet-200">
              {d.icd_3char}
            </span>
            <span className="text-slate-200">{d.title}</span>
            {d.category && <span className="text-xs text-slate-500">· {d.category}</span>}
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[11px] text-slate-500">
        Does not change the category/department predictions — surfaced as a suggested
        differential only.
      </p>
    </div>
  );
}
