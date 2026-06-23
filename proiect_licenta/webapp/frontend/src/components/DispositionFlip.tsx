// The hero visual: triage screening verdict vs the post-nurse refined verdict,
// side by side — the thesis's central before/after-nurse comparison.
export default function DispositionFlip({
  triageAdmit, refinedAdmit,
}: {
  triageAdmit: boolean;
  refinedAdmit: boolean;
}) {
  const flipped = triageAdmit !== refinedAdmit;
  const Pill = ({ admit, label }: { admit: boolean; label: string }) => (
    <div className="flex-1 text-center">
      <div className="label mb-1">{label}</div>
      <div
        className={`rounded-xl border px-3 py-2 text-sm font-bold ${
          admit
            ? "border-rose-400/40 bg-rose-500/15 text-rose-200"
            : "border-emerald-400/40 bg-emerald-500/15 text-emerald-200"
        }`}
      >
        {admit ? "ADMIT" : "DISCHARGE"}
      </div>
    </div>
  );
  return (
    <div className="flex items-center gap-3">
      <Pill admit={triageAdmit} label="Triage (screening)" />
      <div className="flex flex-col items-center text-slate-400">
        <span className={`text-2xl ${flipped ? "text-amber-300" : ""}`}>→</span>
        {flipped && <span className="text-[10px] text-amber-300">flipped</span>}
      </div>
      <Pill admit={refinedAdmit} label="Doctor (refined)" />
    </div>
  );
}
