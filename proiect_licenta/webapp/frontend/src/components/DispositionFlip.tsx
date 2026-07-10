// Triage screening verdict vs the post-nurse refined verdict.
export default function DispositionFlip({
  triageAdmit, refinedAdmit,
}: {
  triageAdmit: boolean;
  refinedAdmit: boolean;
}) {
  const flipped = triageAdmit !== refinedAdmit;
  const Cell = ({ admit, label }: { admit: boolean; label: string }) => (
    <div className="flex-1">
      <div className="label mb-1">{label}</div>
      <div className={admit ? "status-admit" : "status-discharge"}>
        {admit ? "Admit" : "Discharge"}
      </div>
    </div>
  );
  return (
    <div className="mb-4 flex items-end gap-4 rounded-md border border-slate-200 bg-slate-50 px-4 py-3">
      <Cell admit={triageAdmit} label="Triage (screening)" />
      <div className="pb-1 text-xs text-slate-400">
        {flipped ? (
          <span className="font-semibold text-amber-700">revised &rarr;</span>
        ) : (
          <span>&rarr;</span>
        )}
      </div>
      <Cell admit={refinedAdmit} label="Reassessed (with nurse data)" />
    </div>
  );
}
