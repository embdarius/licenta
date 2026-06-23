import { useState } from "react";
import type { NurseReading } from "../types";

const VITALS: { key: keyof NurseReading; label: string; ph: string }[] = [
  { key: "temperature", label: "Temp (°F)", ph: "98.6" },
  { key: "heartrate", label: "HR (bpm)", ph: "80" },
  { key: "resprate", label: "RR (/min)", ph: "16" },
  { key: "o2sat", label: "O₂ sat (%)", ph: "98" },
  { key: "sbp", label: "SBP", ph: "120" },
  { key: "dbp", label: "DBP", ph: "80" },
];

const emptyReading = (): NurseReading => ({
  temperature: null, heartrate: null, resprate: null, o2sat: null,
  sbp: null, dbp: null, rhythm: "", ts: null,
});

// Web-native equivalent of the stdin nurse tool: collects ONE OR MORE
// chronological reading sets (so the doctor models build real vital trends),
// rhythm per reading, meds, PMH, and prior-admission count.
export default function NurseForm({
  busy, onSubmit,
}: {
  busy: boolean;
  onSubmit: (req: {
    readings: NurseReading[];
    medications_raw: string | null;
    prior_history: string | null;
    n_prior_admissions: number;
  }) => void;
}) {
  const [readings, setReadings] = useState<NurseReading[]>([emptyReading()]);
  const [meds, setMeds] = useState("");
  const [pmh, setPmh] = useState("");
  const [priorAdm, setPriorAdm] = useState("");

  const num = (s: string): number | null => {
    const n = parseFloat(s);
    return Number.isFinite(n) ? n : null;
  };

  const setField = (idx: number, key: keyof NurseReading, raw: string) => {
    setReadings((rs) =>
      rs.map((r, i) =>
        i === idx
          ? { ...r, [key]: key === "rhythm" ? raw : key === "ts" ? num(raw) : num(raw) }
          : r,
      ),
    );
  };

  const submit = () => {
    onSubmit({
      readings,
      medications_raw: meds.trim() || null,
      prior_history: pmh.trim() || null,
      n_prior_admissions: priorAdm.trim() === "" ? -1 : (num(priorAdm) ?? -1),
    });
  };

  return (
    <div className="space-y-4">
      {readings.map((r, idx) => (
        <div key={idx} className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="label">
              {idx === 0 ? "Arrival reading" : `Reading #${idx + 1}`}
            </span>
            {readings.length > 1 && (
              <button
                className="text-xs text-rose-300 hover:underline"
                onClick={() => setReadings((rs) => rs.filter((_, i) => i !== idx))}
              >
                remove
              </button>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            {VITALS.map((v) => (
              <label key={v.key} className="block">
                <span className="label">{v.label}</span>
                <input
                  className="input mt-0.5"
                  placeholder={v.ph}
                  value={(r[v.key] as number | null) ?? ""}
                  onChange={(e) => setField(idx, v.key, e.target.value)}
                />
              </label>
            ))}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <label className="block">
              <span className="label">Cardiac rhythm</span>
              <input
                className="input mt-0.5"
                placeholder="sinus / atrial fibrillation / paced"
                value={r.rhythm ?? ""}
                onChange={(e) => setField(idx, "rhythm", e.target.value)}
              />
            </label>
            <label className="block">
              <span className="label">Time (min since arrival)</span>
              <input
                className="input mt-0.5"
                placeholder="0, 15, 30…"
                value={r.ts ?? ""}
                onChange={(e) => setField(idx, "ts", e.target.value)}
              />
            </label>
          </div>
        </div>
      ))}

      <button
        className="btn-ghost w-full"
        onClick={() => setReadings((rs) => [...rs, emptyReading()])}
      >
        + Add another reading (capture the vital trend)
      </button>

      <div className="grid gap-2 sm:grid-cols-2">
        <label className="block">
          <span className="label">Current medications</span>
          <input className="input mt-0.5" placeholder="aspirin, metoprolol…"
            value={meds} onChange={(e) => setMeds(e.target.value)} />
        </label>
        <label className="block">
          <span className="label">Chronic conditions / PMH</span>
          <input className="input mt-0.5" placeholder="CHF, diabetes, prior stroke"
            value={pmh} onChange={(e) => setPmh(e.target.value)} />
        </label>
      </div>
      <label className="block sm:w-1/2">
        <span className="label">Prior hospital admissions (count)</span>
        <input className="input mt-0.5" placeholder="0, 1, 2…"
          value={priorAdm} onChange={(e) => setPriorAdm(e.target.value)} />
      </label>

      <button className="btn-primary w-full" disabled={busy} onClick={submit}>
        {busy ? "Refining disposition…" : "Submit nurse data → refine disposition"}
      </button>
    </div>
  );
}
