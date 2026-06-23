import { useState } from "react";
import { api } from "../api";
import type { IntakeForm } from "../types";

const EXAMPLES = [
  "I'm a 68 year old man, the ambulance brought me in. I've got crushing chest pain and I can't catch my breath. The pain is like an 8. I have heart failure and high blood pressure.",
  "My 7 year old daughter has a fever and a really bad cough since yesterday, we walked in. She seems okay otherwise, maybe a 3 out of 10.",
  "45 year old woman, walked in with terrible stomach pain in the lower right side, throwing up. Pain is a 9. No medical history.",
];

const blankForm = (): IntakeForm => ({
  chief_complaints: "", pain_score: -1, age: 50, gender: "unknown",
  arrival_transport: "walk_in", subject_id: -1, prior_history: "",
  n_prior_admissions: -1,
  ems_vitals: { temperature: null, heartrate: null, resprate: null, o2sat: null, sbp: null, dbp: null },
});

// Free-text intake → one LLM parse call → editable, pre-filled form. The LLM
// seeds the free-text-derived fields (complaints/age/gender/transport/pain);
// the form collects the structured interview fields the live crew would gather
// via ask_patient (MRN, PMH, prior admissions, EMS vitals).
export default function IntakePanel({
  busy, onSubmit,
}: {
  busy: boolean;
  onSubmit: (form: IntakeForm) => void;
}) {
  const [narrative, setNarrative] = useState("");
  const [form, setForm] = useState<IntakeForm>(blankForm());
  const [parsed, setParsed] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const set = <K extends keyof IntakeForm>(k: K, v: IntakeForm[K]) =>
    setForm((f) => ({ ...f, [k]: v }));
  const setVital = (k: string, v: string) =>
    setForm((f) => ({
      ...f,
      ems_vitals: { ...f.ems_vitals, [k]: v === "" ? null : parseFloat(v) },
    }));

  const doParse = async () => {
    if (!narrative.trim()) return;
    setParsing(true); setErr(null);
    try {
      const r = await api.parse(narrative);
      setForm((f) => ({
        ...f,
        chief_complaints: r.chief_complaints || f.chief_complaints,
        pain_score: r.pain_score,
        age: r.age ?? f.age,
        gender: r.gender === "m" ? "male" : r.gender === "f" ? "female" : f.gender,
        arrival_transport: r.arrival_transport,
      }));
      setParsed(true);
    } catch (e: any) {
      setErr(e.message ?? "Parse failed");
    } finally {
      setParsing(false);
    }
  };

  const isEms = ["ambulance", "helicopter"].includes(form.arrival_transport);

  return (
    <div className="space-y-4">
      <div>
        <span className="label">Patient describes their symptoms (free text)</span>
        <textarea
          className="input mt-1 h-24 resize-none"
          placeholder="e.g. I'm 68, the ambulance brought me in with crushing chest pain…"
          value={narrative}
          onChange={(e) => setNarrative(e.target.value)}
        />
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex, i) => (
            <button key={i} className="chip bg-white/5 text-slate-400 hover:bg-white/10"
              onClick={() => setNarrative(ex)}>
              example {i + 1}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button className="btn-primary" onClick={doParse} disabled={parsing || !narrative.trim()}>
          {parsing ? "Parsing with LLM…" : "✨ Parse with LLM"}
        </button>
        <span className="text-xs text-slate-500">
          The LLM maps lay wording → ED terminology. Review &amp; correct below.
        </span>
      </div>
      {err && <p className="text-sm text-rose-300">{err}</p>}

      {(parsed || form.chief_complaints) && (
        <div className="animate-fade-up space-y-3 border-t border-white/10 pt-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block sm:col-span-2">
              <span className="label">Chief complaints (ED terminology)</span>
              <input className="input mt-0.5" value={form.chief_complaints}
                onChange={(e) => set("chief_complaints", e.target.value)} />
            </label>
            <label className="block">
              <span className="label">Pain (0–10, -1 = unknown)</span>
              <input className="input mt-0.5" type="number" value={form.pain_score}
                onChange={(e) => set("pain_score", parseInt(e.target.value) || -1)} />
            </label>
            <label className="block">
              <span className="label">Age</span>
              <input className="input mt-0.5" type="number" value={form.age}
                onChange={(e) => set("age", parseInt(e.target.value) || 0)} />
            </label>
            <label className="block">
              <span className="label">Gender</span>
              <select className="input mt-0.5" value={form.gender}
                onChange={(e) => set("gender", e.target.value)}>
                <option value="male">male</option>
                <option value="female">female</option>
                <option value="unknown">unknown</option>
              </select>
            </label>
            <label className="block">
              <span className="label">Arrival transport</span>
              <select className="input mt-0.5" value={form.arrival_transport}
                onChange={(e) => set("arrival_transport", e.target.value)}>
                <option value="walk_in">walk-in</option>
                <option value="ambulance">ambulance</option>
                <option value="helicopter">helicopter</option>
              </select>
            </label>
            <label className="block">
              <span className="label">MRN (subject_id, -1 if new)</span>
              <input className="input mt-0.5" type="number" value={form.subject_id}
                onChange={(e) => set("subject_id", parseInt(e.target.value) || -1)} />
            </label>
            <label className="block">
              <span className="label">Prior admissions (-1 if unknown)</span>
              <input className="input mt-0.5" type="number" value={form.n_prior_admissions}
                onChange={(e) => set("n_prior_admissions", parseInt(e.target.value) || -1)} />
            </label>
            <label className="block sm:col-span-2">
              <span className="label">Past medical history / chronic conditions</span>
              <input className="input mt-0.5" placeholder="CHF, diabetes, prior stroke…"
                value={form.prior_history}
                onChange={(e) => set("prior_history", e.target.value)} />
            </label>
          </div>

          {isEms && (
            <div className="rounded-xl border border-sky-400/20 bg-sky-500/5 p-3">
              <span className="label">EMS vitals at handoff (ambulance/helicopter only)</span>
              <div className="mt-1.5 grid grid-cols-3 gap-2 sm:grid-cols-6">
                {(["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"] as const).map((k) => (
                  <label key={k} className="block">
                    <span className="label">{k}</span>
                    <input className="input mt-0.5"
                      value={(form.ems_vitals as any)[k] ?? ""}
                      onChange={(e) => setVital(k, e.target.value)} />
                  </label>
                ))}
              </div>
            </div>
          )}

          <button className="btn-primary w-full" disabled={busy || !form.chief_complaints}
            onClick={() => onSubmit(form)}>
            {busy ? "Running triage…" : "Run triage →"}
          </button>
        </div>
      )}
    </div>
  );
}
