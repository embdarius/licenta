// Renders the agent's active question with the right input widget for its
// `kind`, and returns the user's answer string. `intake_form` renders an
// editable confirmation form built from the parsed intake.
import { useState } from "react";
import type { ActiveQuestion } from "../types";

const RHYTHMS = ["sinus", "atrial fibrillation", "atrial flutter", "paced",
  "tachycardia", "bradycardia", "other"];

export default function QuestionPrompt({
  question, onAnswer,
}: {
  question: ActiveQuestion;
  onAnswer: (text: string) => void;
}) {
  const { kind, prompt, meta } = question;
  const role = meta?.role || "Agent";

  return (
    <div className="rounded-lg border border-clinical/40 bg-clinical-light/60 p-3">
      <div className="mb-2 flex items-center gap-2">
        <span className="h-2 w-2 animate-pulse rounded-full bg-clinical" />
        <span className="text-xs font-semibold text-clinical-dark">{role} is asking</span>
      </div>
      <p className="mb-2 text-sm text-slate-800">{prompt}</p>
      {kind === "intake_form"
        ? <IntakeConfirm parsed={meta?.parsed ?? {}} onAnswer={onAnswer} />
        : <SimpleAnswer kind={kind} onAnswer={onAnswer} />}
    </div>
  );
}

function SimpleAnswer({ kind, onAnswer }: { kind: string; onAnswer: (t: string) => void }) {
  const [val, setVal] = useState("");
  const send = (t: string) => onAnswer(t);

  if (kind === "yesno") {
    return (
      <div className="flex gap-2">
        <button className="btn-primary" onClick={() => send("yes")}>Yes</button>
        <button className="btn-ghost" onClick={() => send("no")}>No</button>
      </div>
    );
  }

  if (kind === "rhythm") {
    return (
      <div className="flex flex-wrap items-center gap-2">
        <select className="input max-w-[220px]" value={val} onChange={(e) => setVal(e.target.value)}>
          <option value="">Select rhythm…</option>
          {RHYTHMS.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <button className="btn-primary" disabled={!val} onClick={() => send(val)}>Send</button>
        <button className="btn-ghost" onClick={() => send("skip")}>Skip</button>
      </div>
    );
  }

  const numeric = kind === "number" || kind === "integer";
  const skippable = numeric || kind === "bp";
  return (
    <form className="flex flex-wrap items-center gap-2"
      onSubmit={(e) => { e.preventDefault(); if (val.trim()) send(val.trim()); }}>
      <input
        className="input max-w-xs"
        type={numeric ? "number" : "text"}
        inputMode={numeric ? "decimal" : undefined}
        placeholder={kind === "bp" ? "120/80" : numeric ? "value" : "Type your answer…"}
        value={val}
        onChange={(e) => setVal(e.target.value)}
        autoFocus
      />
      <button className="btn-primary" type="submit" disabled={!val.trim()}>Send</button>
      {skippable && (
        <button className="btn-ghost" type="button" onClick={() => send("skip")}>Skip</button>
      )}
    </form>
  );
}

// Editable confirmation of the parsed intake before triage runs.
function IntakeConfirm({ parsed, onAnswer }: { parsed: any; onAnswer: (t: string) => void }) {
  const ccWasList = Array.isArray(parsed.chief_complaints);
  const [form, setForm] = useState({
    subject_id: parsed.subject_id ?? -1,
    chief_complaints: ccWasList
      ? (parsed.chief_complaints as string[]).join(", ")
      : (parsed.chief_complaints ?? ""),
    pain_score: parsed.pain_score ?? -1,
    age: parsed.age ?? 50,
    gender: parsed.gender ?? "unknown",
    arrival_transport: parsed.arrival_transport ?? "walk_in",
    prior_history: parsed.prior_history ?? "",
    n_prior_admissions: parsed.n_prior_admissions ?? -1,
  });
  const set = (k: string, v: any) => setForm((f) => ({ ...f, [k]: v }));

  const confirm = () => {
    // Rebuild the JSON, preserving the parser's chief_complaints type and any
    // vitals the parser included (ambulance/helicopter).
    const cc = String(form.chief_complaints);
    const out: any = {
      ...parsed,
      subject_id: Number(form.subject_id),
      chief_complaints: ccWasList
        ? cc.split(",").map((s) => s.trim()).filter(Boolean)
        : cc,
      pain_score: Number(form.pain_score),
      age: Number(form.age),
      gender: form.gender,
      arrival_transport: form.arrival_transport,
      prior_history: form.prior_history,
      n_prior_admissions: Number(form.n_prior_admissions),
    };
    onAnswer(JSON.stringify(out));
  };

  const field = (label: string, node: React.ReactNode) => (
    <label className="block">
      <span className="label">{label}</span>
      <div className="mt-0.5">{node}</div>
    </label>
  );

  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <div className="grid gap-3 sm:grid-cols-2">
        {field("Chief complaints",
          <input className="input" value={form.chief_complaints}
            onChange={(e) => set("chief_complaints", e.target.value)} />)}
        {field("Pain (0–10, -1 unknown)",
          <input className="input" type="number" value={form.pain_score}
            onChange={(e) => set("pain_score", e.target.value)} />)}
        {field("Age",
          <input className="input" type="number" value={form.age}
            onChange={(e) => set("age", e.target.value)} />)}
        {field("Gender",
          <select className="input" value={form.gender} onChange={(e) => set("gender", e.target.value)}>
            <option value="male">male</option><option value="female">female</option>
            <option value="m">m</option><option value="f">f</option>
            <option value="unknown">unknown</option>
          </select>)}
        {field("Arrival transport",
          <select className="input" value={form.arrival_transport}
            onChange={(e) => set("arrival_transport", e.target.value)}>
            <option value="walk_in">walk-in</option>
            <option value="ambulance">ambulance</option>
            <option value="helicopter">helicopter</option>
          </select>)}
        {field("MRN (subject_id, -1 if new)",
          <input className="input" type="number" value={form.subject_id}
            onChange={(e) => set("subject_id", e.target.value)} />)}
        {field("Prior admissions (-1 unknown)",
          <input className="input" type="number" value={form.n_prior_admissions}
            onChange={(e) => set("n_prior_admissions", e.target.value)} />)}
        {field("Past medical history",
          <input className="input" value={form.prior_history}
            onChange={(e) => set("prior_history", e.target.value)} />)}
      </div>
      <button className="btn-primary mt-3 w-full" onClick={confirm}>
        Confirm and continue to triage
      </button>
    </div>
  );
}
