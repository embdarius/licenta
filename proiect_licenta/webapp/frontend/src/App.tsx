import { useEffect, useState } from "react";
import { api } from "./api";
import type { IntakeForm, NurseReading, StageKey, StageResponse } from "./types";
import ProgressRail from "./components/ProgressRail";
import StageCard from "./components/StageCard";
import IntakePanel from "./components/IntakePanel";
import AcuityGauge from "./components/AcuityGauge";
import DispositionCard from "./components/DispositionCard";
import DispositionFlip from "./components/DispositionFlip";
import Top3Bars from "./components/Top3Bars";
import NurseForm from "./components/NurseForm";
import IcdDifferential from "./components/IcdDifferential";
import ComparisonPanel from "./components/ComparisonPanel";

export default function App() {
  const [sid, setSid] = useState<string | null>(null);
  const [triage, setTriage] = useState<StageResponse | null>(null);
  const [initial, setInitial] = useState<StageResponse | null>(null);
  const [nurse, setNurse] = useState<StageResponse | null>(null);
  const [dispo, setDispo] = useState<StageResponse | null>(null);
  const [reassess, setReassess] = useState<StageResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const newSession = async () => {
    setTriage(null); setInitial(null); setNurse(null); setDispo(null);
    setReassess(null); setErr(null);
    const r = await api.session();
    setSid(r.session_id);
  };
  useEffect(() => { newSession(); }, []);

  const done = new Set<StageKey>();
  if (triage) { done.add("intake"); done.add("triage"); }
  if (initial) done.add("initial");
  if (nurse) done.add("nurse");
  if (dispo) done.add("disposition");
  if (reassess) done.add("reassessment");
  const active: StageKey = !triage ? "intake"
    : !nurse ? "nurse"
    : !reassess ? "disposition" : "reassessment";

  const runTriage = async (form: IntakeForm) => {
    if (!sid) return;
    setBusy(true); setErr(null);
    try {
      const t = await api.triage({ ...form, session_id: sid });
      setTriage(t);
      const init = await api.doctorInitial(sid);
      setInitial(init);
    } catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };

  const runNurse = async (req: {
    readings: NurseReading[];
    medications_raw: string | null;
    prior_history: string | null;
    n_prior_admissions: number;
  }) => {
    if (!sid) return;
    setBusy(true); setErr(null);
    try {
      const n = await api.nurse({ ...req, session_id: sid });
      setNurse(n);
      const d = await api.disposition(sid);
      setDispo(d);
      const r = await api.reassessment(sid);
      setReassess(r);
    } catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };

  const admittedInitial = initial && initial.triage_admit &&
    initial.payload?.status !== "NOT_ADMITTED";
  const admittedFinal = reassess && reassess.refined_admit &&
    reassess.payload?.status !== "NOT_ADMITTED";

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            ED Decision Support
            <span className="ml-2 text-sky-400">· Live Inference</span>
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Multi-agent pipeline on MIMIC-IV — NLP parsing (LLM) → triage, diagnosis,
            disposition &amp; reassessment (XGBoost). Predictions identical to the live crew.
          </p>
        </div>
        <button className="btn-ghost" onClick={newSession}>↺ New patient</button>
      </header>

      <div className="grid gap-8 lg:grid-cols-[200px,1fr]">
        <aside className="lg:sticky lg:top-8 lg:self-start">
          <ProgressRail done={done} active={active} />
        </aside>

        <main className="space-y-6">
          {err && (
            <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {err}
            </div>
          )}

          {/* Stage 0 — Intake */}
          {!triage && (
            <StageCard index={1} title="Patient intake"
              subtitle="Free text → LLM parses into structured, editable fields">
              <IntakePanel busy={busy} onSubmit={runTriage} />
            </StageCard>
          )}

          {/* Stage 1 — Triage */}
          {triage && (
            <StageCard index={2} title="Triage assessment" accent="#f97316"
              subtitle="ESI acuity + screening disposition (triage v3)">
              <AcuityGauge acuity={triage.payload.acuity_prediction} />
              <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
                <span className="label">Screening disposition</span>
                <span className={`chip ${triage.triage_admit ? "bg-rose-500/20 text-rose-200" : "bg-emerald-500/20 text-emerald-200"}`}>
                  {triage.payload.disposition_prediction.prediction} ·{" "}
                  {triage.payload.disposition_prediction.confidence}
                </span>
                {triage.payload.vital_signs?.abnormalities_detected?.length > 0 && (
                  <span className="chip bg-amber-400/15 text-amber-200">
                    abnormal: {triage.payload.vital_signs.abnormalities_detected.join(", ")}
                  </span>
                )}
                {triage.payload.prior_history_used?.pmh_categories_fired?.length > 0 && (
                  <span className="chip bg-violet-500/15 text-violet-200">
                    PMH fired: {triage.payload.prior_history_used.pmh_categories_fired.join(", ")}
                  </span>
                )}
              </div>
            </StageCard>
          )}

          {/* Stage 2 — Initial diagnosis */}
          {initial && (
            <StageCard index={3} title="Initial doctor assessment" accent="#22d3ee"
              subtitle="Preliminary diagnosis + department (v3_base, triage data only)">
              {admittedInitial ? (
                <div className="grid gap-6 sm:grid-cols-2">
                  <div>
                    <div className="label mb-2">Diagnosis (top 3)</div>
                    <Top3Bars entries={initial.payload.diagnosis_prediction.top_3_categories}
                      color="#22d3ee" />
                  </div>
                  <div>
                    <div className="label mb-2">Department (top 3)</div>
                    <Top3Bars entries={initial.payload.department_prediction.top_3_departments}
                      color="#a78bfa" showCode />
                  </div>
                </div>
              ) : (
                <p className="text-sm text-slate-300">
                  Triage screened this patient for <b>discharge</b>, so no admission
                  diagnosis is generated yet. The post-nurse disposition may still upgrade
                  to admit.
                </p>
              )}
            </StageCard>
          )}

          {/* Stage 3 — Nurse */}
          {triage && !nurse && (
            <StageCard index={4} title="Nurse data collection" accent="#34d399"
              subtitle="Vital signs (one or more readings), rhythm, medications, PMH">
              <NurseForm busy={busy} onSubmit={runNurse} />
            </StageCard>
          )}
          {nurse && (
            <StageCard index={4} title="Nurse data collected" accent="#34d399"
              subtitle="Vitals, rhythm, meds, PMH recorded">
              <div className="flex flex-wrap gap-2 text-xs">
                {Object.entries(nurse.payload.vital_signs)
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <span key={k} className="chip bg-white/5 text-slate-300">{k}: {String(v)}</span>
                  ))}
                {nurse.payload.vital_trajectory && Object.keys(nurse.payload.vital_trajectory).length > 0 && (
                  <span className="chip bg-sky-500/15 text-sky-200">
                    {Math.max(...Object.values(nurse.payload.vital_trajectory).map((a: any) => a.length))} readings · trend captured
                  </span>
                )}
                {nurse.payload.medications_raw && (
                  <span className="chip bg-white/5 text-slate-300">meds: {nurse.payload.medications_raw}</span>
                )}
              </div>
            </StageCard>
          )}

          {/* Stage 4 — Disposition refinement (hero) */}
          {dispo && (
            <StageCard index={5} title="Disposition refinement" accent="#fb7185"
              subtitle="Calibrated admit/discharge using all post-nurse signals (ECE 0.0036)">
              <DispositionFlip triageAdmit={!!triage?.triage_admit} refinedAdmit={!!dispo.refined_admit} />
              <div className="mt-4">
                <DispositionCard payload={dispo.payload} />
              </div>
            </StageCard>
          )}

          {/* Stage 5 — Reassessment */}
          {reassess && (
            <StageCard index={6} title="Enhanced reassessment" accent="#a78bfa"
              subtitle="Diagnosis + department with nurse data (v3), gated on the refined verdict">
              {admittedFinal ? (
                <>
                  <div className="grid gap-6 sm:grid-cols-2">
                    <div>
                      <div className="label mb-2">Diagnosis (top 3)</div>
                      <Top3Bars entries={reassess.payload.diagnosis_prediction.top_3_categories}
                        color="#22d3ee" />
                    </div>
                    <div>
                      <div className="label mb-2">Department (top 3)</div>
                      <Top3Bars entries={reassess.payload.department_prediction.top_3_departments}
                        color="#a78bfa" showCode />
                    </div>
                  </div>
                  <IcdDifferential exact={reassess.payload.diagnosis_prediction.exact_diagnoses} />
                  <ComparisonPanel initial={initial?.payload} enhanced={reassess.payload} />
                </>
              ) : (
                <p className="text-sm text-slate-300">
                  Final disposition is <b>discharge</b>. No admission workup generated.
                  {triage?.triage_admit && " (The disposition model overrode triage's admit.)"}
                </p>
              )}
            </StageCard>
          )}
        </main>
      </div>
    </div>
  );
}
