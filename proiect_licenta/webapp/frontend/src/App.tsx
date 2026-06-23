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
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded bg-clinical text-sm font-bold text-white">
              ED
            </div>
            <div>
              <h1 className="text-sm font-semibold leading-tight text-slate-900">
                Emergency Department Decision Support
              </h1>
              <p className="text-xs text-slate-500">
                Clinical inference pipeline · MIMIC-IV models · research prototype
              </p>
            </div>
          </div>
          <button className="btn-ghost" onClick={newSession}>New patient</button>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-6 py-6">
        <div className="grid gap-8 lg:grid-cols-[230px,1fr]">
          <aside className="lg:sticky lg:top-6 lg:self-start">
            <ProgressRail done={done} active={active} />
          </aside>

          <main className="space-y-5">
            {err && (
              <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {err}
              </div>
            )}

            {/* Stage 1 — Intake */}
            {!triage && (
              <StageCard index={1} title="Patient intake"
                subtitle="Free-text presentation parsed into structured fields for review">
                <IntakePanel busy={busy} onSubmit={runTriage} />
              </StageCard>
            )}

            {/* Stage 2 — Triage */}
            {triage && (
              <StageCard index={2} title="Triage assessment"
                subtitle="ESI acuity and screening disposition (triage v3)"
                aside={
                  <span className={triage.triage_admit ? "status-admit" : "status-discharge"}>
                    {triage.payload.disposition_prediction.prediction} ·{" "}
                    {triage.payload.disposition_prediction.confidence}
                  </span>
                }>
                <AcuityGauge acuity={triage.payload.acuity_prediction} />
                {(triage.payload.vital_signs?.abnormalities_detected?.length > 0 ||
                  triage.payload.prior_history_used?.pmh_categories_fired?.length > 0) && (
                  <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-100 pt-3 text-xs">
                    {triage.payload.vital_signs?.abnormalities_detected?.length > 0 && (
                      <span className="chip border border-amber-200 bg-amber-50 text-amber-800">
                        Abnormal vitals: {triage.payload.vital_signs.abnormalities_detected.join(", ")}
                      </span>
                    )}
                    {triage.payload.prior_history_used?.pmh_categories_fired?.length > 0 && (
                      <span className="chip border border-slate-200 bg-slate-50 text-slate-600">
                        History recognized: {triage.payload.prior_history_used.pmh_categories_fired.join(", ")}
                      </span>
                    )}
                  </div>
                )}
              </StageCard>
            )}

            {/* Stage 3 — Initial assessment */}
            {initial && (
              <StageCard index={3} title="Initial assessment"
                subtitle="Preliminary diagnosis and department (v3-base, triage data only)">
                {admittedInitial ? (
                  <div className="grid gap-6 sm:grid-cols-2">
                    <div>
                      <div className="label mb-2">Diagnosis category</div>
                      <Top3Bars entries={initial.payload.diagnosis_prediction.top_3_categories} />
                    </div>
                    <div>
                      <div className="label mb-2">Department</div>
                      <Top3Bars entries={initial.payload.department_prediction.top_3_departments} showCode />
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-slate-600">
                    Triage screened this patient for <span className="font-medium">discharge</span>;
                    no admission diagnosis is generated at this stage. The post-nurse disposition
                    may revise this verdict.
                  </p>
                )}
              </StageCard>
            )}

            {/* Stage 4 — Nurse */}
            {triage && !nurse && (
              <StageCard index={4} title="Nurse data collection"
                subtitle="Vital signs (one or more readings), cardiac rhythm, medications, history">
                <NurseForm busy={busy} onSubmit={runNurse} />
              </StageCard>
            )}
            {nurse && (
              <StageCard index={4} title="Nurse data recorded"
                subtitle="Vital signs, rhythm, medications and history collected">
                <div className="flex flex-wrap gap-2 text-xs">
                  {Object.entries(nurse.payload.vital_signs)
                    .filter(([, v]) => v != null)
                    .map(([k, v]) => (
                      <span key={k} className="chip border border-slate-200 bg-slate-50 text-slate-600">
                        {k}: {String(v)}
                      </span>
                    ))}
                  {nurse.payload.vital_trajectory && Object.keys(nurse.payload.vital_trajectory).length > 0 && (
                    <span className="chip border border-clinical/30 bg-clinical-light text-clinical-dark">
                      {Math.max(...Object.values(nurse.payload.vital_trajectory).map((a: any) => a.length))} readings · trend captured
                    </span>
                  )}
                  {nurse.payload.medications_raw && (
                    <span className="chip border border-slate-200 bg-slate-50 text-slate-600">
                      medications: {nurse.payload.medications_raw}
                    </span>
                  )}
                </div>
              </StageCard>
            )}

            {/* Stage 5 — Disposition refinement */}
            {dispo && (
              <StageCard index={5} title="Disposition refinement"
                subtitle="Calibrated admit/discharge from all post-nurse signals (ECE 0.0036)">
                <DispositionFlip triageAdmit={!!triage?.triage_admit} refinedAdmit={!!dispo.refined_admit} />
                <DispositionCard payload={dispo.payload} />
              </StageCard>
            )}

            {/* Stage 6 — Reassessment */}
            {reassess && (
              <StageCard index={6} title="Reassessment"
                subtitle="Diagnosis and department with nurse data (v3), gated on the refined verdict">
                {admittedFinal ? (
                  <>
                    <div className="grid gap-6 sm:grid-cols-2">
                      <div>
                        <div className="label mb-2">Diagnosis category</div>
                        <Top3Bars entries={reassess.payload.diagnosis_prediction.top_3_categories} />
                      </div>
                      <div>
                        <div className="label mb-2">Department</div>
                        <Top3Bars entries={reassess.payload.department_prediction.top_3_departments} showCode />
                      </div>
                    </div>
                    <IcdDifferential exact={reassess.payload.diagnosis_prediction.exact_diagnoses} />
                    <ComparisonPanel initial={initial?.payload} enhanced={reassess.payload} />
                  </>
                ) : (
                  <p className="text-sm text-slate-600">
                    Final disposition is <span className="font-medium">discharge</span>; no admission
                    workup is generated.
                    {triage?.triage_admit && " The disposition model revised triage's admit decision."}
                  </p>
                )}
              </StageCard>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
