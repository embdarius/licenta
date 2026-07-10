// Maps a finished prediction tool's JSON output to the matching rich
// visualization (reusing the Phase-1 components) and renders it inline in the
// conversation, so the live runtime shows the same polished result cards.
import AcuityGauge from "./AcuityGauge";
import DispositionCard from "./DispositionCard";
import DispositionFlip from "./DispositionFlip";
import Top3Bars from "./Top3Bars";
import IcdDifferential from "./IcdDifferential";
import ComparisonPanel from "./ComparisonPanel";

function Frame({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="ml-10 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="label mb-3">{title}</div>
      {children}
    </div>
  );
}

export default function RichResultCard({
  tool, output, initial,
}: {
  tool: string;
  output: any;
  initial?: any;
}) {
  if (!output || typeof output !== "object") return null;

  if (tool === "triage_prediction_tool" && output.acuity_prediction) {
    const dispo = output.disposition_prediction;
    const admit = String(dispo?.prediction || "").toUpperCase().includes("ADMIT");
    return (
      <Frame title="Triage result">
        <AcuityGauge acuity={output.acuity_prediction} />
        {dispo && (
          <div className="mt-3 flex items-center gap-2 border-t border-slate-100 pt-3 text-sm">
            <span className="label">Screening disposition</span>
            <span className={admit ? "status-admit" : "status-discharge"}>
              {dispo.prediction} · {dispo.confidence}
            </span>
          </div>
        )}
      </Frame>
    );
  }

  if (tool === "doctor_disposition_tool" && output.disposition_prediction) {
    const triageAdmit = !!output.triage_v3_baseline?.triage_admit_input;
    const refined = !!output.disposition_prediction?.is_admitted;
    return (
      <Frame title="Disposition refinement">
        <DispositionFlip triageAdmit={triageAdmit} refinedAdmit={refined} />
        <DispositionCard payload={output} />
      </Frame>
    );
  }

  const isDiagTool = tool === "doctor_prediction_tool_v3_base" ||
    tool === "doctor_prediction_tool_v3";
  if (isDiagTool) {
    if (output.status === "NOT_ADMITTED" || !output.diagnosis_prediction) {
      return (
        <Frame title="Assessment">
          <p className="text-sm text-slate-600">
            {output.message || "Discharge recommended - no admission diagnosis generated."}
          </p>
        </Frame>
      );
    }
    const enhanced = tool === "doctor_prediction_tool_v3";
    return (
      <Frame title={enhanced ? "Enhanced assessment" : "Initial assessment"}>
        <div className="grid gap-6 sm:grid-cols-2">
          <div>
            <div className="label mb-2">Diagnosis category</div>
            <Top3Bars entries={output.diagnosis_prediction.top_3_categories} />
          </div>
          <div>
            <div className="label mb-2">Department</div>
            <Top3Bars entries={output.department_prediction.top_3_departments} showCode />
          </div>
        </div>
        {enhanced && (
          <>
            <IcdDifferential exact={output.diagnosis_prediction.exact_diagnoses} />
            {initial && <ComparisonPanel initial={initial} enhanced={output} />}
          </>
        )}
      </Frame>
    );
  }

  return null;
}
