// Wire types mirroring webapp/backend/schemas.py. The prediction `payload`s are
// raw tool JSON, typed loosely here and accessed defensively in components.

export interface ParseResponse {
  chief_complaints: string;
  age: number | null;
  gender: string | null;
  arrival_transport: string;
  pain_score: number;
}

export interface EmsVitals {
  temperature?: number | null;
  heartrate?: number | null;
  resprate?: number | null;
  o2sat?: number | null;
  sbp?: number | null;
  dbp?: number | null;
}

export interface IntakeForm {
  chief_complaints: string;
  pain_score: number;
  age: number;
  gender: string;
  arrival_transport: string;
  subject_id: number;
  prior_history: string;
  n_prior_admissions: number;
  ems_vitals: EmsVitals;
}

export interface StageResponse {
  session_id: string;
  stage: string;
  payload: any;
  predicted_acuity: number | null;
  triage_admit: boolean | null;
  refined_admit: boolean | null;
}

export interface NurseReading {
  temperature?: number | null;
  heartrate?: number | null;
  resprate?: number | null;
  o2sat?: number | null;
  sbp?: number | null;
  dbp?: number | null;
  rhythm?: string | null;
  ts?: number | null;
}

export interface Top3Entry {
  category?: string;
  code?: string;
  name?: string;
  probability?: string;
  probability_raw?: number;
}

export type StageKey =
  | "intake"
  | "triage"
  | "initial"
  | "nurse"
  | "disposition"
  | "reassessment";
