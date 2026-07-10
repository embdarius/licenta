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

// Live runtime SSE events + transcript model
export interface LiveEvent {
  seq: number;
  type:
    | "started" | "task_started" | "task_completed" | "task_output"
    | "agent_started" | "agent_message"
    | "tool_started" | "tool_finished"
    | "question" | "reasoning" | "final" | "error" | "done";
  task?: string;
  agent?: string;
  text?: string;
  tool?: string;
  args?: any;
  output?: any;
  kind?: string;
  prompt?: string;
  meta?: any;
  thought?: string;
  result?: string;
  message?: string;
}

export interface ActiveQuestion {
  seq: number;
  kind: string;
  prompt: string;
  meta: any;
}

export type TranscriptItem =
  | { id: number; kind: "task"; task: string }
  | { id: number; kind: "agent"; agent: string; text: string }
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "tool"; tool: string; agent: string; status: "running" | "done"; args?: any; output?: any }
  | { id: number; kind: "card"; tool: string; output: any }
  | { id: number; kind: "final"; text: string };

export type StageKey =
  | "intake"
  | "triage"
  | "initial"
  | "nurse"
  | "disposition"
  | "reassessment";
