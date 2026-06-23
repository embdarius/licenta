import type {
  IntakeForm, NurseReading, ParseResponse, StageResponse,
} from "./types";

const BASE = "/api";

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  session: () => post<{ session_id: string }>("/session"),
  parse: (narrative: string) => post<ParseResponse>("/parse", { narrative }),
  triage: (req: IntakeForm & { session_id: string }) =>
    post<StageResponse>("/triage", req),
  doctorInitial: (sid: string) =>
    post<StageResponse>(`/doctor-initial?session_id=${sid}`),
  nurse: (req: {
    session_id: string;
    readings: NurseReading[];
    medications_raw: string | null;
    prior_history: string | null;
    n_prior_admissions: number;
  }) => post<StageResponse>("/nurse", req),
  disposition: (sid: string) =>
    post<StageResponse>(`/disposition?session_id=${sid}`),
  reassessment: (sid: string) =>
    post<StageResponse>(`/reassessment?session_id=${sid}`),
};
