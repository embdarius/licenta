import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { ActiveQuestion, LiveEvent } from "./types";
import AgentRoster, { TASK_META } from "./components/AgentRoster";
import AgentMessage from "./components/AgentMessage";
import UserMessage from "./components/UserMessage";
import ToolActivity from "./components/ToolActivity";
import QuestionPrompt from "./components/QuestionPrompt";
import RichResultCard from "./components/RichResultCard";
import ReasoningPanel from "./components/ReasoningPanel";

type DistributiveOmit<T, K extends keyof any> = T extends any ? Omit<T, K> : never;

type Feed =
  | { id: number; t: "task"; task: string }
  | { id: number; t: "agent"; agent: string; text: string }
  | { id: number; t: "user"; text: string }
  | { id: number; t: "tool"; tool: string; agent: string; status: "running" | "done"; args?: any; output?: any }
  | { id: number; t: "card"; tool: string; output: any; initial?: any }
  | { id: number; t: "reasoning"; steps: string[] }
  | { id: number; t: "final"; text: string };

type FeedInput = DistributiveOmit<Feed, "id">;

export default function App() {
  const [phase, setPhase] = useState<"intro" | "live">("intro");
  const [narrative, setNarrative] = useState("");
  const [sid, setSid] = useState<string | null>(null);
  const [feed, setFeed] = useState<Feed[]>([]);
  const [activeQ, setActiveQ] = useState<ActiveQuestion | null>(null);
  const [doneTasks, setDoneTasks] = useState<Set<string>>(new Set());
  const [activeTask, setActiveTask] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const idRef = useRef(0);
  const reasoningRef = useRef<string[]>([]);          // steps for the active task
  const v3baseRef = useRef<any>(null);                // for the comparison panel
  const esRef = useRef<EventSource | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const nid = () => ++idRef.current;
  const push = (item: FeedInput) =>
    setFeed((f) => [...f, { id: nid(), ...item } as Feed]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feed, activeQ]);

  const EXAMPLES = [
    "I'm a 68 year old man, walked in, with crushing chest pain and shortness of breath. The pain is an 8. I have heart failure and high blood pressure.",
    "My 7 year old daughter has had a high fever and a bad cough since yesterday, we walked in. She's miserable, maybe a 4 out of 10.",
    "45 year old woman, walked in with severe lower-right belly pain and vomiting, pain is a 9, no medical history.",
  ];

  const handleEvent = (ev: LiveEvent) => {
    switch (ev.type) {
      case "task_started":
        reasoningRef.current = [];
        setActiveTask(ev.task ?? null);
        push({ t: "task", task: ev.task ?? "" });
        break;
      case "agent_message":
        push({ t: "agent", agent: ev.agent ?? "Agent", text: ev.text ?? "" });
        break;
      case "tool_started":
        push({ t: "tool", tool: ev.tool ?? "", agent: ev.agent ?? "",
               status: "running", args: ev.args });
        break;
      case "tool_finished":
        setFeed((f) => {
          const copy = [...f];
          for (let i = copy.length - 1; i >= 0; i--) {
            const it = copy[i];
            if (it.t === "tool" && it.tool === ev.tool && it.status === "running") {
              copy[i] = { ...it, status: "done", output: ev.output };
              break;
            }
          }
          return copy;
        });
        if (ev.tool === "doctor_prediction_tool_v3_base") v3baseRef.current = ev.output;
        if (isResultTool(ev.tool) && ev.output && typeof ev.output === "object") {
          push({ t: "card", tool: ev.tool!, output: ev.output,
                 initial: ev.tool === "doctor_prediction_tool_v3" ? v3baseRef.current : undefined });
        }
        break;
      case "question":
        setActiveQ({ seq: ev.seq, kind: ev.kind ?? "text",
                     prompt: ev.prompt ?? "", meta: ev.meta ?? {} });
        break;
      case "reasoning":
        if (ev.thought) reasoningRef.current.push(ev.thought);
        if (ev.tool && !ev.thought) reasoningRef.current.push(`→ used ${ev.tool}`);
        break;
      case "task_completed":
        if (reasoningRef.current.length)
          push({ t: "reasoning", steps: [...reasoningRef.current] });
        if (ev.task) setDoneTasks((s) => new Set(s).add(ev.task!));
        break;
      case "final":
        push({ t: "final", text: ev.result ?? "" });
        break;
      case "error":
        setError(ev.message ?? "An error occurred.");
        break;
      case "done":
        setFinished(true);
        setActiveTask(null);
        esRef.current?.close();
        break;
    }
  };

  const start = async () => {
    if (!narrative.trim()) return;
    const { session_id } = await api.startLive(narrative.trim());
    setSid(session_id);
    setPhase("live");
    push({ t: "user", text: narrative.trim() });
    const es = new EventSource(api.liveStreamUrl(session_id));
    esRef.current = es;
    es.onmessage = (m) => {
      try { handleEvent(JSON.parse(m.data)); } catch { /* ignore */ }
    };
    es.onerror = () => { /* SSE auto-reconnects; ignore transient errors */ };
  };

  const answer = async (text: string) => {
    if (!sid || !activeQ) return;
    const display = activeQ.kind === "intake_form" ? "Confirmed the intake details." : text;
    push({ t: "user", text: display });
    setActiveQ(null);
    try { await api.answerLive(sid, text); }
    catch (e: any) { setError(e.message); }
  };

  const reset = () => {
    esRef.current?.close();
    setPhase("intro"); setSid(null); setFeed([]); setActiveQ(null);
    setDoneTasks(new Set()); setActiveTask(null); setFinished(false);
    setError(null); setNarrative(""); reasoningRef.current = []; v3baseRef.current = null;
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded bg-clinical text-sm font-bold text-white">ED</div>
            <div>
              <h1 className="text-sm font-semibold leading-tight text-slate-900">
                Emergency Department Decision Support
              </h1>
              <p className="text-xs text-slate-500">
                Live multi-agent runtime · MIMIC-IV models · research prototype
              </p>
            </div>
          </div>
          {phase === "live" && <button className="btn-ghost" onClick={reset}>New patient</button>}
        </div>
      </header>

      {phase === "intro" ? (
        <div className="mx-auto max-w-2xl px-6 py-14">
          <h2 className="text-lg font-semibold text-slate-800">Describe the patient's symptoms</h2>
          <p className="mt-1 text-sm text-slate-500">
            The intake specialist will read this, ask any follow-up questions, and the
            care team (triage, physician, nurse) will work through the case live.
          </p>
          <textarea
            className="input mt-4 h-32 resize-none"
            placeholder="e.g. I'm 68, walked in with crushing chest pain and shortness of breath…"
            value={narrative}
            onChange={(e) => setNarrative(e.target.value)}
          />
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400">Examples:</span>
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="chip border border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-100"
                onClick={() => setNarrative(ex)}>{i + 1}</button>
            ))}
          </div>
          <button className="btn-primary mt-4" disabled={!narrative.trim()} onClick={start}>
            Begin assessment
          </button>
        </div>
      ) : (
        <div className="mx-auto max-w-6xl px-6 py-6">
          <div className="grid gap-8 lg:grid-cols-[210px,1fr]">
            <aside className="lg:sticky lg:top-6 lg:self-start">
              <AgentRoster doneTasks={doneTasks} activeTask={activeTask} />
            </aside>

            <main className="space-y-4">
              {error && (
                <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
              )}

              {feed.map((it) => {
                switch (it.t) {
                  case "task":
                    return (
                      <div key={it.id} className="flex items-center gap-3 pt-2">
                        <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                          {TASK_META[it.task]?.label ?? it.task}
                        </span>
                        <span className="h-px flex-1 bg-slate-200" />
                      </div>
                    );
                  case "agent": return <AgentMessage key={it.id} agent={it.agent} text={it.text} />;
                  case "user": return <UserMessage key={it.id} text={it.text} />;
                  case "tool":
                    return <ToolActivity key={it.id} tool={it.tool} status={it.status}
                      args={it.args} output={it.output} />;
                  case "card":
                    return <RichResultCard key={it.id} tool={it.tool} output={it.output} initial={it.initial} />;
                  case "reasoning": return <ReasoningPanel key={it.id} steps={it.steps} />;
                  case "final":
                    return (
                      <div key={it.id} className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
                        <div className="label mb-1 text-emerald-700">Assessment complete</div>
                        <p className="whitespace-pre-wrap text-sm text-slate-700">{it.text}</p>
                      </div>
                    );
                }
              })}

              {activeQ && <QuestionPrompt question={activeQ} onAnswer={answer} />}

              {finished && !activeQ && (
                <div className="pt-2 text-center">
                  <button className="btn-ghost" onClick={reset}>Start a new patient</button>
                </div>
              )}
              <div ref={bottomRef} />
            </main>
          </div>
        </div>
      )}
    </div>
  );
}

function isResultTool(tool?: string): boolean {
  return tool === "triage_prediction_tool" ||
    tool === "doctor_prediction_tool_v3_base" ||
    tool === "doctor_disposition_tool" ||
    tool === "doctor_prediction_tool_v3";
}
