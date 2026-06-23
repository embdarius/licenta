// Small display helpers shared across components.

/** Parse a "54.5%" string (or a 0-1 float) into a 0-100 number. */
export function pct(v: string | number | undefined | null): number {
  if (v == null) return 0;
  if (typeof v === "number") return v <= 1 ? v * 100 : v;
  const n = parseFloat(String(v).replace("%", "").trim());
  return Number.isFinite(n) ? n : 0;
}

export function fmtPct(v: string | number | undefined | null, digits = 1): string {
  return `${pct(v).toFixed(digits)}%`;
}

/** ESI level → palette (text/bg/border/ring) using standard triage colors. */
export const ESI_COLORS: Record<number, { name: string; hex: string; text: string }> = {
  1: { name: "Resuscitation", hex: "#ef4444", text: "text-red-300" },
  2: { name: "Emergent", hex: "#f97316", text: "text-orange-300" },
  3: { name: "Urgent", hex: "#eab308", text: "text-yellow-300" },
  4: { name: "Less urgent", hex: "#22c55e", text: "text-green-300" },
  5: { name: "Non-urgent", hex: "#38bdf8", text: "text-sky-300" },
};

export function esiColor(level: number): string {
  return ESI_COLORS[level]?.hex ?? "#94a3b8";
}
