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

// ESI level → standard triage palette, in solid tones with adequate contrast on
// a light surface (not neon). `bg`/`border` are light fills for level tags.
export const ESI_COLORS: Record<
  number,
  { name: string; hex: string; bg: string; border: string; text: string }
> = {
  1: { name: "Resuscitation", hex: "#b42318", bg: "#fef3f2", border: "#fecdca", text: "text-red-700" },
  2: { name: "Emergent", hex: "#c2410c", bg: "#fff4ed", border: "#fed7aa", text: "text-orange-700" },
  3: { name: "Urgent", hex: "#b7791f", bg: "#fefbeb", border: "#fde68a", text: "text-yellow-700" },
  4: { name: "Less urgent", hex: "#2f855a", bg: "#f0fdf4", border: "#bbf7d0", text: "text-emerald-700" },
  5: { name: "Non-urgent", hex: "#2b6cb0", bg: "#eff6ff", border: "#bfdbfe", text: "text-blue-700" },
};

export function esiColor(level: number): string {
  return ESI_COLORS[level]?.hex ?? "#64748b";
}
