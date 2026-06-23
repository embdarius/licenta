import type { ReactNode } from "react";

export default function StageCard({
  index, title, subtitle, accent = "#38bdf8", children,
}: {
  index: number;
  title: string;
  subtitle?: string;
  accent?: string;
  children: ReactNode;
}) {
  return (
    <section className="card animate-fade-up p-5 sm:p-6">
      <header className="mb-4 flex items-center gap-3">
        <span
          className="flex h-8 w-8 items-center justify-center rounded-lg text-sm font-bold text-white"
          style={{ background: accent }}
        >
          {index}
        </span>
        <div>
          <h2 className="text-lg font-semibold leading-tight text-white">{title}</h2>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
      </header>
      {children}
    </section>
  );
}
