import type { ReactNode } from "react";

export default function StageCard({
  index, title, subtitle, children, aside,
}: {
  index: number;
  title: string;
  subtitle?: string;
  children: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <section className="card animate-fade-in">
      <header className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-3">
        <div className="flex items-baseline gap-3">
          <span className="text-xs font-semibold tabular-nums text-slate-400">
            {String(index).padStart(2, "0")}
          </span>
          <div>
            <h2 className="text-sm font-semibold text-slate-800">{title}</h2>
            {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
          </div>
        </div>
        {aside}
      </header>
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}
