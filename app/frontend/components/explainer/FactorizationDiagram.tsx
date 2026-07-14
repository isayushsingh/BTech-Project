"use client";

function Grid({
  rows,
  cols,
  label,
  cellClass = "bg-white/[0.08]",
}: {
  rows: number;
  cols: number;
  label: string;
  cellClass?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className="grid gap-0.5"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {Array.from({ length: rows * cols }).map((_, i) => (
          <div key={i} className={`h-4 w-4 rounded-sm ${cellClass}`} />
        ))}
      </div>
      <p className="text-xs text-muted">{label}</p>
    </div>
  );
}

export default function FactorizationDiagram() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4">
      <Grid rows={6} cols={5} label="R — users × movies (sparse)" />
      <span className="text-xl text-muted">≈</span>
      <Grid rows={6} cols={2} label="P — user factors" cellClass="bg-violet-500/40" />
      <span className="text-xl text-muted">·</span>
      <Grid rows={2} cols={5} label="Qᵀ — item factors" cellClass="bg-sky-500/40" />
    </div>
  );
}
