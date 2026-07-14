"use client";

function Grid({
  rows,
  cols,
  label,
  cellClass = "bg-neutral-200 dark:bg-neutral-800",
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
      <p className="text-xs text-neutral-500">{label}</p>
    </div>
  );
}

export default function FactorizationDiagram() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4">
      <Grid rows={6} cols={5} label="R — users × movies (sparse)" />
      <span className="text-xl text-neutral-400">≈</span>
      <Grid rows={6} cols={2} label="P — user factors" cellClass="bg-violet-200 dark:bg-violet-900" />
      <span className="text-xl text-neutral-400">·</span>
      <Grid rows={2} cols={5} label="Qᵀ — item factors" cellClass="bg-sky-200 dark:bg-sky-900" />
    </div>
  );
}
