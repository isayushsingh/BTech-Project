"use client";

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-32 shrink-0 text-muted">{label}</span>
      <div className="h-2 flex-1 rounded-full bg-white/[0.07]">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
      </div>
      <span className="w-10 shrink-0 text-right text-muted">{value.toFixed(2)}</span>
    </div>
  );
}

export default function BatmanRobinProblem() {
  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <div className="rounded-lg border border-surface-border p-4">
        <p className="mb-3 text-sm font-medium">Batman &amp; Robin (1997)</p>
        <div className="flex flex-col gap-2">
          <Bar label="Cast/crew similarity" value={0.78} color="bg-sky-500" />
          <Bar label="Critical rating" value={0.28} color="bg-red-500" />
        </div>
        <p className="mt-3 text-xs text-muted">
          Shares Batman, Gotham, and much of the crew — but it&apos;s a 3.7/10 movie.
          Pure text similarity would rank it highly anyway.
        </p>
      </div>
      <div className="rounded-lg border border-surface-border p-4">
        <p className="mb-3 text-sm font-medium">Batman Begins (2005)</p>
        <div className="flex flex-col gap-2">
          <Bar label="Cast/crew similarity" value={0.77} color="bg-sky-500" />
          <Bar label="Critical rating" value={0.75} color="bg-emerald-500" />
        </div>
        <p className="mt-3 text-xs text-muted">
          Similar text similarity — but a well-reviewed movie, so the quality gate lets
          it rank above Batman &amp; Robin in the final blend.
        </p>
      </div>
    </div>
  );
}
