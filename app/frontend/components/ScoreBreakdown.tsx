export default function ScoreBreakdown({
  contentScore,
  collabScore,
  blendedScore,
}: {
  contentScore: number;
  collabScore: number;
  blendedScore: number;
}) {
  const rows = [
    { label: "Content", value: contentScore, color: "bg-sky-400" },
    { label: "Collaborative", value: collabScore, color: "bg-violet-400" },
    { label: "Blended", value: blendedScore, color: "bg-accent" },
  ];

  return (
    <div className="flex flex-col gap-1">
      {rows.map((row) => (
        <div key={row.label} className="flex items-center gap-2 font-mono text-[11px]">
          <span className="w-20 shrink-0 text-muted">{row.label}</span>
          <div className="h-1.5 flex-1 rounded-full bg-white/[0.07]">
            <div
              className={`h-1.5 rounded-full ${row.color}`}
              style={{ width: `${Math.max(row.value, 0.02) * 100}%` }}
            />
          </div>
          <span className="w-8 shrink-0 text-right text-muted">
            {row.value.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}
