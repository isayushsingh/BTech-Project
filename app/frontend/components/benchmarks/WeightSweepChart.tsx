import type { WeightSweepPoint } from "@/lib/types";

const CHART_HEIGHT_PX = 128;

export default function WeightSweepChart({
  sweep,
  productionWeight,
  bestWeight,
}: {
  sweep: WeightSweepPoint[];
  productionWeight: number;
  bestWeight: number;
}) {
  const max = Math.max(...sweep.map((p) => p.ndcg_at_k ?? 0));

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-end gap-1.5" style={{ height: CHART_HEIGHT_PX }}>
        {sweep.map((p) => {
          const heightPx = max > 0 ? Math.max(((p.ndcg_at_k ?? 0) / max) * CHART_HEIGHT_PX, 2) : 2;
          const isBest = p.w === bestWeight;
          return (
            <div
              key={p.w}
              className={`flex-1 rounded-t-sm transition-all ${
                isBest ? "bg-accent" : "bg-white/15"
              }`}
              style={{ height: heightPx }}
              title={`w=${p.w}: NDCG@10=${p.ndcg_at_k}`}
            />
          );
        })}
      </div>
      <div className="flex gap-1.5">
        {sweep.map((p) => (
          <span key={p.w} className="flex-1 text-center text-[10px] text-muted">
            {p.w.toFixed(1)}
          </span>
        ))}
      </div>
      <p className="text-[11px] text-muted">
        blend weight (w) → NDCG@10. Production uses w={productionWeight.toFixed(1)}
        {productionWeight === bestWeight ? " (the empirical best)" : ""}.
      </p>
    </div>
  );
}
