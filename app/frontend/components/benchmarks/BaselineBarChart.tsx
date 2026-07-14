import type { BaselineResult } from "@/lib/types";

const ORDER = ["popularity", "item_knn", "svd_only", "content_only", "hybrid"];
const COLORS: Record<string, string> = {
  popularity: "bg-neutral-400 dark:bg-neutral-600",
  item_knn: "bg-amber-500",
  svd_only: "bg-sky-500",
  content_only: "bg-violet-500",
  hybrid: "bg-emerald-500",
};

export default function BaselineBarChart({
  baselines,
}: {
  baselines: Record<string, BaselineResult>;
}) {
  const rows = ORDER.filter((key) => baselines[key]).map((key) => ({
    key,
    ...baselines[key],
  }));
  const max = Math.max(...rows.map((r) => r.ndcg_at_k ?? 0));

  return (
    <div className="flex flex-col gap-3">
      {rows.map((row) => (
        <div key={row.key} className="flex items-center gap-3 text-xs">
          <span className="w-40 shrink-0 text-neutral-600 dark:text-neutral-400">
            {row.label}
          </span>
          <div className="h-3 flex-1 rounded-full bg-neutral-100 dark:bg-neutral-900">
            <div
              className={`h-3 rounded-full ${COLORS[row.key]}`}
              style={{ width: `${((row.ndcg_at_k ?? 0) / max) * 100}%` }}
            />
          </div>
          <span className="w-14 shrink-0 text-right font-medium">
            {row.ndcg_at_k?.toFixed(4) ?? "—"}
          </span>
        </div>
      ))}
      <p className="mt-1 text-[11px] text-neutral-500">NDCG@10 — higher is better</p>
    </div>
  );
}
