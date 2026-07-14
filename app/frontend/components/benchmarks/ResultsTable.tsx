import type { BaselineResult } from "@/lib/types";

const ORDER = ["popularity", "item_knn", "svd_only", "content_only", "hybrid"];

function cell(value: number | null, digits = 4) {
  return value === null ? <span className="text-neutral-300 dark:text-neutral-700">—</span> : value.toFixed(digits);
}

export default function ResultsTable({
  baselines,
}: {
  baselines: Record<string, BaselineResult>;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-black/10 text-left text-xs text-neutral-500 dark:border-white/10">
            <th className="py-2 pr-4 font-normal">Approach</th>
            <th className="py-2 px-3 font-normal">Precision@10</th>
            <th className="py-2 px-3 font-normal">Recall@10</th>
            <th className="py-2 px-3 font-normal">NDCG@10</th>
            <th className="py-2 px-3 font-normal">RMSE</th>
            <th className="py-2 px-3 font-normal">MAE</th>
          </tr>
        </thead>
        <tbody>
          {ORDER.filter((key) => baselines[key]).map((key) => {
            const row = baselines[key];
            return (
              <tr key={key} className="border-b border-black/5 dark:border-white/5">
                <td className="py-2 pr-4 font-medium">{row.label}</td>
                <td className="py-2 px-3">{cell(row.precision_at_k)}</td>
                <td className="py-2 px-3">{cell(row.recall_at_k)}</td>
                <td className="py-2 px-3">{cell(row.ndcg_at_k)}</td>
                <td className="py-2 px-3">{cell(row.rmse)}</td>
                <td className="py-2 px-3">{cell(row.mae)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
