"use client";

const MOVIES = ["Toy Story", "Heat", "Casino", "Se7en", "Jumanji"];
const USERS = ["you", "u2", "u3", "u4"];
const DATA: (number | "?" | null)[][] = [
  [5, null, 3, "?", 4],
  [null, 4, 4, 2, null],
  [3, 3, null, 1, 5],
  [4, null, 2, null, 3],
];

export default function UtilityMatrix() {
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-1 text-xs">
        <thead>
          <tr>
            <th />
            {MOVIES.map((m) => (
              <th key={m} className="px-2 pb-2 font-normal text-muted">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {USERS.map((u, r) => (
            <tr key={u}>
              <td className="pr-2 text-right font-medium text-muted">{u}</td>
              {DATA[r].map((cell, c) => (
                <td
                  key={c}
                  className={`h-10 w-14 rounded-md text-center align-middle ${
                    cell === "?"
                      ? "border-2 border-dashed border-accent text-accent"
                      : cell === null
                        ? "bg-white/[0.04] text-white/20"
                        : "bg-white/[0.06] font-medium text-foreground"
                  }`}
                >
                  {cell ?? "·"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-center text-xs text-muted">
        Most cells are empty — nobody rates everything. SVD factorizes this sparse
        matrix to estimate the missing (&quot;?&quot;) ratings.
      </p>
    </div>
  );
}
