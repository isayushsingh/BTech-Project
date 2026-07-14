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
              <th key={m} className="px-2 pb-2 font-normal text-neutral-500">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {USERS.map((u, r) => (
            <tr key={u}>
              <td className="pr-2 text-right font-medium text-neutral-500">{u}</td>
              {DATA[r].map((cell, c) => (
                <td
                  key={c}
                  className={`h-10 w-14 rounded-md text-center align-middle ${
                    cell === "?"
                      ? "border-2 border-dashed border-amber-400 text-amber-500"
                      : cell === null
                        ? "bg-neutral-100 text-neutral-300 dark:bg-neutral-900 dark:text-neutral-700"
                        : "bg-neutral-100 font-medium dark:bg-neutral-900"
                  }`}
                >
                  {cell ?? "·"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-center text-xs text-neutral-500">
        Most cells are empty — nobody rates everything. SVD factorizes this sparse
        matrix to estimate the missing (&quot;?&quot;) ratings.
      </p>
    </div>
  );
}
