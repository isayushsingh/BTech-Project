"use client";

import { useState } from "react";

const CONTENT_SCORE = 0.85;
const COLLAB_SCORE = 0.35;

export default function HybridBlendSlider() {
  const [w, setW] = useState(0.5);
  const blended = w * COLLAB_SCORE + (1 - w) * CONTENT_SCORE;

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex w-full max-w-sm flex-col gap-3">
        <Row label="Content score" value={CONTENT_SCORE} color="bg-sky-500" />
        <Row label="Collaborative score" value={COLLAB_SCORE} color="bg-violet-500" />
        <Row label="Blended" value={blended} color="bg-emerald-500" bold />
      </div>
      <div className="flex w-full max-w-sm items-center gap-3 text-xs text-neutral-500">
        <span>more content</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={w}
          onChange={(e) => setW(parseFloat(e.target.value))}
          className="flex-1 accent-emerald-500"
        />
        <span>more collaborative</span>
      </div>
      <p className="text-xs text-neutral-500">
        blended = w · collaborative + (1 − w) · content, w = {w.toFixed(2)}
      </p>
    </div>
  );
}

function Row({
  label,
  value,
  color,
  bold,
}: {
  label: string;
  value: number;
  color: string;
  bold?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`w-32 shrink-0 ${bold ? "font-medium" : "text-neutral-500"}`}>{label}</span>
      <div className="h-2 flex-1 rounded-full bg-neutral-200 dark:bg-neutral-800">
        <div
          className={`h-2 rounded-full ${color} transition-[width]`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
      <span className="w-10 shrink-0 text-right text-neutral-500">{value.toFixed(2)}</span>
    </div>
  );
}
