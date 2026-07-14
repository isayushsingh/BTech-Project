"use client";

const STEPS = [
  { title: "You rate 5–10 movies", detail: "no account, just this session" },
  { title: "Solve for your latent vector", detail: "pᵤ = (QᵀQ + λI)⁻¹ Qᵀ(r − μ − bᵢ)" },
  { title: "Score every unrated movie", detail: "μ + bᵢ + pᵤ · qᵢ" },
  { title: "Ranked recommendations", detail: "no retraining needed" },
];

export default function FoldInFlow() {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch sm:gap-2">
      {STEPS.map((step, i) => (
        <div key={step.title} className="flex flex-1 items-center gap-2">
          <div className="flex-1 rounded-lg border border-surface-border p-3 text-center">
            <p className="text-xs font-medium">{step.title}</p>
            <p className="mt-1 font-mono text-[10px] text-muted">{step.detail}</p>
          </div>
          {i < STEPS.length - 1 && (
            <span className="hidden shrink-0 text-white/15 sm:block">→</span>
          )}
        </div>
      ))}
    </div>
  );
}
