export default function Badge({
  children,
  variant = "accent",
  className = "",
}: {
  children: React.ReactNode;
  variant?: "accent" | "neutral";
  className?: string;
}) {
  const styles =
    variant === "accent"
      ? "bg-accent text-accent-foreground"
      : "bg-white/[0.06] text-muted border border-surface-border";
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 font-mono text-[10px] font-medium uppercase tracking-wide ${styles} ${className}`}
    >
      {children}
    </span>
  );
}
