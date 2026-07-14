export default function Card({
  children,
  className = "",
  id,
}: {
  children: React.ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <div
      id={id}
      className={`rounded-2xl border border-surface-border bg-surface ${className}`}
    >
      {children}
    </div>
  );
}
