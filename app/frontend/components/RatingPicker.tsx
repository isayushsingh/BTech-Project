"use client";

export default function RatingPicker({
  value,
  onChange,
}: {
  value: number;
  onChange: (rating: number) => void;
}) {
  const stars = [1, 2, 3, 4, 5];
  return (
    <div className="flex gap-0.5">
      {stars.map((star) => (
        <button
          key={star}
          type="button"
          aria-label={`Rate ${star} stars`}
          onClick={() => onChange(star)}
          className="text-lg leading-none"
        >
          <span className={star <= value ? "text-amber-500" : "text-neutral-300 dark:text-neutral-700"}>
            ★
          </span>
        </button>
      ))}
    </div>
  );
}
