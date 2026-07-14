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
          <span className={star <= value ? "text-accent" : "text-white/15"}>
            ★
          </span>
        </button>
      ))}
    </div>
  );
}
