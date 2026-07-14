"use client";

import { useState } from "react";
import Image from "next/image";

export default function PosterImage({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="flex h-full items-center justify-center p-2 text-center text-xs text-neutral-500">
        {alt}
      </div>
    );
  }

  return (
    <Image
      src={src}
      alt={alt}
      fill
      sizes="200px"
      className="object-cover"
      onError={() => setFailed(true)}
    />
  );
}
