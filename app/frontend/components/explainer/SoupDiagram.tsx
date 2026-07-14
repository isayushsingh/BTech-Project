"use client";

import { motion } from "framer-motion";

const TAGS: { label: string; group: string; color: string }[] = [
  { label: "jealousy", group: "keywords", color: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300" },
  { label: "toy", group: "keywords", color: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300" },
  { label: "boy", group: "keywords", color: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300" },
  { label: "tomhanks", group: "cast", color: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300" },
  { label: "timallen", group: "cast", color: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300" },
  { label: "donrickles", group: "cast", color: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300" },
  { label: "johnlasseter", group: "director", color: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300" },
  { label: "Animation", group: "genres", color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" },
  { label: "Comedy", group: "genres", color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" },
  { label: "Family", group: "genres", color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" },
];

export default function SoupDiagram() {
  return (
    <div className="flex flex-col items-center gap-4">
      <p className="text-xs text-neutral-500">Toy Story's metadata, tagged by source</p>
      <div className="flex flex-wrap justify-center gap-2">
        {TAGS.map((tag, i) => (
          <motion.span
            key={tag.label}
            initial={{ opacity: 0, y: -10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.05, duration: 0.4 }}
            className={`rounded-full px-2.5 py-1 text-xs font-medium ${tag.color}`}
          >
            {tag.label}
          </motion.span>
        ))}
      </div>
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.6, duration: 0.5 }}
        className="text-2xl text-neutral-400"
      >
        ↓
      </motion.div>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.8, duration: 0.5 }}
        className="max-w-lg rounded-lg border border-black/10 bg-neutral-50 p-4 font-mono text-xs leading-relaxed text-neutral-600 dark:border-white/10 dark:bg-neutral-900 dark:text-neutral-400"
      >
        &quot;jealousy toy boy tomhanks timallen donrickles johnlasseter Animation
        Comedy Family&quot;
      </motion.div>
      <p className="text-xs text-neutral-500">
        one &quot;soup&quot; string per movie, ready to vectorize
      </p>
    </div>
  );
}
