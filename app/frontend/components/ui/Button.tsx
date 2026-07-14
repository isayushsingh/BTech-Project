import Link from "next/link";

const VARIANTS = {
  solid: "bg-foreground text-background hover:opacity-90",
  outline: "border border-surface-border text-foreground hover:bg-white/[0.04]",
  accent: "bg-accent text-accent-foreground hover:opacity-90",
} as const;

const BASE =
  "inline-flex items-center justify-center gap-2 rounded-full px-5 py-2.5 font-mono text-sm font-medium transition-colors disabled:opacity-40";

type Variant = keyof typeof VARIANTS;

type CommonProps = {
  children: React.ReactNode;
  variant?: Variant;
  className?: string;
};

type LinkButtonProps = CommonProps & {
  href: string;
  onClick?: never;
  type?: never;
  disabled?: never;
};

type ClickButtonProps = CommonProps & {
  href?: undefined;
  onClick?: () => void;
  type?: "button" | "submit";
  disabled?: boolean;
};

export default function Button(props: LinkButtonProps | ClickButtonProps) {
  const { children, variant = "solid", className = "" } = props;
  const classes = `${BASE} ${VARIANTS[variant]} ${className}`;

  if (props.href) {
    return (
      <Link href={props.href} className={classes}>
        {children}
      </Link>
    );
  }

  return (
    <button
      type={props.type ?? "button"}
      onClick={props.onClick}
      disabled={props.disabled}
      className={classes}
    >
      {children}
    </button>
  );
}
