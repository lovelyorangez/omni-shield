const AGENT_ORDER = [
  "RouterAgent",
  "OCRAgent",
  "LayoutAgent",
  "TextPIIAgent",
  "ContextAgent",
  "BBRefinerAgent",
  "CriticAgent",
  "RedactionAgent",
];

const AGENT_COLORS = {
  RouterAgent:    "#6366f1", // indigo
  OCRAgent:       "#0ea5e9", // sky
  LayoutAgent:    "#8b5cf6", // violet
  TextPIIAgent:   "#ec4899", // pink
  ContextAgent:   "#f59e0b", // amber
  BBRefinerAgent: "#10b981", // emerald
  CriticAgent:    "#ef4444", // red
  RedactionAgent: "#64748b", // slate
};

export default function PipelineTimingChart({ timings }) {
  if (!timings || Object.keys(timings).length === 0) return null;

  const entries = AGENT_ORDER
    .filter(name => timings[name] != null)
    .map(name => ({ name, seconds: timings[name] }));

  if (entries.length === 0) return null;

  const maxSeconds = Math.max(...entries.map(e => e.seconds), 0.01);
  const total = entries.reduce((s, e) => s + e.seconds, 0);

  return (
    <div className="mt-4 rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-widest text-zinc-400">
          Pipeline Timing
        </span>
        <span className="text-xs text-zinc-500">
          total {total.toFixed(1)}s
        </span>
      </div>

      <div className="space-y-2">
        {entries.map(({ name, seconds }) => {
          const pct = (seconds / maxSeconds) * 100;
          const color = AGENT_COLORS[name] ?? "#94a3b8";
          return (
            <div key={name}>
              <div className="mb-0.5 flex items-center justify-between">
                <span className="text-xs text-zinc-300">{name}</span>
                <span className="text-xs tabular-nums text-zinc-400">
                  {seconds.toFixed(2)}s
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-700">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${pct}%`, backgroundColor: color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
