import type { EpochSnap } from "../types";

const COLUMNS = [
  { id: "input", name: "Log-mel", shape: "64 × 128", units: 6 },
  { id: "conv1", name: "Conv 16", shape: "32 × 64", units: 7 },
  { id: "conv2", name: "Conv 32", shape: "16 × 32", units: 7 },
  { id: "conv3", name: "Conv 64", shape: "8 × 16", units: 6 },
  { id: "embed", name: "Embed", shape: "64-d", units: 5 },
  { id: "head", name: "Head", shape: "logit", units: 1 },
] as const;

type Node = { id: string; x: number; y: number; r: number };

const WIDTH = 1000;
const HEIGHT = 520;

function mix(t: number): string {
  const clamp = Math.min(1, Math.max(0, t));
  const from = [118, 128, 124];
  const to = [247, 250, 246];
  const rgb = from.map((channel, index) => Math.round(channel + (to[index] - channel) * clamp));
  return `rgb(${rgb.join(",")})`;
}

function peak(history: EpochSnap[], id: string, field: "activation" | "grad_norm"): number {
  return Math.max(...history.map((row) => row[field]?.[id] ?? 0), 1e-6);
}

function measured(snap: EpochSnap | undefined, field: "activation" | "grad_norm", id: string): number | null {
  const value = snap?.[field]?.[id];
  return value === undefined ? null : value;
}

function lossOf(row: EpochSnap): number {
  return typeof row.test_loss === "number" ? row.test_loss : (row.train_loss ?? 0);
}

function place(): Node[][] {
  return COLUMNS.map((column, index) => {
    const x = 86 + (index * (WIDTH - 172)) / (COLUMNS.length - 1);
    const gap = column.units === 1 ? 0 : 36;
    const span = gap * (column.units - 1);
    const top = 196 - span / 2;
    return Array.from({ length: column.units }, (_, unit) => ({
      id: column.id,
      x,
      y: column.units === 1 ? 196 : top + unit * gap,
      r: column.units === 1 ? 20 : 12,
    }));
  });
}

const NODES = place();

function lossPoint(history: EpochSnap[], index: number): { x: number; y: number } {
  const losses = history.map(lossOf);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = max - min || 1;
  return {
    x: 86 + (index / Math.max(history.length - 1, 1)) * (WIDTH - 172),
    y: 500 - ((lossOf(history[index]) - min) / span) * 36,
  };
}

function lossTrace(history: EpochSnap[], through: number): string {
  if (history.length < 2) return "";
  const end = Math.min(history.length, Math.max(through, 1));
  return history
    .slice(0, end)
    .map((_, index) => {
      const point = lossPoint(history, index);
      return `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`;
    })
    .join(" ");
}

export function NeuronStage({
  history,
  epoch,
  playing,
  selected,
  onSelect,
  bestEpoch,
  liveProbability,
}: {
  history: EpochSnap[];
  epoch: number;
  playing: boolean;
  selected: string;
  onSelect: (id: string) => void;
  bestEpoch: number | null;
  liveProbability?: number;
}) {
  const snap = history[epoch - 1];
  const traceIsTest = history.some((row) => typeof row.test_loss === "number");
  const fullTrace = lossTrace(history, history.length);
  const trace = lossTrace(history, epoch);
  const marker = history.length > 1 ? lossPoint(history, Math.max(epoch - 1, 0)) : null;
  const motion = playing && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const spine = NODES.map((column) => column[Math.floor(column.length / 2)]);

  return (
    <div className={`graph-panel ${playing ? "is-playing" : ""}`}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="DistressNet audio branch during training">
        <rect width={WIDTH} height={HEIGHT} fill="#141311" />
        <text x="28" y="36" className="graph-kicker">
          Forward
        </text>
        <text x={WIDTH - 28} y="36" className="graph-kicker" textAnchor="end">
          Gradient
        </text>

        {NODES.slice(0, -1).map((column, index) =>
          column.flatMap((from) =>
            NODES[index + 1].map((to) => {
              const dest = to.id;
              const rawGrad = measured(snap, "grad_norm", dest);
              const grad = rawGrad === null ? 0.4 : rawGrad / peak(history, dest, "grad_norm");
              const warm = grad > 0.55;
              return (
                <line
                  key={`${from.x}-${from.y}-${to.x}-${to.y}`}
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={warm ? "#c46a58" : "#8eaea3"}
                  strokeWidth={0.7 + grad * 1.5}
                  opacity={0.18 + grad * 0.55}
                />
              );
            }),
          ),
        )}

        {motion
          ? spine.slice(0, -1).map((from, index) => {
              const to = spine[index + 1];
              const rawGrad = measured(snap, "grad_norm", to.id);
              const grad = rawGrad === null ? 0.4 : rawGrad / peak(history, to.id, "grad_norm");
              return (
                <g key={`flow-${index}`}>
                  <circle r="4" fill="#f4fff8">
                    <animateMotion
                      dur={`${1.05 + index * 0.08}s`}
                      repeatCount="indefinite"
                      path={`M${from.x},${from.y} L${to.x},${to.y}`}
                    />
                  </circle>
                  <circle r="3.2" fill="#e7a090" opacity={0.35 + grad * 0.65}>
                    <animateMotion
                      dur={`${1.5 + index * 0.06}s`}
                      repeatCount="indefinite"
                      path={`M${to.x},${to.y} L${from.x},${from.y}`}
                    />
                  </circle>
                </g>
              );
            })
          : null}

        {NODES.map((column, index) => {
          const spec = COLUMNS[index];
          const rawActivation = measured(snap, "activation", spec.id);
          const activation =
            spec.id === "input"
              ? 0.62
              : rawActivation === null
                ? (snap?.test_f1 ?? 0.35)
                : rawActivation / peak(history, spec.id, "activation");
          const isSelected = selected === spec.id;
          return (
            <g key={spec.id} onClick={() => onSelect(spec.id)} className="neuron-col">
              {column.map((node, unit) => (
                <circle
                  key={unit}
                  cx={node.x}
                  cy={node.y}
                  r={node.r}
                  fill={mix(0.28 + activation * 0.72 * (0.82 + (unit % 3) * 0.06))}
                  stroke={isSelected ? "#f6f1e8" : "#1c1b19"}
                  strokeWidth={isSelected ? 2.5 : 1}
                />
              ))}
              <text x={column[0].x} y="392" textAnchor="middle" className="graph-name">
                {spec.name}
              </text>
              <text x={column[0].x} y="410" textAnchor="middle" className="graph-shape">
                {spec.shape}
              </text>
            </g>
          );
        })}

        {liveProbability !== undefined ? (
          <text x={NODES[5][0].x} y={NODES[5][0].y + 42} textAnchor="middle" className="graph-prob">
            {liveProbability.toFixed(2)}
          </text>
        ) : null}

        <text x="86" y="436" className="graph-kicker">
          {traceIsTest ? "Test loss" : "Train loss"}
          {snap ? ` ${lossOf(snap).toFixed(3)}` : ""}
          {bestEpoch === epoch ? "  ·  best" : ""}
        </text>
        <path d={fullTrace} fill="none" stroke="#c46a58" strokeWidth="2" opacity="0.28" />
        <path d={trace} fill="none" stroke="#e7b2a6" strokeWidth="2.4" />
        {marker ? <circle cx={marker.x} cy={marker.y} r="4.5" fill="#f6f1e8" /> : null}
      </svg>
    </div>
  );
}

export function layerScale(history: EpochSnap[], epoch: number, id: string, field: "activation" | "grad_norm"): number | null {
  const snap = history[epoch - 1];
  const value = snap?.[field]?.[id];
  if (value === undefined) return null;
  return value / peak(history, id, field);
}
