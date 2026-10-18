type Series = { name: string; color: string; values: number[] };

export function LineChart({ series }: { series: Series[] }) {
  const width = 640;
  const height = 220;
  const pad = { l: 36, r: 12, t: 16, b: 28 };
  const values = series.flatMap((item) => item.values);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  function x(index: number, length: number) {
    return pad.l + (index / Math.max(length - 1, 1)) * innerW;
  }
  function y(value: number) {
    return pad.t + (1 - (value - min) / span) * innerH;
  }

  const ticks = [max, (max + min) / 2, min];

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img">
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="grid" x1={pad.l} x2={width - pad.r} y1={y(tick)} y2={y(tick)} />
          <text className="tick" x={4} y={y(tick) + 4}>
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      {series.map((item) => {
        const d = item.values
          .map((value, index) => `${index === 0 ? "M" : "L"}${x(index, item.values.length).toFixed(1)},${y(value).toFixed(1)}`)
          .join(" ");
        return <path key={item.name} d={d} fill="none" stroke={item.color} strokeWidth="2.4" />;
      })}
      <text className="tick" x={pad.l} y={height - 8}>
        epoch 1
      </text>
      <text className="tick" x={width - pad.r} y={height - 8} textAnchor="end">
        epoch {series[0]?.values.length ?? 0}
      </text>
    </svg>
  );
}

export function Sparkline({
  values,
  activeIndex,
}: {
  values: number[];
  activeIndex: number;
}) {
  const width = 96;
  const height = 30;
  const max = Math.max(...values, 1e-6);
  const coords = values.map((value, index) => {
    const x = (index / Math.max(values.length - 1, 1)) * (width - 4) + 2;
    const y = height - 3 - (value / max) * (height - 8);
    return { x, y };
  });
  const d = coords.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
  const active = coords[Math.min(activeIndex, coords.length - 1)];
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <path d={d} />
      {active ? <circle cx={active.x} cy={active.y} r="2.4" /> : null}
    </svg>
  );
}
