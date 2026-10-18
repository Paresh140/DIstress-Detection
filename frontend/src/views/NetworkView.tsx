import { useEffect, useState } from "react";
import { getNetwork } from "../api";
import { NeuronStage, layerScale } from "../components/NeuronStage";
import type { Analysis, NetworkSpec } from "../types";

const DETAILS: Record<string, string> = {
  input: "Each clip becomes a 64 by 128 log-mel image. These units stand in for that grid. The real tensor is much wider than the drawing.",
  conv1: "3×3 convolution, 16 filters, batch norm, ReLU, then a 2×2 pool. Brightness is the mean absolute activation on a fixed probe batch.",
  conv2: "3×3 convolution, 32 filters, batch norm, ReLU, pool. Warmer edges into this column mean a larger gradient on this epoch.",
  conv3: "3×3 convolution, 64 filters, batch norm, ReLU, pool. This is the last spatial map before the embedding.",
  embed: "Adaptive pool to 4×4, then 1024 → 128 → 64, with dropout 0.3 while training. The head reads only this vector.",
  head: "One linear logit. Distress if the sigmoid is at least 0.50. The pale dots travel forward. The clay dots travel backward with the gradient.",
};

export function NetworkView({ analysis }: { analysis: Analysis | null }) {
  const [spec, setSpec] = useState<NetworkSpec | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState("conv2");
  const [epoch, setEpoch] = useState(1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    getNetwork()
      .then((next) => {
        setSpec(next);
        setEpoch(1);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!playing || !spec?.history.length) return;
    const timer = window.setInterval(() => {
      setEpoch((current) => (current >= spec.history.length ? 1 : current + 1));
    }, 850);
    return () => window.clearInterval(timer);
  }, [playing, spec]);

  if (error) return <p className="status error">{error}</p>;
  if (!spec) return <p className="status">Loading the network…</p>;

  const snap = spec.history[epoch - 1];
  const last = spec.history.length;
  const activation = layerScale(spec.history, epoch, selected, "activation");
  const grad = layerScale(spec.history, epoch, selected, "grad_norm");

  function play() {
    setPlaying((value) => {
      if (!value && epoch >= last) setEpoch(1);
      return !value;
    });
  }

  return (
    <section className="network-page">
      <header className="page-head">
        <p className="eyebrow">Network</p>
        <h1>Watch the audio branch train.</h1>
        <p className="lede">
          Each column is a layer. Circle brightness is that layer’s mean activation. Edge warmth is the gradient that
          updated it. {spec.audio_params.toLocaleString()} audio parameters actually moved. The circles are a drawing of
          the layers, not one circle per weight.
        </p>
      </header>

      <div className="theater">
        <NeuronStage
          history={spec.history}
          epoch={epoch}
          playing={playing}
          selected={selected}
          onSelect={setSelected}
          bestEpoch={spec.best_epoch}
          liveProbability={analysis?.probability}
        />
        <aside className="train-readout">
          <p className="eyebrow">
            Epoch {String(epoch).padStart(2, "0")}
            {snap && spec.best_epoch === snap.epoch ? " · best checkpoint" : ""}
          </p>
          <p className="metric-kicker">Held-out accuracy</p>
          <p className="epoch-num">{snap ? snap.test_accuracy.toFixed(2) : "—"}</p>
          <dl className="readout-metrics">
            <div>
              <dt>Precision</dt>
              <dd>{snap ? snap.test_precision.toFixed(2) : "—"}</dd>
            </div>
            <div>
              <dt>Recall</dt>
              <dd>{snap ? snap.test_recall.toFixed(2) : "—"}</dd>
            </div>
            <div>
              <dt>F1</dt>
              <dd>{snap ? snap.test_f1.toFixed(2) : "—"}</dd>
            </div>
            <div>
              <dt>Loss</dt>
              <dd>{snap ? snap.test_loss.toFixed(3) : "—"}</dd>
            </div>
          </dl>
          {spec.best ? (
            <p className="fine">
              Best accuracy {spec.best.accuracy.toFixed(2)} at epoch {spec.best.epoch}, with F1 {spec.best.f1.toFixed(2)}.
            </p>
          ) : null}
          <div>
            <p className="eyebrow">{selected}</p>
            <h2>{COLUMNS_NAME[selected]}</h2>
            <p>{DETAILS[selected]}</p>
          </div>
          <dl className="readout-metrics">
            <div>
              <dt>Activation</dt>
              <dd>{activation === null ? "—" : activation.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Gradient</dt>
              <dd>{grad === null ? "—" : grad.toFixed(2)}</dd>
            </div>
          </dl>
          {analysis ? (
            <p className="fine">
              Last clip {analysis.label} at {analysis.probability.toFixed(2)}, drawn on the head.
            </p>
          ) : null}
        </aside>
      </div>

      <div className="transport">
        <button type="button" onClick={() => setEpoch((value) => Math.max(1, value - 1))} disabled={epoch <= 1}>
          Previous
        </button>
        <button type="button" className="play" onClick={play} disabled={!spec.history.length}>
          {playing ? "Pause" : "Play training"}
        </button>
        <button type="button" onClick={() => setEpoch((value) => Math.min(last, value + 1))} disabled={epoch >= last}>
          Next
        </button>
        <input
          type="range"
          min={1}
          max={Math.max(last, 1)}
          value={epoch}
          onChange={(event) => {
            setPlaying(false);
            setEpoch(Number(event.target.value));
          }}
          aria-label="Training epoch"
        />
      </div>
      <p className="fine train-note">
        Pale dots are the forward pass. Clay dots are the backward pass. Replay walks the 16 real epochs. Vision and
        fusion were not updated in this run.
      </p>
    </section>
  );
}

const COLUMNS_NAME: Record<string, string> = {
  input: "Log-mel input",
  conv1: "Conv block 16",
  conv2: "Conv block 32",
  conv3: "Conv block 64",
  embed: "Embedding",
  head: "Distress head",
};
