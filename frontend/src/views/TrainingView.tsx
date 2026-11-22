import { useEffect, useState } from "react";
import {
  clipUrl,
  getKaggleUrl,
  getStudy,
  getTrainLive,
  getTraining,
  getVoiceReport,
  loadKaggleResults,
  startTraining,
  type StudyReport,
  type VoiceReport,
} from "../api";
import type { TrainEpoch } from "../types";
import { LineChart } from "../components/Charts";
import { NeuronStage, layerScale } from "../components/NeuronStage";
import type { TrainingReport } from "../types";

function actorSpan(actors: number[]): string {
  const sorted = [...actors].sort((a, b) => a - b);
  if (!sorted.length) return "none";
  const pad = (actor: number) => String(actor).padStart(2, "0");
  const contiguous = sorted.every((actor, index) => index === 0 || actor === sorted[index - 1] + 1);
  if (contiguous && sorted.length > 1) return `${pad(sorted[0])}–${pad(sorted[sorted.length - 1])}`;
  return sorted.map(pad).join(", ");
}

export function TrainingView() {
  const [report, setReport] = useState<TrainingReport | null>(null);
  const [error, setError] = useState("");
  const [epoch, setEpoch] = useState(1);
  const [layer, setLayer] = useState("conv2");
  const [following, setFollowing] = useState(false);
  const [starting, setStarting] = useState(false);
  const [serverDone, setServerDone] = useState(false);
  const [liveHistory, setLiveHistory] = useState<TrainEpoch[] | null>(null);
  const [liveBest, setLiveBest] = useState<TrainingReport["best"] | null>(null);
  const [liveNote, setLiveNote] = useState("");
  const [plannedEpochs, setPlannedEpochs] = useState(16);
  const [resultsVersion, setResultsVersion] = useState(0);

  useEffect(() => {
    let active = true;
    getTraining()
      .then((next) => {
        if (!active) return;
        setReport(next);
        setEpoch(next.best.epoch);
      })
      .catch((err: Error) => {
        if (active) setError(err.message);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!following || serverDone) return;
    const timer = window.setInterval(() => {
      getTrainLive()
        .then((live) => {
          setPlannedEpochs(live.epochs);
          if (live.history.length) {
            setLiveHistory(live.history);
            if (live.best) setLiveBest(live.best);
            setEpoch((current) => (current < 1 ? 1 : current));
          } else {
            setLiveNote("Loading the clips. The first epoch has not finished yet.");
          }
          if (live.phase === "error") {
            setFollowing(false);
            setError(live.error || "Training stopped before it finished.");
          } else if (!live.running && live.phase === "done") {
            setServerDone(true);
          }
        })
        .catch((err: Error) => setError(err.message));
    }, 700);
    return () => window.clearInterval(timer);
  }, [following, serverDone]);

  useEffect(() => {
    if (!following || !liveHistory?.length || epoch < 1) return;
    if (epoch < liveHistory.length) {
      const timer = window.setTimeout(() => setEpoch((current) => current + 1), 1400);
      return () => window.clearTimeout(timer);
    }
    if (serverDone && epoch >= liveHistory.length) {
      const timer = window.setTimeout(() => {
        getTraining()
          .then((next) => {
            setReport(next);
            setLiveHistory(null);
            setLiveBest(null);
            setFollowing(false);
            setServerDone(false);
            setEpoch(next.best.epoch);
            setLiveNote(`This run is saved. Actors ${actorSpan(next.split.test_actors)} scored it.`);
          })
          .catch((err: Error) => setError(err.message));
      }, 1400);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [following, liveHistory, epoch, serverDone]);

  async function trainAgain() {
    setError("");
    setStarting(true);
    setEpoch(0);
    setLiveHistory(null);
    setLiveBest(null);
    setServerDone(false);
    setLiveNote("Loading the clips. The first epoch has not finished yet.");
    try {
      const started = await startTraining();
      setPlannedEpochs(started.epochs);
      setStarting(false);
      setFollowing(true);
    } catch (err) {
      setStarting(false);
      setFollowing(false);
      setLiveNote("");
      setError(err instanceof Error ? err.message : "Training could not start.");
    }
  }

  if (!report) {
    return (
      <section className="training-page">
        <VoiceSection onLoaded={() => setResultsVersion((n) => n + 1)} />
        <p className={error ? "status error" : "status"}>{error || "Loading the training run…"}</p>
      </section>
    );
  }

  const view = report;
  const history = liveHistory && epoch > 0 ? liveHistory.slice(0, epoch) : view.history;
  const best = liveBest ?? view.best;
  const detailed = history.some((row) => typeof row.test_loss === "number");
  const showingLive = following && liveHistory !== null && epoch > 0;
  const stageEpoch = Math.min(Math.max(epoch, 1), Math.max(history.length, 1));
  const selected = history[stageEpoch - 1];
  const misses = view.predictions.filter((item) => !item.correct).slice(0, 6);
  const hits = view.predictions.filter((item) => item.correct).slice(0, 4);
  const shown = [...misses, ...hits].slice(0, 8);

  return (
    <section className="training-page">
      <VoiceSection onLoaded={() => setResultsVersion((n) => n + 1)} />

      <header className="page-head">
        <p className="eyebrow">First prototype: the CNN</p>
        <h1>
          Actors {actorSpan(view.split.train_actors)} taught it. Actors {actorSpan(view.split.test_actors)} scored
          it.
        </h1>
        <p className="lede">{view.narrative}</p>
        <div className="train-bar">
          <button type="button" className="file-btn" onClick={() => void trainAgain()} disabled={starting || following}>
            {following ? "Training" : "Train on this Mac"}
          </button>
          <p className="fine">
            {showingLive
              ? `Epoch ${epoch} of ${plannedEpochs}. Each epoch stays on the network so the forward and backward pass can be seen.`
              : liveNote ||
                "The log-mel CNN came before the WavLM model above. Train on this Mac retrains it on the clips already here."}
          </p>
        </div>
        {error ? <p className="status error">{error}</p> : null}
      </header>

      <NeuronStage
        history={history}
        epoch={stageEpoch}
        playing={showingLive}
        selected={layer}
        onSelect={setLayer}
        bestEpoch={best.epoch}
      />
      <p className="fine train-note">
          <>
            Click a column, or a row in the epoch table. Brightness is activation
            {layerScale(history, stageEpoch, layer, "activation") !== null
              ? ` (${layerScale(history, stageEpoch, layer, "activation")!.toFixed(2)} on ${layer})`
              : ""}
            . Warmer edges are larger gradients on that epoch.
            {showingLive ? " Pale dots move forward and clay dots move backward on this epoch." : ""}
          </>
      </p>

      <div className="stat-grid">
        <Stat label="Test F1" value={best.f1.toFixed(2)} note={`epoch ${best.epoch}, chosen on validation`} />
        <Stat label="Accuracy" value={best.accuracy.toFixed(2)} note="held-out actors" />
        <Stat label="Precision" value={best.precision.toFixed(2)} note="distress predictions" />
        <Stat label="Recall" value={best.recall.toFixed(2)} note="distress clips caught" />
      </div>

      <div className="chart-grid">
        <figure>
          <figcaption>Loss</figcaption>
          <LineChart
            series={
              detailed
                ? [
                    { name: "Train", color: "#1b1914", values: history.map((row) => row.train_loss) },
                    { name: "Test", color: "#a33b2b", values: history.map((row) => row.test_loss ?? 0) },
                  ]
                : [{ name: "Train", color: "#1b1914", values: history.map((row) => row.train_loss) }]
            }
          />
          <p className="legend">
            <i className="swatch ink" /> Train {detailed ? <><i className="swatch clay" /> Test</> : null}
          </p>
        </figure>
        <figure>
          <figcaption>Held-out scores</figcaption>
          <LineChart
            series={
              detailed
                ? [
                    { name: "F1", color: "#1f4d3a", values: history.map((row) => row.test_f1) },
                    { name: "Accuracy", color: "#a9782e", values: history.map((row) => row.test_accuracy ?? 0) },
                  ]
                : [
                    { name: "F1", color: "#1f4d3a", values: history.map((row) => row.test_f1) },
                    {
                      name: "Val balanced accuracy",
                      color: "#a9782e",
                      values: history.map((row) => row.val_balanced_accuracy ?? 0),
                    },
                  ]
            }
          />
          <p className="legend">
            <i className="swatch pine" /> F1 <i className="swatch gold" /> {detailed ? "Accuracy" : "Val balanced accuracy"}
          </p>
        </figure>
      </div>

      <div className="split-layout">
        <article className="panel">
          <h2>Confusion matrix</h2>
          {view.confusion ? (
            <>
              <p className="fine">Best checkpoint, actors {actorSpan(view.split.test_actors)}. Rows are the true class.</p>
              <div className="matrix">
                <span />
                <span>Pred calm</span>
                <span>Pred distress</span>
                <span>True calm</span>
                <b>{view.confusion.tn}</b>
                <b>{view.confusion.fp}</b>
                <span>True distress</span>
                <b>{view.confusion.fn}</b>
                <b>{view.confusion.tp}</b>
              </div>
            </>
          ) : (
            <p className="fine">
              This report has no confusion counts.
            </p>
          )}
        </article>
        <article className="panel">
          <h2>Actor split</h2>
          <div className="actors">
            {Array.from({ length: 24 }, (_, index) => {
              const actor = index + 1;
              const trainSet = new Set(view.split.train_actors);
              const testSet = new Set(view.split.test_actors);
              const valSet = new Set(view.split.val_actors ?? []);
              const kind = trainSet.has(actor) ? "train" : testSet.has(actor) ? "test" : valSet.has(actor) ? "val" : "out";
              return (
                <span key={actor} className={kind}>
                  {String(actor).padStart(2, "0")}
                </span>
              );
            })}
          </div>
          <p className="fine">
            Green trained the weights. Clay scored the run.
            {view.split.val_actors?.length ? " Gold chose the checkpoint." : ""}
            {view.split.unused_actors.length ? " Grey actors are outside this run." : ""}
          </p>
        </article>
      </div>

      <article className="panel">
        <h2>Recipe</h2>
        <dl className="recipe">
          <div>
            <dt>Loss</dt>
            <dd>
              {view.loss}, pos_weight {view.pos_weight.toFixed(2)}
            </dd>
          </div>
          <div>
            <dt>Optimizer</dt>
            <dd>
              {view.optimizer}, lr {view.learning_rate}, weight decay {view.weight_decay}
            </dd>
          </div>
          <div>
            <dt>Data</dt>
            <dd>
              {view.split.train_clips} train clips, {view.split.test_clips} test clips, batch {view.batch_size}, seed{" "}
              {view.seed}
            </dd>
          </div>
          <div>
            <dt>Labels</dt>
            <dd>
              Distress: {view.label.distress.join(", ")}. Calm: {view.label.calm.join(", ")}. Excluded:{" "}
              {view.label.excluded.join(", ")}.
            </dd>
          </div>
          <div>
            <dt>Augmentation</dt>
            <dd>{view.augmentation}</dd>
          </div>
          <div>
            <dt>Device</dt>
            <dd>{view.device}</dd>
          </div>
        </dl>
      </article>

      <article className="panel">
        <h2>Epoch {selected?.epoch}</h2>
        {selected?.grad_norm?.conv1 !== undefined ? (
          <p className="fine">
            Gradient norms this epoch — conv1 {selected.grad_norm.conv1.toFixed(3)}, conv2{" "}
            {selected.grad_norm.conv2.toFixed(3)}, conv3 {selected.grad_norm.conv3.toFixed(3)}, embedding{" "}
            {selected.grad_norm.embed.toFixed(3)}, head {selected.grad_norm.head.toFixed(3)}.
          </p>
        ) : selected ? (
          <p className="fine">
            Train loss {selected.train_loss.toFixed(3)}
            {selected.val_balanced_accuracy !== undefined
              ? `, validation balanced accuracy ${selected.val_balanced_accuracy.toFixed(3)}`
              : ""}
            , test F1 {selected.test_f1.toFixed(3)}.
          </p>
        ) : null}
        <div className="epoch-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Epoch</th>
                <th>Train loss</th>
                {detailed ? <th>Test loss</th> : <th>Val balanced accuracy</th>}
                {detailed ? <th>Accuracy</th> : null}
                <th>F1</th>
              </tr>
            </thead>
            <tbody>
              {history.map((row) => (
                <tr
                  key={row.epoch}
                  className={row.epoch === stageEpoch ? "is-selected" : ""}
                  onClick={() => {
                    if (!following) setEpoch(row.epoch);
                  }}
                >
                  <td>
                    {row.epoch}
                    {row.epoch === best.epoch ? " · best" : ""}
                  </td>
                  <td>{row.train_loss.toFixed(3)}</td>
                  <td>{detailed ? (row.test_loss ?? 0).toFixed(3) : (row.val_balanced_accuracy ?? 0).toFixed(3)}</td>
                  {detailed ? <td>{(row.test_accuracy ?? 0).toFixed(3)}</td> : null}
                  <td>{row.test_f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

      <article className="panel">
        <h2>Held-out clips</h2>
        {view.predictions.length ? (
        <ul className="clips">
          {shown.map((item) => (
            <li key={item.file} className={item.correct ? "hit" : "miss"}>
              <div>
                <strong>
                  Actor {String(item.actor).padStart(2, "0")} · {item.emotion}
                </strong>
                <p>
                  True {item.truth} · predicted {item.predicted} · {item.probability.toFixed(2)}
                </p>
              </div>
              <audio controls preload="metadata" src={clipUrl(item.file)} />
            </li>
          ))}
        </ul>
        ) : (
          <p className="fine">This report has no held-out clip scores.</p>
        )}
      </article>

      <StudySection key={resultsVersion} />

      <p className="citation">{view.citation}</p>
    </section>
  );
}

function VoiceSection({ onLoaded }: { onLoaded: () => void }) {
  const [voice, setVoice] = useState<VoiceReport | null>(null);
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function loadVoice() {
    getVoiceReport()
      .then(setVoice)
      .catch((err: Error) => setVoice({ ready: false, error: err.message }));
  }

  useEffect(loadVoice, []);

  async function trainOnGpu() {
    setError("");
    // Open the tab inside the click so the browser does not block it, then point it at Kaggle.
    const tab = window.open("", "_blank");
    try {
      const { url } = await getKaggleUrl();
      if (tab) tab.location.href = url;
      else window.open(url, "_blank", "noopener");
      setNote(
        "Kaggle is in the other tab. Turn on a GPU and Internet, then Save Version → Save & Run All. When it finishes, download distressnet_results.zip from Output and load it here.",
      );
    } catch (err) {
      tab?.close();
      setError(err instanceof Error ? err.message : "Kaggle could not be opened.");
    }
  }

  async function loadResults(files: FileList | null) {
    if (!files?.length) return;
    setError("");
    setLoading(true);
    try {
      const done = await loadKaggleResults(Array.from(files));
      const model = done.voice_model === "wavlm" ? "Monitor now uses the new WavLM model." : done.voice_error;
      setNote(`Loaded ${done.installed.join(", ")}. ${model}`);
      loadVoice();
      onLoaded();
    } catch (err) {
      setError(err instanceof Error ? err.message : "The Kaggle results could not be loaded.");
    } finally {
      setLoading(false);
    }
  }

  const summary = voice?.summary;
  const history = voice?.history ?? [];
  const misses = (voice?.predictions ?? []).filter((item) => !item.correct);
  const layers = summary?.layer_weights ?? [];
  const topLayers = layers
    .map((weight, index) => ({ index, weight }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 3);

  return (
    <>
      <header className="page-head">
        <p className="eyebrow">Training: the live voice model</p>
        <h1>WavLM listens. A small head decides distress.</h1>
        <p className="lede">
          {summary
            ? `A frozen ${summary.encoder} encoder turns each clip into 13 layers of features; the ${summary.head} head mixes the layers and scores distress. It trained on ${summary.train_clips} clips from ${summary.corpora.join(" and ")}, chose epoch ${summary.selected_epoch} on ${summary.val_clips} validation clips, and ${summary.test} (${summary.test_clips} clips) scored it once.`
            : voice?.error || "Loading the voice model…"}
        </p>
        <div className="train-bar">
          <button type="button" className="detect-btn" onClick={() => void trainOnGpu()}>
            Train on GPU
          </button>
          <label className="file-btn">
            {loading ? "Loading results" : "Load Kaggle results"}
            <input
              type="file"
              accept=".zip,.pt,.json,.md"
              multiple
              disabled={loading}
              onChange={(event) => {
                void loadResults(event.target.files);
                event.target.value = "";
              }}
            />
          </label>
          <p className="fine">
            {note ||
              "Train on GPU opens the Kaggle notebook with the RAVDESS and CREMA-D datasets attached. It trains this model and reruns every evaluation; load its distressnet_results.zip here and Monitor uses the new model."}
          </p>
        </div>
        {error ? <p className="status error">{error}</p> : null}
      </header>

      {summary ? (
        <div className="stat-grid">
          <Stat
            label="Balanced accuracy"
            value={summary.test_balanced_accuracy.toFixed(2)}
            note={`calibrated, threshold ${summary.threshold.toFixed(2)}`}
          />
          <Stat label="AUROC" value={summary.test_auroc?.toFixed(2) ?? "-"} note="held-out actors 21–24" />
          <Stat
            label="Specificity"
            value={summary.test_specificity?.toFixed(2) ?? "-"}
            note="calm clips left calm"
          />
          <Stat label="Calibration error" value={summary.test_ece?.toFixed(3) ?? "-"} note={`temperature ${summary.temperature.toFixed(2)}`} />
        </div>
      ) : null}

      {history.length ? (
        <div className="chart-grid">
          <figure>
            <figcaption>Loss</figcaption>
            <LineChart
              series={[
                { name: "Train", color: "#1b1914", values: history.map((row) => row.train_loss) },
                { name: "Validation", color: "#a33b2b", values: history.map((row) => row.val_loss) },
              ]}
            />
            <p className="legend">
              <i className="swatch ink" /> Train <i className="swatch clay" /> Validation
            </p>
          </figure>
          <figure>
            <figcaption>Scores at threshold 0.5</figcaption>
            <LineChart
              series={[
                { name: "Val balanced accuracy", color: "#a9782e", values: history.map((row) => row.val_balanced_accuracy) },
                { name: "Test F1", color: "#1f4d3a", values: history.map((row) => row.test_f1) },
              ]}
            />
            <p className="legend">
              <i className="swatch gold" /> Val balanced accuracy (chooses the epoch) <i className="swatch pine" /> Test F1
              (only drawn)
            </p>
          </figure>
        </div>
      ) : summary ? (
        <p className="fine">
          This model was trained before epochs were logged. Train on GPU again to draw every epoch and every test clip.
        </p>
      ) : null}

      {summary && (topLayers.length || misses.length) ? (
        <div className="split-layout">
          {topLayers.length ? (
            <article className="panel">
              <h2>Which WavLM layers it listens to</h2>
              <p className="fine">
                The head learns one weight per layer (0 is the CNN feature encoder, 12 the last transformer layer).
              </p>
              <ul>
                {topLayers.map((item) => (
                  <li key={item.index}>
                    Layer {item.index}: {(item.weight * 100).toFixed(1)}%
                  </li>
                ))}
              </ul>
            </article>
          ) : null}
          {voice?.predictions?.length ? (
            <article className="panel">
              <h2>Held-out clips it got wrong</h2>
              <p className="fine">
                {misses.length} of {voice.predictions.length} test clips, at threshold {summary.threshold.toFixed(2)}.
              </p>
              <div className="epoch-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Clip</th>
                      <th>Emotion</th>
                      <th>Truth</th>
                      <th>P(distress)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {misses.slice(0, 10).map((item) => (
                      <tr key={item.file}>
                        <td>{item.file}</td>
                        <td>{item.emotion}</td>
                        <td>{item.truth}</td>
                        <td>{item.probability.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          ) : null}
        </div>
      ) : null}
    </>
  );
}

// The live voice model's head; the table opens on it when the report has it.
const LIVE_STUDY_MODEL = "wavlm/binary-mean";

function studyModelLabel(key: string): string {
  const [model, norm] = key.split("+");
  const name = model === "cnn" ? "CNN" : model === "egemaps" ? "eGeMAPS" : model.replace("wavlm", "WavLM").replace("/", " ");
  return norm ? `${name}, ${norm} normalization` : name;
}

function StudySection() {
  const [study, setStudy] = useState<StudyReport | null>(null);
  const [model, setModel] = useState("");

  useEffect(() => {
    getStudy()
      .then(setStudy)
      .catch(() => setStudy({ ready: false, summary: "The cross-corpus study could not be loaded." }));
  }, []);

  const models = study?.models ?? (study?.runs ? { cnn: study.runs } : {});
  const keys = Object.keys(models);
  const chosen = keys.includes(model) ? model : keys.includes(LIVE_STUDY_MODEL) ? LIVE_STUDY_MODEL : keys[0];
  const rows = chosen ? models[chosen] : [];

  return (
    <article className="panel">
      <h2>Cross-corpus audio study</h2>
      <p className="fine">
        IEMOCAP is not on this machine, so the audio result is RAVDESS and CREMA-D. The shared label is sad, angry, and
        fearful against neutral. Calm is left out of this map because CREMA-D has no calm class. Text is not a trained
        branch: both corpora repeat fixed sentences in every emotion.
      </p>
      {!study?.ready ? <p className="fine">{study?.summary ?? "Loading the study…"}</p> : null}
      {study?.ready && rows.length ? (
        <>
          <p className="fine">{study.selection}</p>
          <label className="fine">
            Model{" "}
            <select value={chosen} onChange={(event) => setModel(event.target.value)}>
              {keys.map((key) => (
                <option key={key} value={key}>
                  {studyModelLabel(key)}
                </option>
              ))}
            </select>
          </label>
          {study.updated ? (
            <p className="fine">
              Report written{" "}
              {new Date(study.updated * 1000).toLocaleString()}.
            </p>
          ) : null}
          <div className="epoch-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Balanced accuracy</th>
                  <th>Specificity</th>
                  <th>AUROC</th>
                  <th>Distress F1</th>
                  <th>Always-majority F1</th>
                  <th>Clips</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((run) => {
                  const ci = run.calibrated?.balanced_accuracy_ci95;
                  return (
                    <tr key={run.name}>
                      <td>{run.name}</td>
                      <td>
                        {(run.calibrated?.balanced_accuracy ?? run.test_balanced_accuracy).toFixed(2)}
                        {ci ? ` [${ci[0].toFixed(2)}, ${ci[1].toFixed(2)}]` : ""}
                      </td>
                      <td>{(run.calibrated?.specificity ?? run.test_specificity)?.toFixed(2) ?? "-"}</td>
                      <td>{run.test_auroc?.toFixed(2) ?? "-"}</td>
                      <td>{(run.calibrated?.f1 ?? run.test_f1).toFixed(2)}</td>
                      <td>{run.majority_f1.toFixed(2)}</td>
                      <td>
                        {run.train_clips} train / {run.test_clips} test
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="fine">
            Balanced accuracy, specificity and F1 are calibrated on validation speakers when the run has it; brackets are
            the speaker-bootstrap 95% interval. Normalized runs standardize features with unlabeled audio from the test
            corpus or speaker.
          </p>
          <p className="fine">{study.text?.finding}</p>
          <p className="fine">{study.iemocap}</p>
          {study.disagreement?.ready && study.disagreement.cells ? (
            <>
              <h3>Words against voice</h3>
              <p className="fine">{study.disagreement.rule}</p>
              <div className="epoch-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Assigned case</th>
                      <th>Clips</th>
                      <th>Voice heard distress</th>
                      <th>Judge distress</th>
                      <th>Judge calm</th>
                    </tr>
                  </thead>
                  <tbody>
                    {study.disagreement.cells.map((cell) => (
                      <tr key={cell.case}>
                        <td>{cell.case}</td>
                        <td>{cell.clips}</td>
                        <td>{cell.heard_distress}</td>
                        <td>{cell.judge_distress}</td>
                        <td>{cell.judge_calm}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
        </>
      ) : null}
    </article>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <article className="stat">
      <p className="eyebrow">{label}</p>
      <p className="stat-value">{value}</p>
      <p className="fine">{note}</p>
    </article>
  );
}
