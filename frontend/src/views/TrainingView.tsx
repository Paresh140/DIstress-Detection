import { useEffect, useState } from "react";
import { clipUrl, getGpuLive, getStudy, getTrainLive, getTraining, startGpuTraining, startTraining, type StudyReport, type TrainLive } from "../api";
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
  const [remote, setRemote] = useState(false);
  const [starting, setStarting] = useState(false);
  const [serverDone, setServerDone] = useState(false);
  const [liveHistory, setLiveHistory] = useState<TrainEpoch[] | null>(null);
  const [liveBest, setLiveBest] = useState<TrainingReport["best"] | null>(null);
  const [liveNote, setLiveNote] = useState("");
  const [plannedEpochs, setPlannedEpochs] = useState(16);
  const [overlay, setOverlay] = useState<TrainingReport | null>(null);
  const [loggedOnly, setLoggedOnly] = useState(false);

  function playLoggedRun(live: TrainLive) {
    if (!live.display) return;
    setOverlay(live.display);
    setLoggedOnly(true);
    setRemote(true);
    setLiveHistory(live.history);
    setLiveBest(live.best);
    setPlannedEpochs(live.epochs);
    setServerDone(true);
    setEpoch(1);
    setFollowing(true);
  }

  useEffect(() => {
    let active = true;
    Promise.all([getTraining(), getGpuLive()])
      .then(([next, live]) => {
        if (!active) return;
        setReport(next);
        const playSavedGpu = next.epochs_ran >= 24 && sessionStorage.getItem("distressnet-gpu-played") !== "1";
        if (playSavedGpu) {
          sessionStorage.setItem("distressnet-gpu-played", "1");
          setRemote(true);
          setLoggedOnly(false);
          setLiveHistory(next.history);
          setLiveBest(next.best);
          setPlannedEpochs(next.epochs_ran);
          setServerDone(true);
          setEpoch(1);
          setFollowing(true);
          return;
        }
        if (next.epochs_ran >= 24) {
          setEpoch(next.best.epoch);
          return;
        }
        if (live.logged_only && live.display && live.history.length) {
          playLoggedRun(live);
          return;
        }
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
    if (!loggedOnly) return;
    const timer = window.setInterval(() => {
      getTraining()
        .then((next) => {
          if (next.epochs_ran < 24) return;
          setReport(next);
          setOverlay(null);
          setLoggedOnly(false);
          setLiveHistory(null);
          setLiveBest(null);
          setFollowing(false);
          setServerDone(false);
          setEpoch(next.best.epoch);
          setLiveNote("Drive copied the checkpoint. Actors 21–24 scored it.");
        })
        .catch(() => undefined);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [loggedOnly]);

  useEffect(() => {
    if (!following || serverDone) return;
    const timer = window.setInterval(() => {
      const pull = remote ? getGpuLive() : getTrainLive();
      pull
        .then((live) => {
          setPlannedEpochs(live.epochs);
          if (live.history.length) {
            setLiveHistory(live.history);
            if (live.best) setLiveBest(live.best);
            setEpoch((current) => (current < 1 ? 1 : current));
          } else if (live.phase === "downloading") {
            setLiveNote("The GPU is downloading all 24 actors into Drive.");
          } else if (remote) {
            setLiveNote("Colab is in the other tab. This page stays here and draws each epoch.");
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
    }, remote ? 2000 : 700);
    return () => window.clearInterval(timer);
  }, [following, serverDone, remote]);

  useEffect(() => {
    if (!following || !liveHistory?.length || epoch < 1) return;
    if (epoch < liveHistory.length) {
      const timer = window.setTimeout(() => setEpoch((current) => current + 1), 1400);
      return () => window.clearTimeout(timer);
    }
    if (serverDone && epoch >= liveHistory.length) {
      const timer = window.setTimeout(() => {
        if (loggedOnly) {
          setFollowing(false);
          setServerDone(false);
          setLiveHistory(null);
          setEpoch(liveBest?.epoch ?? epoch);
          setLiveNote(
            "Actors 21–24 scored this run. The checkpoint is on Drive and has not reached this Mac yet, so the voice network here is still the earlier one.",
          );
          return;
        }
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
  }, [following, liveHistory, epoch, serverDone, loggedOnly, liveBest]);

  async function trainOnGpu() {
    setError("");
    setOverlay(null);
    setLoggedOnly(false);
    setRemote(true);
    setStarting(true);
    setEpoch(0);
    setLiveHistory(null);
    setLiveBest(null);
    setServerDone(false);
    setLiveNote("Colab is in the other tab. This page stays here and draws each epoch.");
    try {
      const started = await startGpuTraining();
      setPlannedEpochs(started.epochs);
      window.open(started.url, "_blank", "noopener");
      setStarting(false);
      setFollowing(true);
    } catch (err) {
      setStarting(false);
      setFollowing(false);
      setRemote(false);
      setLiveNote("");
      setError(err instanceof Error ? err.message : "The GPU run could not start.");
    }
  }

  async function trainAgain() {
    setError("");
    setOverlay(null);
    setLoggedOnly(false);
    setRemote(false);
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
    return <p className={error ? "status error" : "status"}>{error || "Loading the training run…"}</p>;
  }

  const view = overlay ?? report;
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
      <header className="page-head">
        <p className="eyebrow">Training</p>
        <h1>
          Actors {actorSpan(view.split.train_actors)} taught it. Actors {actorSpan(view.split.test_actors)} scored
          it.
        </h1>
        <p className="lede">{view.narrative}</p>
        <div className="train-bar">
          <button
            type="button"
            className="detect-btn"
            onClick={() => void trainOnGpu()}
            disabled={starting || showingLive}
          >
            {showingLive && remote ? (view.logged_only ? "Playing the GPU run" : "Training on GPU") : "Train on GPU"}
          </button>
          <button type="button" className="file-btn" onClick={() => void trainAgain()} disabled={starting || following}>
            {following && !remote ? "Training" : "Train on this Mac"}
          </button>
          <p className="fine">
            {showingLive
              ? `Epoch ${epoch} of ${plannedEpochs}. ${
                  view.logged_only
                    ? "These are the epochs Colab printed. Drive has not copied the checkpoint to this Mac yet."
                    : "Each epoch stays on the network so the forward and backward pass can be seen."
                }`
              : liveNote ||
                "Train on GPU opens the Colab run for all 24 actors. This page draws each epoch when Drive syncs it. Train on this Mac uses the clips already here."}
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
        {view.logged_only ? (
          <>
            Brightness follows this epoch’s test F1. Pale dots move forward and clay dots move backward.
            Layer activations and gradients are in the checkpoint file, which has not reached this Mac yet.
          </>
        ) : (
          <>
            Click a column, or a row in the epoch table. Brightness is activation
            {layerScale(history, stageEpoch, layer, "activation") !== null
              ? ` (${layerScale(history, stageEpoch, layer, "activation")!.toFixed(2)} on ${layer})`
              : ""}
            . Warmer edges are larger gradients on that epoch.
            {showingLive ? " Pale dots move forward and clay dots move backward on this epoch." : ""}
          </>
        )}
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
              The confusion counts are in the saved report. Drive has not copied that file to this Mac yet.
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
          <p className="fine">Held-out clip scores are in the saved report. Drive has not copied that file to this Mac yet.</p>
        )}
      </article>

      <StudySection />

      <p className="citation">{view.citation}</p>
    </section>
  );
}

function StudySection() {
  const [study, setStudy] = useState<StudyReport | null>(null);

  useEffect(() => {
    getStudy()
      .then(setStudy)
      .catch(() => setStudy({ ready: false, summary: "The cross-corpus study could not be loaded." }));
  }, []);

  return (
    <article className="panel">
      <h2>Cross-corpus audio study</h2>
      <p className="fine">
        IEMOCAP is not on this machine, so the audio result is RAVDESS and CREMA-D. The shared label is sad, angry, and
        fearful against neutral. Calm is left out of this map because CREMA-D has no calm class. Text is not a trained
        branch: both corpora repeat fixed sentences in every emotion.
      </p>
      {!study?.ready ? <p className="fine">{study?.summary ?? "Loading the study…"}</p> : null}
      {study?.ready && study.runs ? (
        <>
          <p className="fine">{study.selection}</p>
          <div className="epoch-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Balanced accuracy</th>
                  <th>Distress F1</th>
                  <th>Always-majority F1</th>
                  <th>Clips</th>
                </tr>
              </thead>
              <tbody>
                {study.runs.map((run) => (
                  <tr key={run.name}>
                    <td>{run.name}</td>
                    <td>{run.test_balanced_accuracy.toFixed(2)}</td>
                    <td>{run.test_f1.toFixed(2)}</td>
                    <td>{run.majority_f1.toFixed(2)}</td>
                    <td>
                      {run.train_clips} train / {run.test_clips} test
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
