import { useCallback, useEffect, useRef, useState } from "react";
import { addEnrollment, getEnrollment, resetEnrollment, type Enrollment } from "../api";
import { MicRecorder } from "../voice";

const MAX_SECONDS = 10;
const SKIP_KEY = "distressnet-calibration-skipped";

function readSkipped(): boolean {
  try {
    return window.localStorage.getItem(SKIP_KEY) === "1";
  } catch {
    return false;
  }
}

function writeSkipped(value: boolean) {
  try {
    if (value) window.localStorage.setItem(SKIP_KEY, "1");
    else window.localStorage.removeItem(SKIP_KEY);
  } catch {
    /* the choice is only remembered when storage works */
  }
}

/**
 * The first step on Monitor: the person reads a few neutral sentences in their normal voice,
 * and the voice model is z-scored against that reading from then on. Without it, a new
 * microphone looks like a new corpus and the voice model calls nearly everything distress.
 */
export function Calibration({ onChange }: { onChange?: (status: Enrollment) => void }) {
  const [status, setStatus] = useState<Enrollment | null>(null);
  const [skipped, setSkipped] = useState(readSkipped);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const recorder = useRef<MicRecorder | null>(null);
  const started = useRef(0);

  const update = useCallback(
    (next: Enrollment) => {
      setStatus(next);
      onChange?.(next);
    },
    [onChange],
  );

  useEffect(() => {
    getEnrollment().then(update).catch(() => setStatus(null));
    return () => recorder.current?.release();
  }, [update]);

  const stop = useCallback(async () => {
    const active = recorder.current;
    recorder.current = null;
    setRecording(false);
    if (!active) return;
    const file = active.stop(1.0);
    if (!file) {
      setError("That was too short. Read the whole sentence, then press Done.");
      return;
    }
    setBusy(true);
    try {
      update(await addEnrollment(file));
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save that reading.");
    } finally {
      setBusy(false);
    }
  }, [update]);

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => {
      const seconds = (performance.now() - started.current) / 1000;
      setElapsed(seconds);
      if (seconds >= MAX_SECONDS) void stop();
    }, 200);
    return () => window.clearInterval(timer);
  }, [recording, stop]);

  async function start() {
    setError("");
    const next = new MicRecorder();
    try {
      await next.start();
    } catch {
      next.release();
      setError("Microphone access was blocked. Allow the mic, then try again.");
      return;
    }
    recorder.current = next;
    started.current = performance.now();
    setElapsed(0);
    setRecording(true);
  }

  if (!status) return null;
  const done = status.active;
  const sentence = status.script[status.clips % Math.max(status.script.length, 1)] ?? "";
  const target = Math.max(status.needed, Math.min(status.recommended, status.script.length));
  const shown = Math.min(status.clips, target);

  if (skipped && !done) {
    return (
      <div className="calibration compact">
        <p>
          Voice not calibrated. Readings from a new microphone lean toward distress until it is.{" "}
          <button type="button" className="link" onClick={() => { writeSkipped(false); setSkipped(false); }}>
            Calibrate now
          </button>
        </p>
      </div>
    );
  }

  if (done) {
    return (
      <div className="calibration compact done">
        <p>
          Voice calibrated from {status.clips} calm readings. Each clip is now compared with how you sound when calm.{" "}
          {status.clips < status.recommended ? (
            <button type="button" className="link" onClick={() => void start()} disabled={recording || busy}>
              {recording ? "Listening…" : `Add another (${status.recommended} is better)`}
            </button>
          ) : null}{" "}
          <button type="button" className="link" disabled={recording || busy} onClick={() => void resetEnrollment().then(update)}>
            Start over
          </button>
        </p>
        {recording ? (
          <p className="calibration-line">
            “{sentence}” <button type="button" className="detect-btn" onClick={() => void stop()}>Done</button>
          </p>
        ) : null}
      </div>
    );
  }

  return (
    <section className="calibration" aria-label="Calibrate your voice">
      <p className="eyebrow">Step 1 · Calibrate your voice</p>
      <h2>Speak these sentences so I can learn your voice.</h2>
      <p className="fine">
        Read each one in your normal, calm voice. The voice model then judges every clip against how you sound when
        calm, instead of against the actors it was trained on. Nothing is diagnosed, and the readings stay on this
        machine.
      </p>
      {status.available ? (
        <>
          <div className="calibration-progress" role="progressbar" aria-valuemin={0} aria-valuemax={target} aria-valuenow={shown}>
            {Array.from({ length: target }, (_, i) => (
              <span key={i} className={i < shown ? "on" : ""} />
            ))}
            <small>
              {shown} of {status.needed} needed{target > status.needed ? `, ${target} is better` : ""}
            </small>
          </div>
          <p className="calibration-line">“{sentence}”</p>
          <div className="record-row">
            <button
              type="button"
              className={`record-btn ${recording ? "live" : ""}`}
              disabled={busy}
              onClick={() => (recording ? void stop() : void start())}
            >
              <i />
              {recording ? `Done (0:${String(Math.floor(elapsed)).padStart(2, "0")})` : busy ? "Saving…" : "Read it"}
            </button>
            <button type="button" className="link" onClick={() => { writeSkipped(true); setSkipped(true); }}>
              Skip for now
            </button>
          </div>
        </>
      ) : (
        <p className="status error">
          {status.note} Until then the voice model runs uncalibrated.{" "}
          <button type="button" className="link" onClick={() => { writeSkipped(true); setSkipped(true); }}>
            Continue without it
          </button>
        </p>
      )}
      {error ? <p className="status error">{error}</p> : null}
    </section>
  );
}
