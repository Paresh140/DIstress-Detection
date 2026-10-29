import { useEffect, useRef, useState } from "react";
import { addEnroll, analyzeClip, analyzeUpload, analyzeWords, clipUrl, getEnroll, getSamples, resetEnroll } from "../api";
import type { EnrollStatus } from "../api";
import type { Analysis, Sample } from "../types";

const MAX_RECORD_SECONDS = 8;

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `0:${String(whole).padStart(2, "0")}`;
}

type SpeechRec = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecEvent) => void) | null;
  onerror: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecEvent = {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
};

function createRecognizer(onText: (text: string) => void): SpeechRec | null {
  const host = window as Window & {
    SpeechRecognition?: new () => SpeechRec;
    webkitSpeechRecognition?: new () => SpeechRec;
  };
  const Ctor = host.SpeechRecognition || host.webkitSpeechRecognition;
  if (!Ctor) return null;
  const recognition = new Ctor();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";
  recognition.onresult = (event) => {
    let text = "";
    for (let index = 0; index < event.results.length; index += 1) {
      text += `${event.results[index][0]?.transcript ?? ""} `;
    }
    onText(text.trim());
  };
  recognition.onerror = () => undefined;
  return recognition;
}

function drawSpectrogram(canvas: HTMLCanvasElement, spec: number[][]) {
  const rows = spec.length;
  const cols = spec[0]?.length ?? 0;
  canvas.width = cols;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  if (!context || !cols) return;
  let min = Infinity;
  let max = -Infinity;
  for (const row of spec) {
    for (const value of row) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  const span = max - min || 1;
  const image = context.createImageData(cols, rows);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const t = (spec[rows - 1 - r][c] - min) / span;
      const index = (r * cols + c) * 4;
      image.data[index] = 28 + t * 150;
      image.data[index + 1] = 32 + (1 - Math.abs(t - 0.55)) * 40;
      image.data[index + 2] = 24 + (1 - t) * 30;
      image.data[index + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

export function Monitor({
  analysis,
  onAnalysis,
}: {
  analysis: Analysis | null;
  onAnalysis: (analysis: Analysis) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const startedRef = useRef(0);
  const recognitionRef = useRef<SpeechRec | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [audioSrc, setAudioSrc] = useState("");
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);
  const [recordedUrl, setRecordedUrl] = useState("");
  const [transcript, setTranscript] = useState("");
  const [enroll, setEnroll] = useState<EnrollStatus | null>(null);

  useEffect(() => {
    getSamples().then(setSamples).catch(() => setSamples([]));
    getEnroll().then(setEnroll).catch(() => setEnroll(null));
    return () => releaseMic();
  }, []);

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => {
      const seconds = (performance.now() - startedRef.current) / 1000;
      setElapsed(seconds);
      if (seconds >= MAX_RECORD_SECONDS) stopRecording();
    }, 200);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(() => {
    if (canvasRef.current && analysis) drawSpectrogram(canvasRef.current, analysis.spectrogram);
  }, [analysis]);

  function stopRecognition() {
    try {
      recognitionRef.current?.stop();
    } catch {
      /* already stopped */
    }
    recognitionRef.current = null;
  }

  function releaseMic() {
    stopRecognition();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const context = contextRef.current;
    contextRef.current = null;
    if (context && context.state !== "closed") void context.close();
  }

  function finishRecording() {
    const context = contextRef.current;
    const chunks = chunksRef.current;
    const rate = context?.sampleRate ?? 48000;
    releaseMic();
    setRecording(false);
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (length < rate * 0.4) {
      setRecordedFile(null);
      setError("That recording is too short. Hold Record and speak for a couple of seconds.");
      return;
    }
    const samples = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    const blob = encodeWav(samples, rate);
    const file = new File([blob], "recording.wav", { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    setRecordedUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return url;
    });
    setRecordedFile(file);
    setAudioSrc(url);
    setError("");
  }

  function stopRecording() {
    if (!contextRef.current && !streamRef.current) return;
    finishRecording();
  }

  async function startRecording() {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context = new AudioContext();
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const silent = context.createGain();
      silent.gain.value = 0;
      chunksRef.current = [];
      processor.onaudioprocess = (event) => {
        chunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(silent);
      silent.connect(context.destination);
      streamRef.current = stream;
      contextRef.current = context;
      startedRef.current = performance.now();
      setElapsed(0);
      setTranscript("");
      const recognition = createRecognizer(setTranscript);
      recognitionRef.current = recognition;
      try {
        recognition?.start();
      } catch {
        /* recognition is optional */
      }
      setRecording(true);
    } catch {
      setError("Microphone access was blocked. Allow the mic, then press Record again.");
    }
  }

  async function run(task: () => Promise<Analysis>, src: string) {
    setBusy(true);
    setError("");
    try {
      const result = await task();
      onAnalysis(result);
      setAudioSrc(src);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyze that clip.");
    } finally {
      setBusy(false);
    }
  }

  async function enrollCalm(task: () => Promise<EnrollStatus>) {
    setBusy(true);
    setError("");
    try {
      setEnroll(await task());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save that calm reading.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="monitor">
      <header className="page-head">
        <p className="eyebrow">Monitor</p>
        <h1>Listen for the distress class.</h1>
        <p className="lede">
          A voice model scores how the clip sounds, and Whisper writes down the words for a language model to read. The
          judge puts the reading in one of three bands: calm, check-in, or support. A crisis screen runs on the words
          first. If the band is check-in or support, the assistant asks what is going on and listens. This is not a
          diagnosis.
        </p>
      </header>

      <div className="stage">
        <div
          className="drop"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            const file = event.dataTransfer.files?.[0];
            if (!file) return;
            const url = URL.createObjectURL(file);
            void run(() => analyzeUpload(file, transcript.trim()), url);
          }}
        >
          <p>Record your voice, then detect distress. A wav file or a held-out clip works too.</p>
          <div className="record-row">
            <button
              type="button"
              className={`record-btn ${recording ? "live" : ""}`}
              disabled={busy}
              onClick={() => (recording ? stopRecording() : void startRecording())}
            >
              <i />
              {recording ? `Stop ${formatClock(elapsed)}` : "Record"}
            </button>
            <button
              type="button"
              className="detect-btn"
              disabled={(!recordedFile && !transcript.trim()) || recording || busy}
              onClick={() => {
                const words = transcript.trim();
                if (recordedFile) {
                  void run(() => analyzeUpload(recordedFile, words), recordedUrl);
                  return;
                }
                void run(() => analyzeWords(words), "");
              }}
            >
              Detect distress
            </button>
          </div>
          <label className="words">
            Words
            <textarea
              value={transcript}
              rows={2}
              placeholder="Heard while you speak, or type it. For example: I am not feeling well today."
              onChange={(event) => setTranscript(event.target.value)}
            />
          </label>
          {recordedFile && !recording ? (
            <p className="fine">Recording ready, {formatClock(elapsed)}. Press Detect distress.</p>
          ) : null}
          {enroll?.available ? (
            <div className="enroll">
              <p className="fine">
                {enroll.active
                  ? "Calm enrollment is on: your voice is compared with your own calm readings."
                  : `Calm enrollment: ${enroll.clips} of ${enroll.needed} readings. Read a sentence calmly, then save it as a calm reading.`}
              </p>
              <button
                type="button"
                disabled={!recordedFile || recording || busy}
                onClick={() => recordedFile && void enrollCalm(() => addEnroll(recordedFile))}
              >
                Save as calm reading
              </button>
              {enroll.clips > 0 ? (
                <button type="button" disabled={busy} onClick={() => void enrollCalm(resetEnroll)}>
                  Reset
                </button>
              ) : null}
            </div>
          ) : null}
          <label className="file-btn">
            Choose wav
            <input
              type="file"
              accept="audio/wav,.wav"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                void run(() => analyzeUpload(file), URL.createObjectURL(file));
              }}
            />
          </label>
          <div className="sample-row">
            {samples.map((sample) => (
              <button
                key={sample.file}
                type="button"
                disabled={busy}
                onClick={() => void run(() => analyzeClip(sample.file), clipUrl(sample.file))}
              >
                Actor {String(sample.actor).padStart(2, "0")} · {sample.emotion}
                <span>{sample.truth}</span>
              </button>
            ))}
          </div>
          {busy ? <p className="status">The voice network, the language model, and the judge are reading this.</p> : null}
          {error ? <p className="status error">{error}</p> : null}
        </div>

        <div className={`reading ${analysis?.label ?? "empty"}`}>
          {analysis ? (
            <>
              <p className="fine">{analysis.judge ? "Judge" : analysis.file}</p>
              <p className="score">{analysis.probability.toFixed(2)}</p>
              <p className="verdict">{analysis.band ?? analysis.label}</p>
              {analysis.voice?.probability != null || analysis.language?.probability != null ? (
                <ul className="signals">
                  {analysis.voice?.probability != null ? (
                    <li>
                      {analysis.voice.model?.includes("wavlm") ? "WavLM voice" : "Voice network"}
                      {analysis.voice.enrolled ? ", against your calm readings" : ""}{" "}
                      <b>{analysis.judge?.voice_band ?? analysis.voice.label}</b> {analysis.voice.probability.toFixed(2)}
                      {analysis.voice.threshold != null ? ` (threshold ${analysis.voice.threshold.toFixed(2)})` : ""}
                    </li>
                  ) : null}
                  {analysis.language?.probability != null ? (
                    <li>
                      {analysis.language.source === "language model" ? "Language model" : "Words"}{" "}
                      <b>{analysis.judge?.words_band ?? analysis.language.label}</b> {analysis.language.probability.toFixed(2)}
                    </li>
                  ) : null}
                  {analysis.judge?.risk && analysis.judge.risk !== "none" ? (
                    <li>
                      Crisis screen <b>{analysis.judge.risk}</b>
                    </li>
                  ) : null}
                </ul>
              ) : null}
              <p className="fine">{analysis.judge?.reason ?? `Threshold 0.50. Logit ${analysis.logit.toFixed(2)}.`}</p>
              {analysis.speech_to_text ? (
                <p className="fine">
                  {analysis.speech_to_text.model} heard: “{analysis.speech_to_text.text || "no words"}”
                </p>
              ) : null}
              {analysis.embedding.length ? (
                <div className="embed" aria-hidden="true">
                  {analysis.embedding.map((value, index) => (
                    <span key={index} style={{ height: `${Math.min(100, Math.abs(value) * 80)}%` }} />
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <>
              <p className="eyebrow">No clip yet</p>
              <p className="score faint">0.50</p>
              <p className="verdict">waiting</p>
              <p className="fine">
                At or above the voice threshold is support, just under it is a check-in, and well under it is calm. The
                words are banded the same way, and the judge keeps the higher band.
              </p>
            </>
          )}
        </div>
      </div>

      <figure className="spec-card">
        <figcaption>
          <span>{analysis?.voice?.model?.includes("wavlm") ? "The clip (WavLM reads the waveform itself)" : "What the network sees"}</span>
          <span>64 mel bands × 128 frames</span>
        </figcaption>
        <canvas ref={canvasRef} className={analysis?.spectrogram?.length ? "" : "hidden-canvas"} />
        {!analysis?.spectrogram?.length ? (
          <p className="fine">
            {analysis ? "No voice clip was attached, so there is no spectrogram." : "The spectrogram appears after a clip is analyzed."}
          </p>
        ) : null}
        {audioSrc ? <audio controls src={audioSrc} /> : null}
      </figure>
    </section>
  );
}
