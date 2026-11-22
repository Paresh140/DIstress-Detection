import type { Analysis, ChatCard, NetworkSpec, Sample, TrainEpoch, TrainingReport } from "./types";

// One id per browser, so the server can keep this person's calm enrollment between requests.
function makeSession(): string {
  const random = Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, "0")).join("");
  return `s-${random}`;
}

let memorySession = "";

export function sessionId(): string {
  try {
    const saved = window.localStorage.getItem("distressnet-session");
    if (saved) return saved;
    const fresh = makeSession();
    window.localStorage.setItem("distressnet-session", fresh);
    return fresh;
  } catch {
    if (!memorySession) memorySession = makeSession();
    return memorySession;
  }
}

function sessionHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return { "X-Session": sessionId(), ...extra };
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function getHealth(): Promise<{ ok: boolean; weights: boolean; report: boolean }> {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getNetwork(): Promise<NetworkSpec> {
  const response = await fetch("/api/network");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export type StudyRun = {
  name: string;
  train_clips: number;
  test_clips: number;
  train_speaker_count: number;
  test_speaker_count: number;
  selected_epoch: number;
  val_balanced_accuracy: number;
  test_accuracy: number;
  test_balanced_accuracy: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  majority_f1: number;
  test_specificity?: number;
  test_auroc?: number;
  calibrated?: {
    balanced_accuracy: number;
    specificity: number;
    f1: number;
    balanced_accuracy_ci95?: [number, number];
  };
};

export type DisagreementCell = {
  case: string;
  clips: number;
  heard_distress: number;
  judge_distress: number;
  judge_calm: number;
};

export type StudyReport = {
  ready: boolean;
  summary?: string;
  selection?: string;
  iemocap?: string;
  text?: { finding: string };
  runs?: StudyRun[];
  models?: Record<string, StudyRun[]>;
  updated?: number;
  disagreement?: {
    ready: boolean;
    note?: string;
    rule?: string;
    cells?: DisagreementCell[];
  };
};

export async function getStudy(): Promise<StudyReport> {
  const response = await fetch("/api/study");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export type VoiceEpoch = {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_balanced_accuracy: number;
  test_f1: number;
  test_balanced_accuracy: number;
};

export type VoiceReport = {
  ready: boolean;
  error?: string;
  summary?: {
    encoder: string;
    head: string;
    train_clips: number;
    val_clips: number;
    test: string;
    test_clips: number;
    corpora: string[];
    selected_epoch: number;
    temperature: number;
    threshold: number;
    test_balanced_accuracy: number;
    test_auroc: number | null;
    test_specificity?: number;
    test_ece?: number;
    layer_weights?: number[];
  };
  epochs?: number;
  history?: VoiceEpoch[];
  predictions?: {
    file: string;
    speaker: string;
    emotion: string;
    truth: "distress" | "calm";
    probability: number;
    correct: boolean;
  }[];
};

export async function getVoiceReport(): Promise<VoiceReport> {
  const response = await fetch("/api/voice");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getKaggleUrl(): Promise<{ url: string }> {
  const response = await fetch("/api/voice/kaggle");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export type KaggleResults = {
  ok: boolean;
  installed: string[];
  voice_model: "wavlm" | "cnn";
  voice_error: string;
};

export async function loadKaggleResults(files: File[]): Promise<KaggleResults> {
  const body = new FormData();
  for (const file of files) body.append("files", file);
  const response = await fetch("/api/voice/results", { method: "POST", body });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export type TrainLive = {
  running: boolean;
  phase: "idle" | "loading" | "training" | "done" | "error";
  epoch: number;
  epochs: number;
  history: TrainEpoch[];
  best: TrainingReport["best"] | null;
  error: string;
};

export async function startTraining(): Promise<{ ok: boolean; epochs: number }> {
  const response = await fetch("/api/train/start", { method: "POST" });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getTrainLive(): Promise<TrainLive> {
  const response = await fetch("/api/train/live");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getTraining(): Promise<TrainingReport> {
  const response = await fetch("/api/training");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getSamples(): Promise<Sample[]> {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error(await readError(response));
  const body = await response.json();
  return body.samples;
}

export async function analyzeClip(file: string): Promise<Analysis> {
  const response = await fetch("/api/analyze/clip", {
    method: "POST",
    headers: sessionHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ file }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function analyzeUpload(file: File, transcript = ""): Promise<Analysis> {
  const body = new FormData();
  body.append("file", file);
  if (transcript.trim()) body.append("transcript", transcript.trim());
  const response = await fetch("/api/analyze/upload", { method: "POST", body, headers: sessionHeaders() });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function analyzeWords(transcript: string): Promise<Analysis> {
  const response = await fetch("/api/analyze/words", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

type ChatReply = { reply: string; prompts: string[]; state: Record<string, unknown> | null; card: ChatCard | null };

function slimAnalysis(analysis: Analysis | null) {
  return analysis
    ? {
        file: analysis.file,
        label: analysis.label,
        probability: analysis.probability,
        logit: analysis.logit,
        modality: analysis.modality,
        transcript: analysis.transcript,
        voice: analysis.voice,
        language: analysis.language,
        judge: analysis.judge,
      }
    : null;
}

/** One spoken chat turn: the server transcribes it, reads the voice and answers. */
export async function sendVoiceChat(
  file: File,
  analysis: Analysis | null,
  history: { role: string; text: string }[] = [],
  state: Record<string, unknown> | null = null,
): Promise<ChatReply & { heard: string; reading: Partial<Analysis> }> {
  const body = new FormData();
  body.append("file", file);
  body.append(
    "payload",
    JSON.stringify({
      analysis: slimAnalysis(analysis),
      history: history.slice(-8).map((turn) => ({ role: turn.role, text: turn.text })),
      state,
    }),
  );
  const response = await fetch("/api/chat/voice", { method: "POST", body, headers: sessionHeaders() });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function sendChat(
  message: string,
  analysis: Analysis | null,
  history: { role: string; text: string }[] = [],
  state: Record<string, unknown> | null = null,
): Promise<ChatReply> {
  const slim = slimAnalysis(analysis);
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      analysis: slim,
      history: history.slice(-8).map((turn) => ({ role: turn.role, text: turn.text })),
      state,
    }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export function clipUrl(file: string): string {
  const [actor, name] = file.split("/");
  return `/api/clips/${actor}/${name}`;
}

export type Enrollment = {
  available: boolean;
  clips: number;
  needed: number;
  recommended: number;
  active: boolean;
  script: string[];
  note: string;
};

export async function getEnrollment(): Promise<Enrollment> {
  const response = await fetch("/api/enroll", { headers: sessionHeaders() });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function addEnrollment(file: File): Promise<Enrollment> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/enroll", { method: "POST", body, headers: sessionHeaders() });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function resetEnrollment(): Promise<Enrollment> {
  const response = await fetch("/api/enroll", { method: "DELETE", headers: sessionHeaders() });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}
