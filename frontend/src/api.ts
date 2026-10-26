import type { Analysis, NetworkSpec, Sample, TrainEpoch, TrainingReport } from "./types";

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

export type TrainLive = {
  running: boolean;
  phase: "idle" | "loading" | "waiting" | "downloading" | "training" | "done" | "error";
  epoch: number;
  epochs: number;
  history: TrainEpoch[];
  best: TrainingReport["best"] | null;
  error: string;
  url?: string;
  logged_only?: boolean;
  narrative?: string;
  display?: TrainingReport | null;
};

export async function startTraining(): Promise<{ ok: boolean; epochs: number }> {
  const response = await fetch("/api/train/start", { method: "POST" });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function startGpuTraining(): Promise<{ ok: boolean; url: string; epochs: number }> {
  const response = await fetch("/api/train/gpu", { method: "POST" });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getGpuLive(): Promise<TrainLive> {
  const response = await fetch("/api/train/gpu");
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
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function analyzeUpload(file: File, transcript = ""): Promise<Analysis> {
  const body = new FormData();
  body.append("file", file);
  if (transcript.trim()) body.append("transcript", transcript.trim());
  const response = await fetch("/api/analyze/upload", { method: "POST", body });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export type EnrollStatus = { available: boolean; clips: number; needed: number; active: boolean };

export async function getEnroll(): Promise<EnrollStatus> {
  const response = await fetch("/api/enroll");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function addEnroll(file: File): Promise<EnrollStatus> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/enroll", { method: "POST", body });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function resetEnroll(): Promise<EnrollStatus> {
  const response = await fetch("/api/enroll", { method: "DELETE" });
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

export async function sendChat(
  message: string,
  analysis: Analysis | null,
  history: { role: string; text: string }[] = [],
  state: Record<string, unknown> | null = null,
): Promise<{ reply: string; prompts: string[]; state: Record<string, unknown> | null }> {
  const slim = analysis
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
