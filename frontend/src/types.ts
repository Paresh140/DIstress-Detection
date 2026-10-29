export type Signal = {
  label: "distress" | "calm" | null;
  probability: number | null;
  logit?: number;
  source?: string;
  evidence?: string;
  note?: string;
  threshold?: number;
  model?: string;
  classes?: Record<string, number>;
};

export type Band = "calm" | "check-in" | "support" | "crisis";

export type Judge = {
  band?: Band;
  risk?: "none" | "concern" | "high";
  voice_band?: string | null;
  words_band?: string | null;
  decision: "distress" | "calm";
  confidence: number;
  reason: string;
  ask: boolean;
  question: string;
  source?: string;
};

export type Analysis = {
  file: string;
  label: "distress" | "calm";
  probability: number;
  logit: number;
  threshold: number;
  modality: string;
  band?: Band;
  transcript?: string;
  speech_to_text?: { text: string; model: string; duration: number } | null;
  voice?: Signal;
  language?: Signal;
  judge?: Judge;
  taps: Record<string, number>;
  embedding: number[];
  spectrogram: number[][];
};

export type LayerSpec = {
  id: string;
  name: string;
  kind: string;
  shape: string;
  detail: string;
  params: number;
};

export type EpochSnap = {
  epoch: number;
  train_loss?: number;
  test_loss?: number;
  test_accuracy?: number;
  test_precision?: number;
  test_recall?: number;
  test_f1: number;
  val_balanced_accuracy?: number;
  activation?: Record<string, number>;
  grad_norm?: Record<string, number>;
};

export type NetworkSpec = {
  name: string;
  task: string;
  loss: string;
  optimizer: string;
  threshold: number;
  audio_layers: LayerSpec[];
  vision_layers: LayerSpec[];
  audio_params: number;
  vision_params: number;
  fusion_params: number;
  trained: boolean;
  history: EpochSnap[];
  best_epoch: number | null;
  best?: {
    epoch: number;
    f1: number;
    accuracy: number;
    precision: number;
    recall: number;
    loss: number;
  };
};

export type Prediction = {
  file: string;
  actor: number;
  emotion: string;
  intensity: string;
  truth: "distress" | "calm";
  predicted: "distress" | "calm";
  probability: number;
  correct: boolean;
};

export type TrainEpoch = {
  epoch: number;
  train_loss: number;
  test_loss?: number;
  train_accuracy?: number;
  test_accuracy?: number;
  test_precision?: number;
  test_recall?: number;
  test_f1: number;
  val_balanced_accuracy?: number;
  grad_norm?: Record<string, number>;
  activation?: Record<string, number>;
};

export type TrainingReport = {
  dataset: string;
  citation: string;
  seed: number;
  device: string;
  epochs_ran: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  weight_decay: number;
  loss: string;
  pos_weight: number;
  threshold: number;
  augmentation: string;
  narrative: string;
  logged_only?: boolean;
  label: { distress: string[]; calm: string[]; excluded: string[] };
  split: {
    train_actors: number[];
    test_actors: number[];
    val_actors?: number[];
    unused_actors: number[];
    train_clips: number;
    test_clips: number;
    train_distress: number;
    train_calm: number;
  };
  parameters: { audio: number; vision_untrained: number; fusion_untrained: number };
  history: TrainEpoch[];
  best: {
    epoch: number;
    f1: number;
    accuracy: number;
    precision: number;
    recall: number;
    loss?: number | null;
  };
  confusion?: { tn: number; fp: number; fn: number; tp: number };
  predictions: Prediction[];
};

export type Sample = {
  file: string;
  actor: number;
  emotion: string;
  intensity: string;
  truth: string;
};

export type ChatMessage = { role: "user" | "assistant"; text: string };
