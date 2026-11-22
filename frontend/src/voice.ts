// Microphone recording to WAV and spoken replies, shared by Monitor and the chat.

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
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

/** Records the mic as raw samples; stop() returns a WAV file, or null if it was too short. */
export class MicRecorder {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private chunks: Float32Array[] = [];

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    const context = new AudioContext();
    await context.resume();
    const source = context.createMediaStreamSource(this.stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const silent = context.createGain();
    silent.gain.value = 0;
    this.chunks = [];
    processor.onaudioprocess = (event) => {
      this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(silent);
    silent.connect(context.destination);
    this.context = context;
  }

  stop(minSeconds = 0.4): File | null {
    const rate = this.context?.sampleRate ?? 48000;
    const chunks = this.chunks;
    this.release();
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (length < rate * minSeconds) return null;
    const samples = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    return new File([encodeWav(samples, rate)], "turn.wav", { type: "audio/wav" });
  }

  release(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    const context = this.context;
    this.context = null;
    this.chunks = [];
    if (context && context.state !== "closed") void context.close();
  }
}

export const canSpeak = typeof window !== "undefined" && "speechSynthesis" in window;

function pickVoice(): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices().filter((voice) => voice.lang.startsWith("en"));
  // Prefer the higher-quality voices browsers ship, then any local English voice.
  return (
    voices.find((voice) => /natural|neural|premium|enhanced|samantha|google us/i.test(voice.name)) ??
    voices.find((voice) => voice.localService) ??
    voices[0]
  );
}

/** Text as it should sound: no list markers, links or markup. */
export function speakable(text: string): string {
  return text
    .replace(/https?:\/\/\S+/g, "")
    .replace(/^\s*[-*•]\s+/gm, "")
    .replace(/^\s*\d+[.)]\s+/gm, "")
    .replace(/[*_`#>]/g, "")
    .replace(/\n+/g, ". ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Speaks the text; resolves when it finishes or is cancelled. */
export function speak(text: string): Promise<void> {
  if (!canSpeak) return Promise.resolve();
  const line = speakable(text);
  if (!line) return Promise.resolve();
  window.speechSynthesis.cancel();
  return new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(line);
    const voice = pickVoice();
    if (voice) utterance.voice = voice;
    utterance.rate = 0.95;
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    window.speechSynthesis.speak(utterance);
  });
}

export function stopSpeaking(): void {
  if (canSpeak) window.speechSynthesis.cancel();
}
