import { useCallback, useEffect, useRef, useState } from "react";
import { sendChat, sendVoiceChat } from "../api";
import type { Analysis, ChatMessage } from "../types";
import { MicRecorder, canSpeak, speak, stopSpeaking } from "../voice";
import { ChatCardView } from "./ChatCards";
import { Readings } from "./Readings";

const MAX_TURN_SECONDS = 30;

const STARTERS = [
  "I'd like to talk about how I'm feeling",
  "How was the network trained?",
  "What does each layer do?",
  "Which actors were held out?",
  "What are the limits?",
];

const GREETING =
  "Hi, I'm here if you'd like to talk about how you're feeling, and I can also walk through a reading, a layer, or the training run. I'm not a counselor and I don't diagnose.";

export function ChatPanel({ analysis }: { analysis: Analysis | null }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([{ role: "assistant", text: GREETING }]);
  const [prompts, setPrompts] = useState(STARTERS);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLInputElement>(null);
  const asked = useRef("");
  const [noticed, setNoticed] = useState("");
  // The support conversation's state (step, risk, what was shared). The server owns it; we only carry it.
  const support = useRef<Record<string, unknown> | null>(null);
  const sending = useRef(false);
  const recorder = useRef<MicRecorder | null>(null);
  const [listening, setListening] = useState(false);
  const [voiceOn, setVoiceOn] = useState(canSpeak);
  const [speaking, setSpeaking] = useState(false);
  const [micError, setMicError] = useState("");
  // Replies are read aloud once the person is talking by voice (or the chat was opened by a recording).
  const talking = useRef(false);
  const voiceOnRef = useRef(voiceOn);
  voiceOnRef.current = voiceOn;

  const say = useCallback((text: string) => {
    if (!voiceOnRef.current || !talking.current || !canSpeak) return;
    setSpeaking(true);
    void speak(text).finally(() => setSpeaking(false));
  }, []);

  const question = analysis?.judge?.ask ? analysis.judge.question : "";
  const waiting = Boolean(question && noticed !== question);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages, busy, open]);

  useEffect(() => {
    if (!open) return;
    field.current?.focus();
    if (question) setNoticed(question);
  }, [open, question]);

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  useEffect(() => {
    if (open) return;
    stopSpeaking();
    recorder.current?.release();
    recorder.current = null;
    setListening(false);
  }, [open]);

  useEffect(() => () => {
    stopSpeaking();
    recorder.current?.release();
  }, []);

  useEffect(() => {
    if (analysis?.judge?.ask && question && asked.current !== question) {
      asked.current = question;
      support.current = null;
      setMessages((current) => [...current, { role: "assistant", text: question }]);
      // Opened by a recording: answer in kind, out loud.
      if (analysis.file && analysis.file !== "words") {
        talking.current = true;
        say(question);
      }
      setPrompts(
        analysis.judge.band === "check-in"
          ? ["I'm okay, thanks", "Not great, honestly", "Why are you asking?"]
          : ["I have had a hard day", "My body feels wrong", "Why did you call this distress?"],
      );
      setOpen(true);
      return;
    }
    if (analysis && !analysis.judge?.ask) {
      support.current = null;
      setPrompts(["Explain this reading", "How was the network trained?", "What did the model get wrong?", "What are the limits?"]);
    }
  }, [analysis, question]);

  const ask = useCallback(async (text: string) => {
    const message = text.trim();
    if (!message || sending.current) return;
    sending.current = true;
    setOpen(true);
    setDraft("");
    const prior = messages;
    setMessages((current) => [...current, { role: "user", text: message }]);
    setBusy(true);
    try {
      const result = await sendChat(message, analysis, prior, support.current);
      support.current = result.state;
      setMessages((current) => [...current, { role: "assistant", text: result.reply, card: result.card }]);
      setPrompts(result.prompts);
      say(result.reply);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", text: error instanceof Error ? error.message : "The assistant could not answer." },
      ]);
    } finally {
      sending.current = false;
      setBusy(false);
    }
  }, [analysis, messages, say]);

  const sendTurn = useCallback(async (file: File) => {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    const prior = messages;
    try {
      const result = await sendVoiceChat(file, analysis, prior, support.current);
      support.current = result.state;
      setMessages((current) => [
        ...current,
        ...(result.heard ? [{ role: "user" as const, text: result.heard, spoken: true, reading: result.reading }] : []),
        { role: "assistant", text: result.reply, card: result.card },
      ]);
      if (result.prompts.length) setPrompts(result.prompts);
      say(result.reply);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", text: error instanceof Error ? error.message : "I could not hear that recording." },
      ]);
    } finally {
      sending.current = false;
      setBusy(false);
    }
  }, [analysis, messages, say]);

  const finishTurn = useCallback(() => {
    const active = recorder.current;
    recorder.current = null;
    setListening(false);
    if (!active) return;
    const file = active.stop();
    if (!file) {
      setMicError("That was too short. Tap the mic and speak for a moment.");
      return;
    }
    void sendTurn(file);
  }, [sendTurn]);

  const toggleMic = useCallback(async () => {
    if (recorder.current) {
      finishTurn();
      return;
    }
    if (busy) return;
    setMicError("");
    stopSpeaking();
    setSpeaking(false);
    const next = new MicRecorder();
    try {
      await next.start();
    } catch {
      next.release();
      setMicError("Microphone access was blocked. Allow the mic, then tap it again.");
      return;
    }
    talking.current = true;
    recorder.current = next;
    setListening(true);
  }, [busy, finishTurn]);

  useEffect(() => {
    if (!listening) return;
    const timer = window.setTimeout(finishTurn, MAX_TURN_SECONDS * 1000);
    return () => window.clearTimeout(timer);
  }, [listening, finishTurn]);

  return (
    <div className={`assistant-dock ${open ? "open" : ""}`}>
      {open ? (
        <section className="assistant-panel" role="dialog" aria-label="Assistant">
          <header className="assistant-head">
            <img src="/assistant-robot.jpg" alt="" />
            <div>
              <p className="eyebrow">Assistant</p>
              <h2>DistressNet</h2>
            </div>
            {canSpeak ? (
              <button
                type="button"
                className={`voice-toggle ${voiceOn ? "on" : ""}`}
                aria-pressed={voiceOn}
                onClick={() => {
                  if (voiceOn) {
                    stopSpeaking();
                    setSpeaking(false);
                  }
                  setVoiceOn(!voiceOn);
                }}
                title={voiceOn ? "Replies are read aloud when you talk by voice" : "Replies are text only"}
              >
                {voiceOn ? "Voice on" : "Voice off"}
              </button>
            ) : null}
            <button type="button" className="assistant-close" onClick={() => setOpen(false)} aria-label="Close assistant">
              Close
            </button>
          </header>
          <div className="chat-log" ref={scroller}>
            {messages.map((message, index) => (
              <div key={index} className={`turn ${message.role}`}>
                <p className={`bubble ${message.role}${message.spoken ? " spoken" : ""}`}>{message.text}</p>
                {message.reading?.judge ? <Readings reading={message.reading} compact /> : null}
                {message.card ? (
                  <ChatCardView card={message.card} onSend={(text) => void ask(text)} live={index === messages.length - 1} />
                ) : null}
              </div>
            ))}
            {busy ? (
              <p className="bubble assistant wait" aria-label="Typing">
                <i />
                <i />
                <i />
              </p>
            ) : null}
          </div>
          {listening || speaking || micError ? (
            <p className={`voice-status ${listening ? "listening" : ""}`} role="status">
              {listening ? "Listening… tap the mic again when you're done." : speaking ? "Speaking… tap the mic to answer." : micError}
            </p>
          ) : null}
          <div className="chips">
            {prompts.map((prompt) => (
              <button key={prompt} type="button" onClick={() => ask(prompt)} disabled={busy}>
                {prompt}
              </button>
            ))}
          </div>
          <form
            className="chat-form"
            onSubmit={(event) => {
              event.preventDefault();
              void ask(draft);
            }}
          >
            <button
              type="button"
              className={`mic ${listening ? "live" : ""}`}
              onClick={() => void toggleMic()}
              disabled={busy && !listening}
              aria-pressed={listening}
              aria-label={listening ? "Stop and send" : "Talk instead of typing"}
              title={listening ? "Stop and send" : "Talk instead of typing"}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                {listening ? (
                  <rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" />
                ) : (
                  <>
                    <rect x="9" y="3" width="6" height="11" rx="3" />
                    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
                  </>
                )}
              </svg>
            </button>
            <input
              ref={field}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={support.current?.active || question ? "Tell me what is going on" : "Say how you feel, or ask about this model"}
              aria-label="Message the assistant"
            />
            <button type="submit" disabled={busy || !draft.trim()}>
              Send
            </button>
          </form>
        </section>
      ) : null}
      <button
        type="button"
        className={`assistant-launch ${waiting ? "waiting" : ""}`}
        aria-expanded={open}
        aria-label={open ? "Close assistant" : "HI any question, open assistant"}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? null : <span className="hi-bubble">HI any question</span>}
        <span className="robot-frame">
          <img src="/assistant-robot.jpg" alt="" />
        </span>
      </button>
    </div>
  );
}
