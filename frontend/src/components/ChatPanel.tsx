import { useEffect, useRef, useState } from "react";
import { sendChat } from "../api";
import type { Analysis, ChatMessage } from "../types";

const STARTERS = [
  "How was the network trained?",
  "What does each layer do?",
  "Which actors were held out?",
  "What are the limits?",
];

const GREETING =
  "I can walk through a reading, a layer, or the training run. I answer from this model, and I do not diagnose.";

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
    if (analysis?.judge?.ask && question && asked.current !== question) {
      asked.current = question;
      setMessages((current) => [...current, { role: "assistant", text: question }]);
      setPrompts(["I have had a hard day", "My body feels wrong", "Why did you call this distress?"]);
      setOpen(true);
      return;
    }
    if (analysis && !analysis.judge?.ask) {
      setPrompts(["Explain this reading", "How was the network trained?", "What did the model get wrong?", "What are the limits?"]);
    }
  }, [analysis, question]);

  async function ask(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setOpen(true);
    setDraft("");
    const prior = messages;
    setMessages((current) => [...current, { role: "user", text: message }]);
    setBusy(true);
    try {
      const result = await sendChat(message, analysis, prior);
      setMessages((current) => [...current, { role: "assistant", text: result.reply }]);
      setPrompts(result.prompts);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", text: error instanceof Error ? error.message : "The assistant could not answer." },
      ]);
    } finally {
      setBusy(false);
    }
  }

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
            <button type="button" className="assistant-close" onClick={() => setOpen(false)} aria-label="Close assistant">
              Close
            </button>
          </header>
          <div className="chat-log" ref={scroller}>
            {messages.map((message, index) => (
              <p key={index} className={`bubble ${message.role}`}>
                {message.text}
              </p>
            ))}
            {busy ? <p className="bubble assistant wait">Listening…</p> : null}
          </div>
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
            <input
              ref={field}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={question ? "Tell me what is going on" : "Ask about this model"}
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
