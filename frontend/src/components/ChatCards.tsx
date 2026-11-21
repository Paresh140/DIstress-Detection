import { useEffect, useRef, useState } from "react";
import type { ChatCard } from "../types";

// Interactive pieces the support conversation sends with a reply. Finishing one sends a
// short message back, so the conversation carries on from what the person did.
export function ChatCardView({ card, onSend, live }: { card: ChatCard; onSend: (text: string) => void; live: boolean }) {
  if (card.type === "breathing") return <BreathingCard card={card} onSend={onSend} live={live} />;
  if (card.type === "grounding") return <GroundingCard card={card} onSend={onSend} live={live} />;
  if (card.type === "mood") return <MoodCard card={card} onSend={onSend} live={live} />;
  return <ResourcesCard card={card} />;
}

type Props<T extends ChatCard["type"]> = { card: Extract<ChatCard, { type: T }>; onSend: (text: string) => void; live: boolean };

function BreathingCard({ card, onSend, live }: Props<"breathing">) {
  const [running, setRunning] = useState(false);
  const [round, setRound] = useState(0);
  const [phase, setPhase] = useState<"in" | "out">("in");
  const [count, setCount] = useState(card.inhale);
  const [done, setDone] = useState(false);
  // The chat redraws while the timer runs; keep the latest callback without restarting the count.
  const send = useRef(onSend);
  send.current = onSend;

  useEffect(() => {
    if (!running) return;
    const timer = window.setTimeout(() => {
      if (count > 1) {
        setCount(count - 1);
        return;
      }
      if (phase === "in") {
        setPhase("out");
        setCount(card.exhale);
        return;
      }
      if (round + 1 >= card.rounds) {
        setRunning(false);
        setDone(true);
        send.current("I finished the breathing exercise");
        return;
      }
      setRound(round + 1);
      setPhase("in");
      setCount(card.inhale);
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [running, count, phase, round, card]);

  function start() {
    setRound(0);
    setPhase("in");
    setCount(card.inhale);
    setDone(false);
    setRunning(true);
  }

  const seconds = phase === "in" ? card.inhale : card.exhale;
  return (
    <div className="chat-card breathing">
      <div
        className={`breath-circle ${running ? phase : ""}`}
        style={{ transitionDuration: `${seconds}s` }}
        aria-live="polite"
      >
        {running ? (
          <>
            <b>{phase === "in" ? "Breathe in" : "Breathe out"}</b>
            <span>{count}</span>
          </>
        ) : (
          <b>{done ? "Well done" : "Ready"}</b>
        )}
      </div>
      <p className="fine">
        {running
          ? `Round ${round + 1} of ${card.rounds}`
          : `In for ${card.inhale}, out for ${card.exhale}, ${card.rounds} rounds.`}
      </p>
      {running ? (
        <button type="button" onClick={() => setRunning(false)}>
          Stop
        </button>
      ) : (
        <button type="button" onClick={start} disabled={!live && !done}>
          {done ? "Go again" : "Start"}
        </button>
      )}
    </div>
  );
}

function GroundingCard({ card, onSend, live }: Props<"grounding">) {
  const [step, setStep] = useState(0);
  const finished = step >= card.steps.length;
  return (
    <div className="chat-card grounding">
      <ol>
        {card.steps.map((text, index) => (
          <li key={text} className={index < step ? "done" : index === step ? "current" : ""}>
            {text}
          </li>
        ))}
      </ol>
      {finished ? (
        <p className="fine">All five done. Take a moment before you go on.</p>
      ) : (
        <button
          type="button"
          disabled={!live}
          onClick={() => {
            const next = step + 1;
            setStep(next);
            if (next >= card.steps.length) onSend("I finished the grounding exercise");
          }}
        >
          {step === card.steps.length - 1 ? "Done" : "Next"}
        </button>
      )}
    </div>
  );
}

function MoodCard({ card, onSend, live }: Props<"mood">) {
  const [picked, setPicked] = useState<number | null>(null);
  return (
    <div className="chat-card mood">
      <p>{card.question}</p>
      <div className="mood-scale" role="group" aria-label="Mood from 0 to 10">
        {Array.from({ length: 11 }, (_, value) => (
          <button
            key={value}
            type="button"
            className={picked === value ? "picked" : ""}
            disabled={!live || picked !== null}
            onClick={() => {
              setPicked(value);
              onSend(`My mood is ${value} out of 10`);
            }}
          >
            {value}
          </button>
        ))}
      </div>
      <p className="mood-ends fine">
        <span>very bad</span>
        <span>very good</span>
      </p>
    </div>
  );
}

function ResourcesCard({ card }: { card: Extract<ChatCard, { type: "resources" }> }) {
  return (
    <div className="chat-card resources">
      <p className="eyebrow">People you can reach</p>
      <ul>
        {card.items.map((item) => (
          <li key={item.name}>
            <div>
              <b>{item.name}</b>
              <span>{item.detail}</span>
            </div>
            {item.href ? <a href={item.href}>{item.action ?? "Open"}</a> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
