import type { Analysis } from "../types";

const BAND_TEXT: Record<string, string> = {
  calm: "calm",
  "check-in": "check-in",
  support: "support",
  crisis: "crisis",
};

function band(value?: string | null): string {
  return value ? BAND_TEXT[value] ?? value : "not read";
}

/**
 * The two readings side by side and the rule that combines them: the voice and the words are
 * banded separately, and the judge keeps the higher band, so neither side can cancel the other.
 * Bands only; scores stay in the studio views.
 */
export function Readings({ reading, compact = false }: { reading: Partial<Analysis>; compact?: boolean }) {
  const judge = reading.judge;
  const voice = reading.voice;
  const words = reading.transcript?.trim() ?? "";
  const voiceBand = judge?.voice_band ?? (voice?.probability != null ? voice.label : null);
  const wordsBand = judge?.words_band ?? null;
  const final = judge?.band ?? reading.band ?? null;
  const risk = judge?.risk && judge.risk !== "none" ? judge.risk : "";
  const source = reading.language?.source === "language model" ? "language model" : reading.language?.source === "phrase list" ? "phrase list" : "";

  return (
    <div className={`readings ${compact ? "compact" : ""}`}>
      <div className={`reading-side band-${voiceBand ?? "none"}`}>
        <p className="eyebrow">What the voice heard</p>
        <p className="side-band">{band(voiceBand)}</p>
        {!compact ? (
          <p className="fine">
            {voice?.probability == null
              ? "No recording, so the voice was not read."
              : voice.enrolled
                ? "Compared with your calm reading."
                : "Not calibrated to your voice yet."}
          </p>
        ) : null}
      </div>
      <div className={`reading-side band-${wordsBand ?? "none"}`}>
        <p className="eyebrow">What the words heard</p>
        <p className="side-band">{band(wordsBand)}</p>
        {!compact ? (
          <p className="fine">{words ? `“${words}”${source ? ` · ${source}` : ""}` : "No words were captured."}</p>
        ) : null}
      </div>
      <p className="max-rule">
        {risk === "high"
          ? "The crisis screen found a crisis in the words, so the bands are skipped and the assistant points to help."
          : `The judge keeps the higher band: ${band(final)}.${risk === "concern" ? " The crisis screen found possible risk, so the assistant asks about safety." : ""}`}
      </p>
    </div>
  );
}
