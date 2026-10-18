import { useEffect, useState } from "react";
import { getHealth } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import type { Analysis } from "./types";
import { Monitor } from "./views/Monitor";
import { NetworkView } from "./views/NetworkView";
import { TrainingView } from "./views/TrainingView";

type View = "monitor" | "network" | "training";

function readView(): View {
  const hash = window.location.hash.replace("#", "");
  if (hash === "network" || hash === "training") return hash;
  return "monitor";
}

export function App() {
  const [view, setView] = useState<View>(readView);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [weights, setWeights] = useState(false);

  useEffect(() => {
    const onHash = () => setView(readView());
    window.addEventListener("hashchange", onHash);
    getHealth()
      .then((health) => setWeights(health.weights))
      .catch(() => setWeights(false));
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="shell">
      <aside className="rail">
        <a className="mark" href="#monitor">
          <span>DistressNet</span>
          <small>Deep learning studio</small>
        </a>
        <nav>
          <a href="#monitor" className={view === "monitor" ? "active" : ""}>
            Monitor
          </a>
          <a href="#network" className={view === "network" ? "active" : ""}>
            Network
          </a>
          <a href="#training" className={view === "training" ? "active" : ""}>
            Training
          </a>
        </nav>
        <p className={`weight-pill ${weights ? "on" : ""}`}>
          {weights ? "Audio branch loaded" : "Weights not loaded"}
        </p>
      </aside>

      <main>
        {view === "monitor" ? <Monitor analysis={analysis} onAnalysis={setAnalysis} /> : null}
        {view === "network" ? <NetworkView analysis={analysis} /> : null}
        {view === "training" ? <TrainingView /> : null}
        <footer>Acted speech from RAVDESS. This is a course model, not a clinical assessment.</footer>
      </main>

      <ChatPanel analysis={analysis} />
    </div>
  );
}
