"""Seed variance of the headline RAVDESS number (audit C2).

Re-runs the 6-fold RAVDESS speaker cross-validation of the binary-mean WavLM
head with several training seeds (default 7, 17, 27) on the cached features.
The folds are fixed (gender-balanced blocks of actors), so the seed changes
the head's initialization and batch order only. Reports calibrated balanced
accuracy as mean ± std over every seed and fold, per seed, and the paired
t-test against eGeMAPS for each seed (eGeMAPS folds from cv_report.json).

Usage, from backend/:
  python -m app.seed_eval
  python -m app.seed_eval --seeds 7 17 27 37 47

Writes artifacts/seed_report.json (cv_report.json is not touched).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .corpora import discover
from .cv import CORPORA, folds, run_ssl
from .speech_encoder import DEFAULT_ENCODER
from .stats import paired

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
HEAD = "binary-mean"


def _stat(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4), "min": round(float(np.min(values)), 4), "max": round(float(np.max(values)), 4)}


def run(seeds: tuple[int, ...] = (7, 17, 27), encoder: str = DEFAULT_ENCODER, head: str = HEAD) -> dict:
    spec = CORPORA["ravdess"]
    clips = discover(spec["root"], spec["name"])
    speakers = sorted({c.speaker for c in clips})
    if speakers != [f"{a:02d}" for a in range(1, 25)]:
        raise SystemExit(f"Needs RAVDESS actors 01-24 under {spec['root']}")
    plan = folds(speakers, spec["folds"], spec["val"], spec["shuffle"])
    cv_report = ARTIFACTS / spec["report"]
    egemaps = None
    if cv_report.is_file():
        saved = json.loads(cv_report.read_text()).get("models", {}).get("egemaps")
        if saved:
            egemaps = [f["calibrated"]["balanced_accuracy"] for f in saved["folds"]]
    per_seed = {}
    every = []
    for seed in seeds:
        print(f"\nseed {seed}")
        result = run_ssl(clips, plan, encoder, head, seed=seed)
        cal = [f["calibrated"]["balanced_accuracy"] for f in result["folds"]]
        raw = [f["balanced_accuracy"] for f in result["folds"]]
        every += cal
        per_seed[str(seed)] = {
            "calibrated_balanced_accuracy": _stat(cal),
            "raw_balanced_accuracy": _stat(raw),
            "folds_calibrated": cal,
            "vs_egemaps": paired(cal, egemaps) if egemaps and len(egemaps) == len(cal) else None,
        }
    seed_means = [v["calibrated_balanced_accuracy"]["mean"] for v in per_seed.values()]
    report = {
        "model": f"{encoder} + {head}",
        "protocol": "RAVDESS 6-fold speaker CV (app.cv folds), calibrated balanced accuracy; only the head's training seed changes.",
        "seeds": list(seeds),
        "all_seeds_and_folds": _stat(every),
        "spread_of_seed_means": _stat(seed_means),
        "per_seed": per_seed,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "seed_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{report['model']}: {report['all_seeds_and_folds']['mean']:.3f} ± {report['all_seeds_and_folds']['std']:.3f} over {len(every)} seed x fold runs")
    for seed, block in per_seed.items():
        vs = block["vs_egemaps"]
        print(f"  seed {seed}: {block['calibrated_balanced_accuracy']['mean']:.3f} ± {block['calibrated_balanced_accuracy']['std']:.3f}" + (f"  vs eGeMAPS diff {vs['mean_difference']:+.3f} p={vs['t_test_p']}" if vs else ""))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = parser.parse_args()
    run(tuple(args.seeds), args.encoder)
