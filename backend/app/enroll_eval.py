"""Calm enrollment measured on held-out speakers (audit C7).

The live studio z-scores every clip against the person's own calm reading
(app/enrollment.py). This script measures that on speakers the head never saw,
using the cached WavLM features (no encoder training, the head trains in
seconds per run on a GPU):

  none       the head as it is live today: global training standardization only
  speaker    full per-speaker standardization over all of a speaker's clips
             (every emotion, unlabeled). The upper bound from the study.
  enroll-K   per-speaker z-score over K calm/neutral clips of that speaker,
             which is what a live user can give. K = 5 by default.

For each condition the head is trained on features normalized the same way
(training speakers normalized with their own clips), the epoch, temperature and
threshold are chosen on validation speakers, and the test speakers are scored
once. Every condition is scored on the same test clips: the enrollment clips
are removed from all of them, so no clip is both a reference and a scored
example. Enrollment clips are drawn at random per speaker; --draws repeats the
draw (and the head training) with new seeds.

Settings (the same speakers as app.voice and app.study):
  RAVDESS 21-24      train RAVDESS 01-18, validate 19-20, test 21-24
  RAVDESS -> CREMA-D train RAVDESS (all but the validation actors), test every CREMA-D speaker
  CREMA-D -> RAVDESS the reverse

Reports balanced accuracy and specificity per test speaker (mean and std over
speakers), pooled metrics, and an enrollment-size sweep (--sweep 1 3 5 10).

Usage, from backend/:
  python -m app.enroll_eval                       # every setting the data allows
  python -m app.enroll_eval --settings ravdess    # only RAVDESS 21-24
  python -m app.enroll_eval --draws 3 --sweep 1 3 5 10

Writes artifacts/enrollment_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import enrollment as enroll
from .corpora import discover
from .cv import labels, score_split, split_ssl
from .dataset import TEST_ACTORS, TRAIN_ACTORS, VAL_ACTORS
from .speech_encoder import DEFAULT_ENCODER, HEADS, cache_path, embed_clips
from .study import _speaker_split
from .train import _binary_metrics

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
HEAD = "binary-mean"
SEED = 7


def _ids(actors: set[int]) -> set[str]:
    return {f"{a:02d}" for a in actors}


def settings(ravdess: list, crema: list, which: tuple[str, ...]) -> list[tuple[str, list, list, list]]:
    out = []
    if "ravdess" in which and ravdess:
        out.append(
            (
                "RAVDESS 21-24",
                [c for c in ravdess if c.speaker in _ids(TRAIN_ACTORS)],
                [c for c in ravdess if c.speaker in _ids(VAL_ACTORS)],
                [c for c in ravdess if c.speaker in _ids(TEST_ACTORS)],
            )
        )
    if ravdess and crema:
        rav, cre = _speaker_split(ravdess), _speaker_split(crema)
        if "rav2crema" in which:
            out.append(("RAVDESS -> CREMA-D", [c for c in ravdess if c.speaker not in rav["val"]], [c for c in ravdess if c.speaker in rav["val"]], crema))
        if "crema2rav" in which:
            out.append(("CREMA-D -> RAVDESS", [c for c in crema if c.speaker not in cre["val"]], [c for c in crema if c.speaker in cre["val"]], ravdess))
    return out


def speaker_norm(features: dict, clips: list) -> dict:
    """Full per-speaker standardization over all of the speaker's clips (the study's --norm speaker)."""
    groups: dict[tuple, list] = {}
    for clip in clips:
        groups.setdefault((clip.corpus, clip.speaker), []).append(str(clip.path))
    out = {}
    for paths in groups.values():
        stats = enroll.calm_stats(np.stack([features[p] for p in paths]))
        for p in paths:
            out[p] = enroll.apply(features[p], stats).astype(np.float16)
    return out


def per_speaker(test: list, prob: np.ndarray, threshold: float) -> dict:
    rows = {}
    for speaker in sorted({c.speaker for c in test}):
        idx = [i for i, c in enumerate(test) if c.speaker == speaker]
        y = labels([test[i] for i in idx])
        if y.min() == y.max():
            continue
        m = _binary_metrics(y, (prob[idx] >= threshold).astype(np.int64))
        rows[speaker] = {"balanced_accuracy": round(m["balanced_accuracy"], 4), "specificity": round(m["specificity"], 4), "clips": len(idx)}
    bal = [r["balanced_accuracy"] for r in rows.values()]
    spec = [r["specificity"] for r in rows.values()]
    return {
        "speakers": rows,
        "balanced_accuracy": {"mean": round(float(np.mean(bal)), 4), "std": round(float(np.std(bal)), 4)},
        "specificity": {"mean": round(float(np.mean(spec)), 4), "std": round(float(np.std(spec)), 4)},
    }


def run_condition(features: dict, train: list, val: list, test: list, seed: int, head: str = HEAD) -> dict:
    scored = score_split(split_ssl(train, val, test, features, head, seed=seed), test)
    cal = scored["calibrated"]
    prob = np.asarray(scored["test_probability"])
    return {
        "pooled": {
            "balanced_accuracy": round(cal["balanced_accuracy"], 4),
            "specificity": round(cal["specificity"], 4),
            "recall": round(cal["recall"], 4),
            "auroc": round(cal["auroc"], 4) if cal.get("auroc") is not None else None,
            "ece": round(cal["ece"], 4),
            "threshold": scored["threshold"],
            "temperature": scored["temperature"],
        },
        "per_speaker": per_speaker(test, prob, scored["threshold"]),
    }


def evaluate_setting(features: dict, train: list, val: list, test: list, k: int, draw: int, head: str = HEAD) -> dict:
    """none / speaker / enroll-k on the same scorable test clips, for one enrollment draw."""
    seed = SEED + 10 * draw
    chosen = enroll.pick_enrollment(train + val + test, k, seed=seed)
    tr, va, te = (enroll.scorable(group, chosen) for group in (train, val, test))
    enrolled = enroll.enrolled_features(features, train + val + test, chosen)
    by_speaker = speaker_norm(features, train + val + test)
    return {
        "draw": draw,
        "seed": seed,
        "test_clips": len(te),
        "none": run_condition(features, tr, va, te, seed, head),
        "speaker": run_condition(by_speaker, tr, va, te, seed, head),
        f"enroll-{k}": run_condition(enrolled, tr, va, te, seed, head),
    }


def _mean_std(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}


def summarize(draws: list[dict], conditions: list[str]) -> dict:
    out = {}
    for cond in conditions:
        pooled_bal = [d[cond]["pooled"]["balanced_accuracy"] for d in draws]
        pooled_spec = [d[cond]["pooled"]["specificity"] for d in draws]
        spk_bal = [v["balanced_accuracy"] for d in draws for v in d[cond]["per_speaker"]["speakers"].values()]
        spk_spec = [v["specificity"] for d in draws for v in d[cond]["per_speaker"]["speakers"].values()]
        out[cond] = {
            "pooled_balanced_accuracy": _mean_std(pooled_bal),
            "pooled_specificity": _mean_std(pooled_spec),
            "per_speaker_balanced_accuracy": _mean_std(spk_bal),
            "per_speaker_specificity": _mean_std(spk_spec),
        }
    return out


def run(features: dict, plan: list[tuple[str, list, list, list]], k: int, draws: int, sweep: tuple[int, ...], head: str = HEAD) -> dict:
    report: dict = {
        "head": head,
        "enroll_clips": k,
        "draws": draws,
        "protocol": __doc__.split("Settings")[0].strip(),
        "settings": {},
    }
    for name, train, val, test in plan:
        print(f"\n{name}: train {len(train)}  val {len(val)}  test {len(test)}")
        rows = [evaluate_setting(features, train, val, test, k, d, head) for d in range(draws)]
        conditions = ["none", "speaker", f"enroll-{k}"]
        entry = {"draws": rows, "summary": summarize(rows, conditions)}
        sweep_rows = {}
        for size in sweep:
            if size == k:
                continue
            seed = SEED
            chosen = enroll.pick_enrollment(train + val + test, size, seed=seed)
            tr, va, te = (enroll.scorable(group, chosen) for group in (train, val, test))
            enrolled = enroll.enrolled_features(features, train + val + test, chosen)
            sweep_rows[f"enroll-{size}"] = {"test_clips": len(te), **run_condition(enrolled, tr, va, te, seed, head)}
        if sweep_rows:
            entry["sweep"] = sweep_rows
        report["settings"][name] = entry
        for cond, stats in entry["summary"].items():
            print(
                f"  {cond:<10} pooled bal {stats['pooled_balanced_accuracy']['mean']:.3f}  spec {stats['pooled_specificity']['mean']:.2f}  "
                f"per speaker bal {stats['per_speaker_balanced_accuracy']['mean']:.3f} ± {stats['per_speaker_balanced_accuracy']['std']:.3f}  "
                f"spec {stats['per_speaker_specificity']['mean']:.2f} ± {stats['per_speaker_specificity']['std']:.2f}"
            )
        for cond, row in sweep_rows.items():
            print(f"  {cond:<10} pooled bal {row['pooled']['balanced_accuracy']:.3f}  spec {row['pooled']['specificity']:.2f}  (draw 0)")
    return report


def table(report: dict) -> str:
    """Markdown table: setting x condition, per-speaker mean ± std and pooled."""
    k = report["enroll_clips"]
    lines = [
        f"| setting | normalization | balanced acc per speaker | specificity per speaker | pooled balanced acc | pooled specificity |",
        "|---|---|---|---|---|---|",
    ]
    names = {"none": "none (live today)", "speaker": "full speaker standardization", f"enroll-{k}": f"calm enrollment ({k} clips)"}
    for setting, entry in report["settings"].items():
        for cond, label in names.items():
            s = entry["summary"][cond]
            lines.append(
                f"| {setting} | {label} | {s['per_speaker_balanced_accuracy']['mean']:.3f} ± {s['per_speaker_balanced_accuracy']['std']:.3f} | "
                f"{s['per_speaker_specificity']['mean']:.2f} ± {s['per_speaker_specificity']['std']:.2f} | "
                f"{s['pooled_balanced_accuracy']['mean']:.3f} | {s['pooled_specificity']['mean']:.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", nargs="+", choices=("ravdess", "rav2crema", "crema2rav"), default=["ravdess", "rav2crema", "crema2rav"])
    parser.add_argument("--k", type=int, default=enroll.ENROLL_CLIPS, help="calm clips per speaker")
    parser.add_argument("--draws", type=int, default=3, help="enrollment draws (each retrains the heads with a new seed)")
    parser.add_argument("--sweep", nargs="*", type=int, default=[1, 3, 10], help="other enrollment sizes, one draw each")
    parser.add_argument("--head", choices=sorted(HEADS), default=HEAD)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = parser.parse_args()

    ravdess = discover(ROOT / "data" / "ravdess", "RAVDESS")
    crema = discover(ROOT / "data" / "crema-d", "CREMA-D")
    if not ravdess:
        raise SystemExit("RAVDESS clips are needed under data/ravdess/")
    plan = settings(ravdess, crema, tuple(args.settings))
    clips = ravdess + crema
    paths = [str(c.path) for c in clips]
    features = dict(zip(paths, embed_clips(paths, args.encoder, cache_path(ARTIFACTS, args.encoder))))
    report = run(features, plan, args.k, args.draws, tuple(args.sweep), args.head)
    report["table"] = table(report)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "enrollment_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + report["table"])


if __name__ == "__main__":
    main()
