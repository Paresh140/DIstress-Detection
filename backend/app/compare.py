"""One comparison table for the report, and a check against the proposal's targets.

Reads the reports the experiments already wrote and runs nothing new:
  artifacts/cv_report.json            python -m app.cv
  artifacts/cv_report_crema-d.json    python -m app.cv --corpus crema-d
  artifacts/cross_corpus_report.json  python -m app.study (with and without --norm)

Writes artifacts/comparison.md and prints it.

Usage, from backend/:
  python -m app.compare
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
CV_REPORTS = {"RAVDESS": "cv_report.json", "CREMA-D": "cv_report_crema-d.json"}
WITHIN_TARGET = {"RAVDESS": 0.89, "CREMA-D": 0.80}
CROSS_TARGET = 0.65
SPEC_TARGET = 0.5
ECE_TARGET = 0.10
CROSS = ("Train RAVDESS, test CREMA-D", "Train CREMA-D, test RAVDESS")


def _load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.is_file() else None


def _ms(stat: dict | None) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}" if stat else "-"


def _mark(ok: bool | None) -> str:
    return "-" if ok is None else ("met" if ok else "not met")


def within_tables(lines: list[str], checks: list[tuple[str, bool | None, str]]) -> None:
    for corpus, file in CV_REPORTS.items():
        report = _load(file)
        lines.append(f"\n## Within {corpus} (speaker cross-validation)\n")
        if not report:
            lines.append(f"`{file}` is missing; run `python -m app.cv{' --corpus crema-d' if corpus == 'CREMA-D' else ''}`.")
            checks.append((f"{corpus} WavLM balanced accuracy ≥ {WITHIN_TARGET[corpus]}", None, "not run"))
            continue
        lines.append(report["protocol"] + "\n")
        lines.append("| model | balanced acc (raw) | balanced acc (calibrated) | macro F1 | AUROC | specificity | ECE raw → calibrated | F1 > always-distress |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, result in report["models"].items():
            raw, cal = result.get("summary", {}), result.get("summary_calibrated", {})
            lines.append(
                f"| {name} | {_ms(raw.get('balanced_accuracy'))} | {_ms(cal.get('balanced_accuracy'))} | {_ms(cal.get('macro_f1'))} | "
                f"{_ms(raw.get('auroc'))} | {_ms(cal.get('specificity'))} | {_ms(raw.get('ece'))} → {_ms(cal.get('ece'))} | "
                f"{cal.get('beats_always_distress_folds', '-')} folds |"
            )
        tests = [t for t in report.get("paired_tests", []) if t["scores"] == "calibrated"]
        if tests:
            lines.append("\nPaired tests on calibrated balanced accuracy, same folds (a minus b):\n")
            lines.append("| a | b | mean difference | paired t p | Wilcoxon p |")
            lines.append("|---|---|---|---|---|")
            for t in tests:
                lines.append(f"| {t['a']} | {t['b']} | {t['mean_difference']:+.3f} | {t['t_test_p']} | {t['wilcoxon_p']} |")
            lines.append("\nWith five or six folds the smallest possible two-sided Wilcoxon p is 0.0625 or 0.031, so read it with the t-test.")
        for name, wavlm in report["models"].items():
            if not name.startswith(("wavlm", "wav2vec2")) or "summary_calibrated" not in wavlm:
                continue
            cal = wavlm["summary_calibrated"]
            bal = cal["balanced_accuracy"]["mean"]
            checks.append((f"{corpus} {name} balanced accuracy ≥ {WITHIN_TARGET[corpus]}", bal >= WITHIN_TARGET[corpus], f"{bal:.3f} calibrated, {wavlm['summary']['balanced_accuracy']['mean']:.3f} raw"))
            folds = cal.get("beats_always_distress_folds", "0/0")
            won, total = (int(x) for x in folds.split("/"))
            checks.append((f"{corpus} {name} F1 above always-distress in every fold", won == total and total > 0, folds))
            ece = cal["ece"]["mean"]
            checks.append((f"{corpus} {name} ECE ≤ {ECE_TARGET} after temperature scaling", ece <= ECE_TARGET, f"{ece:.3f} (raw {wavlm['summary']['ece']['mean']:.3f})"))


def cross_table(lines: list[str], checks: list[tuple[str, bool | None, str]]) -> None:
    report = _load("cross_corpus_report.json")
    lines.append("\n## Across corpora\n")
    if not report or "models" not in report:
        lines.append("`cross_corpus_report.json` has no models; run `python -m app.study`.")
        return
    models = report["models"]
    settings = [row["name"] for row in next(iter(models.values()))]
    lines.append("Balanced accuracy, calibrated with a validation threshold, with the speaker-bootstrap 95% interval and specificity.\n")
    lines.append("| setting | " + " | ".join(models) + " |")
    lines.append("|---|" + "---|" * len(models))
    for i, setting in enumerate(settings):
        cells = []
        for rows in models.values():
            row = rows[i] if i < len(rows) else None
            cal = row.get("calibrated") if row else None
            if not cal:
                cells.append(f"{row['test_balanced_accuracy']:.3f} raw" if row else "-")
                continue
            ci = cal.get("balanced_accuracy_ci95")
            ci_text = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            cells.append(f"{cal['balanced_accuracy']:.3f}{ci_text} (spec {cal['specificity']:.2f})")
        lines.append(f"| {setting} | " + " | ".join(cells) + " |")

    for name, wavlm in models.items():
        if not name.startswith(("wavlm", "wav2vec2")) or "calibrated" not in wavlm[0]:
            continue
        norm = "+" + name.split("+", 1)[1] if "+" in name else ""
        egemaps = models.get(f"egemaps{norm}")
        label = norm.lstrip("+") or "no normalization"
        for setting in CROSS:
            w = next((r for r in wavlm if r["name"] == setting), None)
            e = next((r for r in egemaps or [] if r["name"] == setting), None)
            if not w:
                continue
            bal, spec = w["calibrated"]["balanced_accuracy"], w["calibrated"]["specificity"]
            beats = e is not None and "calibrated" in e and bal > e["calibrated"]["balanced_accuracy"]
            ok = bal >= CROSS_TARGET and spec > SPEC_TARGET and beats
            detail = f"{bal:.3f}, spec {spec:.2f}" + (f", eGeMAPS {e['calibrated']['balanced_accuracy']:.3f}" if e and "calibrated" in e else ", no eGeMAPS run")
            checks.append((f"{setting}, {name} ({label}): ≥ {CROSS_TARGET}, spec > {SPEC_TARGET}, above eGeMAPS", ok, detail))


def main() -> str:
    lines = ["# DistressNet voice models: comparison"]
    checks: list[tuple[str, bool | None, str]] = []
    within_tables(lines, checks)
    cross_table(lines, checks)
    lines.append("\n## Proposal targets\n")
    lines.append("| target | result | value |")
    lines.append("|---|---|---|")
    for name, ok, detail in checks:
        lines.append(f"| {name} | {_mark(ok)} | {detail} |")
    lines.append(
        "\nNormalized cross-corpus runs standardize features with unlabeled audio from the test corpus or speaker; "
        "say so wherever those numbers are quoted."
    )
    text = "\n".join(lines) + "\n"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "comparison.md").write_text(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
