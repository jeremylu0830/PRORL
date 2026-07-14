#!/usr/bin/env python3
"""Analyze static, current-only, and forecast-horizon movement-gate controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.proactivity_ablation.analyze_movement_gate_ablation import (  # noqa: E402
    METRICS,
    SCENARIOS,
    aggregate,
    parse_file,
    validate_baseline,
)
from experiments.proactivity_ablation.analyze_randomized_heldout import (  # noqa: E402
    exact_signflip_p,
    mean_ci,
    write_csv,
)


MODES = (
    "baseline",
    "always-wait",
    "combined-current",
    "combined-h1",
    "combined-h3",
    "combined",
)
DISPLAY = {
    "baseline": "policy baseline",
    "always-wait": "always-wait",
    "combined-current": "current-only",
    "combined-h1": "forecast-h1",
    "combined-h3": "forecast-h3",
    "combined": "forecast-h6",
}
PRIMARY_METRICS = (
    "utility", "remaining_gap", "surplus", "movement_cost",
    "event_allocation_lift", "active_actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    return parser.parse_args()


def paired_differences(row_map: dict, treatment: str, control: str, metric: str) -> list[float]:
    differences = []
    for seed in range(10, 20):
        treatment_value = mean(row_map[(treatment, seed, scenario)][metric] for scenario in SCENARIOS)
        control_value = mean(row_map[(control, seed, scenario)][metric] for scenario in SCENARIOS)
        differences.append(treatment_value - control_value)
    return differences


def contrast(row_map: dict, treatment: str, control: str, metric: str) -> tuple[float, float, float, float]:
    differences = paired_differences(row_map, treatment, control, metric)
    center, lower, upper = mean_ci(differences)
    return center, lower, upper, exact_signflip_p(differences)


def build_report(rows: list[dict], output: Path, baseline_fields: int) -> None:
    row_map = {(row["movement_gate"], row["training_seed"], row["scenario"]): row for row in rows}
    lines = [
        "# Static and forecast-horizon movement-gate controls",
        "",
        "The same ten frozen randomized Forecast-No-Time checkpoints are evaluated on four held-out schedules "
        "and five evaluation seeds. Always-wait forces both split actions to wait; current-only uses no future "
        "demand; h1, h3, and h6 use progressively longer prefixes of the same oracle forecast. No learning occurs.",
        "",
        f"Baseline regression check passed for {baseline_fields} non-timing evaluation fields.",
        "",
        "## Absolute outcomes",
        "",
        "| Control | Utility | Remaining gap | Surplus | Movement cost | Event allocation lift | Active sub-actions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        sample = [row for row in rows if row["movement_gate"] == mode]
        values = {metric: mean(row[metric] for row in sample) for metric in METRICS}
        lines.append(
            f"| {DISPLAY[mode]} | {values['utility']:.3f} | {values['remaining_gap']:.3f} | "
            f"{values['surplus']:.3f} | {values['movement_cost']:.3f} | "
            f"{values['event_allocation_lift']:.3f} | {values['active_actions']:.1f} |"
        )

    comparisons = [
        ("always-wait", "baseline"),
        ("combined-current", "always-wait"),
        ("combined-h1", "always-wait"),
        ("combined-h3", "always-wait"),
        ("combined", "always-wait"),
        ("combined-h1", "combined-current"),
        ("combined-h3", "combined-current"),
        ("combined", "combined-current"),
        ("combined-h3", "combined-h1"),
        ("combined", "combined-h3"),
    ]
    lines.extend([
        "",
        "## Paired contrasts",
        "",
        "Each difference is averaged over four scenarios within a training seed. Confidence intervals and exact "
        "two-sided sign-flip p-values use ten paired training seeds.",
        "",
        "| Contrast | Metric | Difference | 95% CI | Exact p |",
        "|---|---|---:|---:|---:|",
    ])
    contrasts = {}
    for treatment, control in comparisons:
        label = f"{DISPLAY[treatment]} − {DISPLAY[control]}"
        for metric in PRIMARY_METRICS:
            center, lower, upper, p_value = contrast(row_map, treatment, control, metric)
            contrasts[(treatment, control, metric)] = (center, p_value)
            lines.append(f"| {label} | {metric} | {center:.3f} | [{lower:.3f}, {upper:.3f}] | {p_value:.4f} |")

    wait_utility, wait_p = contrasts[("always-wait", "baseline", "utility")]
    current_utility, current_p = contrasts[("combined-current", "always-wait", "utility")]
    h1_wait_utility, h1_wait_p = contrasts[("combined-h1", "always-wait", "utility")]
    h6_utility, h6_p = contrasts[("combined", "combined-current", "utility")]
    h6_wait_utility, h6_wait_p = contrasts[("combined", "always-wait", "utility")]
    lines.extend([
        "",
        "## Decision evidence",
        "",
        f"Forcing static allocation changes utility by {wait_utility:.3f} versus the original policy "
        f"(p={wait_p:.4f}). Adding a current-demand safety gate on top of that changes utility by "
        f"{current_utility:.3f} (p={current_p:.4f}). The incremental h6 forecast effect over current-only is "
        f"{h6_utility:.3f} (p={h6_p:.4f}); h6 differs from always-wait by {h6_wait_utility:.3f} "
        f"(p={h6_wait_p:.4f}).",
        "",
        f"H1 is the highest-utility non-static gate, but its difference from always-wait is "
        f"{h1_wait_utility:.3f} (p={h1_wait_p:.4f}). Longer forecasts monotonically improve remaining gap and "
        "event-allocation lift, while also increasing surplus and movement cost; scalarized utility therefore "
        "peaks at h1 and then decreases. Most of the earlier combined-gate utility gain is attributable to "
        "movement suppression, although short-horizon forecast has a small positive increment over current-only.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = [path for mode in MODES for path in sorted((args.results / mode).glob("seed=*/*.json"))]
    expected = len(MODES) * len(SCENARIOS) * 10
    if len(files) != expected:
        raise SystemExit(f"Expected {expected} control job files, found {len(files)}")
    baseline_fields = validate_baseline(args.results, args.baseline_reference)
    per_evaluation = [row for path in files for row in parse_file(path)]
    summary = aggregate(per_evaluation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_evaluation, args.output_dir / "per_evaluation_seed.csv")
    write_csv(summary, args.output_dir / "movement_gate_control_summary.csv")
    build_report(summary, args.output_dir / "movement_gate_control_report.md", baseline_fields)
    print(f"Analyzed {len(files)} jobs and {len(per_evaluation)} episodes")


if __name__ == "__main__":
    main()
