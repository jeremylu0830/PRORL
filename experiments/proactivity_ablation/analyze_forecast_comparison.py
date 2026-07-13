#!/usr/bin/env python3
"""Four-way fixed-schedule analysis including Forecast-No-Time PRORL."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


CONDITIONS = ("Original", "No-Time", "Calendar-Only", "Forecast-No-Time")
METRICS = (
    "utility", "remaining_gap", "surplus", "movement_cost", "gap_hours",
    "satisfied_hours", "action_substep_rate", "deadline_target_rate",
    "mean_target_allocation_at_event",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_results(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        if path.is_file() and path.name == "full_data.json":
            found.add(path)
        elif path.exists():
            found.update(path.rglob("full_data.json"))
    return sorted(found)


def first(value):
    return value[0] if isinstance(value, list) and value else value


def history(data: dict, prefix: str, name: str) -> list:
    value = data.get(f"{prefix}/{name}", [])
    if isinstance(value, dict):
        return value.get("history", [])
    if not value:
        value = data.get(f"{prefix}/{name}/history", [])
    return value


def classify(features: list[str]) -> str:
    has_time = "time-encoded" in features
    has_demand = "node-demand" in features or "node-delta" in features
    has_forecast = "node-demand-forecast" in features
    if has_forecast and not has_time:
        return "Forecast-No-Time"
    if has_time and not has_demand:
        return "Calendar-Only"
    if has_time:
        return "Original"
    return "No-Time"


def configured_events(config: dict) -> list[tuple[int, int]]:
    events = []
    couples = config["emulator"]["model"]["synthetic_model"]["couples_config"]
    for index, couple in enumerate(couples):
        schedule = couple["stress_every"]
        days = schedule["week_day"] if isinstance(schedule["week_day"], list) else [schedule["week_day"]]
        hours = schedule["hour"] if isinstance(schedule["hour"], list) else [schedule["hour"]]
        target = 2 * index + (1 if couple.get("swap_stress", False) else 0)
        # Configured hour h is action/reward history index h-1.
        events.extend((day * 24 + hour - 1, target) for day in days for hour in hours)
    return sorted(events)


def reconstruct(config: dict, adds: list[int], removes: list[int]) -> list[list[int]]:
    n_nodes = int(config["environment"]["nodes"]["n_nodes"])
    initial = int(config["environment"]["nodes"]["resource_distribution_parameters"]["initial_node_units"])
    allocation = [initial] * n_nodes
    result = []
    for add, remove in zip(adds, removes):
        if add < n_nodes:
            allocation[add] += 1
        if remove < n_nodes:
            allocation[remove] -= 1
        result.append(allocation.copy())
    return result


def summarize(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data["config"]
    features = config["environment"]["state"]["base_features"]
    condition = classify(features)
    training_seed = int(config["random_seeds"]["training"])
    evaluation_seeds = config["random_seeds"]["evaluation"]
    evaluation_seed = int(evaluation_seeds[0] if isinstance(evaluation_seeds, list) else evaluation_seeds)
    prefix = f"evaluation-{evaluation_seed}"
    n_nodes = int(config["environment"]["nodes"]["n_nodes"])
    adds = [int(value) for value in history(data, prefix, "action_history/add")]
    removes = [int(value) for value in history(data, prefix, "action_history/remove")]
    gaps = [float(value) for value in history(data, prefix, "reward_history/remaining_gap")]
    satisfied = [int(value) for value in history(data, prefix, "hour_satisfied_history")]
    allocations = reconstruct(config, adds, removes)
    events = configured_events(config)
    event_hits = [adds[index] == target for index, target in events]
    event_allocations = [allocations[index][target] for index, target in events]
    active = sum(add < n_nodes for add in adds) + sum(remove < n_nodes for remove in removes)
    row = {
        "condition": condition,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "agent_type": config["environment"]["agent"]["type"],
        "utility": float(first(data[f"{prefix}/reward/utility/total"])),
        "remaining_gap": float(first(data[f"{prefix}/reward/remaining_gap/total"])),
        "surplus": float(first(data[f"{prefix}/reward/surplus/total"])),
        "movement_cost": float(first(data[f"{prefix}/reward/cost/total"])),
        "gap_hours": sum(value < 0 for value in gaps),
        "satisfied_hours": sum(satisfied),
        "action_substep_rate": active / (2 * len(adds)),
        "deadline_target_rate": mean(event_hits),
        "mean_target_allocation_at_event": mean(event_allocations),
        "severe_failure": float(first(data[f"{prefix}/reward/remaining_gap/total"])) < -50,
        "result": str(path),
    }
    return row, {"condition": condition, "training_seed": training_seed,
                 "adds": adds, "events": events}


def exact_signflip_p(values: list[float]) -> float:
    if not values or all(value == 0 for value in values):
        return 1.0
    observed = abs(mean(values))
    permutations = (abs(mean(value * sign for value, sign in zip(values, signs)))
                    for signs in itertools.product((-1, 1), repeat=len(values)))
    return sum(value >= observed - 1e-12 for value in permutations) / (2 ** len(values))


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    if len(values) < 2:
        return center, center, center
    # Two-sided 95% t critical value for 10 paired training seeds (df=9).
    critical = 2.262 if len(values) == 10 else 1.96
    half = critical * stdev(values) / math.sqrt(len(values))
    return center, center - half, center + half


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict], output: Path) -> None:
    grouped = {condition: [row for row in rows if row["condition"] == condition]
               for condition in CONDITIONS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, metric, title in zip(
            axes, ("utility", "remaining_gap", "deadline_target_rate"),
            ("Evaluation utility", "Remaining gap", "Targeted add at stress deadline")):
        values = [[row[metric] for row in grouped[condition]] for condition in CONDITIONS]
        axis.boxplot(values, labels=CONDITIONS, showmeans=True)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_offsets(details: list[dict], output: Path) -> None:
    offsets = list(range(-6, 1))
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    for condition, marker in zip(CONDITIONS, ("o", "s", "^", "D")):
        selected = [item for item in details if item["condition"] == condition]
        rates = []
        for offset in offsets:
            hits = []
            for item in selected:
                for event_index, target in item["events"]:
                    hits.append(item["adds"][event_index + offset] == target)
            rates.append(mean(hits))
        axis.plot(offsets, rates, marker=marker, linewidth=2, label=condition)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Action offset from stress-reward index (hours)")
    axis.set_ylabel("Fraction adding to stressed node")
    axis.set_title("Targeted resource additions around fixed demand peaks")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_report(rows: list[dict], output: Path) -> None:
    grouped = {condition: [row for row in rows if row["condition"] == condition]
               for condition in CONDITIONS}
    paired = {condition: {row["training_seed"]: row for row in sample}
              for condition, sample in grouped.items()}
    seeds = sorted(set.intersection(*(set(paired[condition]) for condition in CONDITIONS)))
    lines = [
        "# Forecast-No-Time fixed-schedule comparison",
        "",
        f"Analyzed {len(rows)} completed runs, paired on training seeds {seeds}. "
        "Every saved training run contains one evaluation episode (seed 1000), so inferential tests pair "
        "training seeds rather than treating time steps as independent samples.",
        "",
        "## Aggregate results",
        "",
        "| Condition | Utility | Remaining gap | Surplus | Cost | Gap hours | Satisfied hours | Deadline hit | Target allocation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        sample = grouped[condition]
        value = lambda metric: f"{mean(row[metric] for row in sample):.3f} ± {stdev(row[metric] for row in sample):.3f}"
        lines.append(
            f"| {condition} | {value('utility')} | {value('remaining_gap')} | {value('surplus')} | "
            f"{value('movement_cost')} | {value('gap_hours')} | {value('satisfied_hours')} | "
            f"{value('deadline_target_rate')} | {value('mean_target_allocation_at_event')} |")

    lines.extend(["", "## Paired Forecast-No-Time differences", "",
                  "Positive differences favor Forecast-No-Time for utility, remaining gap, satisfied hours, "
                  "deadline hit and target allocation. Negative gap-hour differences favor Forecast-No-Time.", "",
                  "| Comparator | Metric | Mean difference | 95% CI | Exact sign-flip p |",
                  "|---|---|---:|---:|---:|"])
    comparisons = ("Original", "No-Time", "Calendar-Only")
    reported_metrics = ("utility", "remaining_gap", "movement_cost", "gap_hours",
                        "satisfied_hours", "deadline_target_rate", "mean_target_allocation_at_event")
    for comparator in comparisons:
        for metric in reported_metrics:
            differences = [paired["Forecast-No-Time"][seed][metric] - paired[comparator][seed][metric]
                           for seed in seeds]
            center, lower, upper = mean_ci(differences)
            lines.append(f"| {comparator} | {metric} | {center:.3f} | [{lower:.3f}, {upper:.3f}] | "
                         f"{exact_signflip_p(differences):.4f} |")

    lines.extend(["", "## Integrity and failure audit", ""])
    for condition in CONDITIONS:
        severe = [row["training_seed"] for row in grouped[condition] if row["severe_failure"]]
        agent_types = sorted({row["agent_type"] for row in grouped[condition]})
        lines.append(f"- {condition}: agent type {agent_types}; remaining-gap < -50 seeds: {severe}.")
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- Forecast-No-Time uses a schedule oracle. Its result is an upper bound for explicit forecast state, "
        "not evidence that an implementable predictor has reached the same performance.",
        "- The fixed-schedule result alone cannot establish temporal generalization; frozen-checkpoint shifted-peak "
        "evaluation is required.",
        "- Movement cost and active-action rates must be considered with satisfaction improvements so that continuous "
        "resource movement is not mislabeled as useful anticipation.",
        "- Configured stress hour h corresponds to action/reward history index h-1 in this environment.",
        "",
        "## Figures",
        "",
        "- `figures/fixed_schedule_metrics.png`",
        "- `figures/target_actions_by_offset.png`",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = find_results(args.results)
    rows, details = [], []
    seen = set()
    for path in paths:
        row, detail = summarize(path)
        key = (row["condition"], row["training_seed"])
        if key in seen:
            raise SystemExit(f"Duplicate result for {key}: {path}")
        seen.add(key)
        rows.append(row)
        details.append(detail)
    expected = {(condition, seed) for condition in CONDITIONS for seed in range(10, 20)}
    if seen != expected:
        raise SystemExit(f"Result set mismatch; missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    rows.sort(key=lambda row: (row["condition"], row["training_seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "fixed_schedule_summary.csv")
    plot_metrics(rows, figures / "fixed_schedule_metrics.png")
    plot_offsets(details, figures / "target_actions_by_offset.png")
    build_report(rows, args.output_dir / "fixed_schedule_report.md")
    print(f"Analyzed {len(rows)} runs; wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
