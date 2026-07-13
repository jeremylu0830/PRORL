#!/usr/bin/env python3
"""Analyze whether PRORL actions anticipate synthetic demand peaks.

This analysis deliberately aligns configured (one-based) simulation hours with
the zero-based action/reward histories.  It also treats the split add/remove
actions separately and reconstructs node allocations from the configured
initial allocation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


METRICS = ("utility", "remaining_gap", "surplus", "cost")
CONDITIONS = ("Original", "No-Time", "Calendar-Only")


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


def configured_peaks(config: dict) -> list[tuple[int, int]]:
    """Return (simulation hour, stressed node) for the first stress event."""
    synthetic = config["emulator"]["model"]["synthetic_model"]
    peaks: list[tuple[int, int]] = []
    for couple_index, couple in enumerate(synthetic["couples_config"]):
        schedule = couple["stress_every"]
        days = schedule["week_day"]
        hours = schedule["hour"]
        days = days if isinstance(days, list) else [days]
        hours = hours if isinstance(hours, list) else [hours]
        # swapped starts False and toggles at the first stress event.
        target = 2 * couple_index + (1 if couple.get("swap_stress", False) else 0)
        peaks.extend((day * 24 + hour, target) for day in days for hour in hours)
    return sorted(peaks)


def reconstruct_allocations(config: dict, adds: list[int], removes: list[int]) -> list[list[int]]:
    n_nodes = config["environment"]["nodes"]["n_nodes"]
    initial = config["environment"]["nodes"]["resource_distribution_parameters"]["initial_node_units"]
    allocation = [initial] * n_nodes
    result: list[list[int]] = []
    for add, remove in zip(adds, removes):
        # For split actions, n_nodes is the wait/pool-side action index.
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
    has_time = "time-encoded" in features
    has_demand_signal = "node-demand" in features or "node-delta" in features
    if has_time and not has_demand_signal:
        condition = "Calendar-Only"
    elif has_time:
        condition = "Original"
    else:
        condition = "No-Time"
    seed = int(config["random_seeds"]["training"])
    evaluation_seeds = config["random_seeds"]["evaluation"]
    evaluation_seed = evaluation_seeds[0] if isinstance(evaluation_seeds, list) else evaluation_seeds
    prefix = f"evaluation-{evaluation_seed}"
    n_nodes = int(config["environment"]["nodes"]["n_nodes"])

    adds = [int(value) for value in history(data, prefix, "action_history/add")]
    removes = [int(value) for value in history(data, prefix, "action_history/remove")]
    gaps = [float(value) for value in history(data, prefix, "reward_history/remaining_gap")]
    utilities = [float(value) for value in history(data, prefix, "reward_history/utility")]
    satisfied = [int(value) for value in history(data, prefix, "hour_satisfied_history")]
    allocations = reconstruct_allocations(config, adds, removes)
    peaks = configured_peaks(config)

    # An action at history index h-1 is evaluated against simulation hour h:
    # EnvWrapper applies the action, advances the state, then computes reward.
    event_indices = [(hour - 1, target) for hour, target in peaks]
    deadline_hits = [int(adds[index] == target) for index, target in event_indices]
    event_allocations = [allocations[index][target] for index, target in event_indices]
    gap_hours = sum(value < 0 for value in gaps)
    severe = float(first(data.get(f"{prefix}/reward/remaining_gap/total"))) < -50

    totals = {
        metric: float(first(data.get(f"{prefix}/reward/{metric}/total"))) for metric in METRICS
    }
    active_subactions = sum(add < n_nodes for add in adds) + sum(remove < n_nodes for remove in removes)
    action_rate = active_subactions / (2 * len(adds)) if adds else math.nan

    row = {
        "condition": condition,
        "training_seed": seed,
        "evaluation_seed": evaluation_seed,
        "utility": totals["utility"],
        "remaining_gap": totals["remaining_gap"],
        "surplus": totals["surplus"],
        "movement_cost": totals["cost"],
        "gap_hours": gap_hours,
        "satisfied_hours": sum(satisfied),
        "action_substep_rate": action_rate,
        "deadline_target_hits": sum(deadline_hits),
        "deadline_target_rate": mean(deadline_hits),
        "mean_target_allocation_at_event": mean(event_allocations),
        "severe_failure": severe,
        "result": str(path),
    }
    detail = {
        "condition": condition,
        "seed": seed,
        "adds": adds,
        "removes": removes,
        "gaps": gaps,
        "utilities": utilities,
        "satisfied": satisfied,
        "allocations": allocations,
        "events": event_indices,
        "peaks": peaks,
        "total_gap": totals["remaining_gap"],
    }
    return row, detail


def exact_signflip_p(differences: list[float]) -> float:
    observed = abs(mean(differences))
    values = []
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        values.append(abs(mean(value * sign for value, sign in zip(differences, signs))))
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def write_summary(rows: list[dict], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_seed_metrics(rows: list[dict], output: Path) -> None:
    by_condition = {condition: {row["training_seed"]: row for row in rows if row["condition"] == condition}
                    for condition in CONDITIONS}
    seeds = sorted(set.intersection(*(set(by_condition[condition]) for condition in CONDITIONS)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, metric, label in zip(axes, ("utility", "remaining_gap"), ("Total utility", "Total remaining gap")):
        for condition, marker in zip(CONDITIONS, ("o", "s", "^")):
            axis.plot(seeds, [by_condition[condition][s][metric] for s in seeds],
                      marker=marker, label=condition)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Training seed")
    axes[-1].set_xticks(seeds)
    fig.suptitle("Paired evaluation outcomes")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_deadline_actions(details: list[dict], output: Path) -> None:
    offsets = list(range(-6, 1))
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for condition, marker in zip(CONDITIONS, ("o", "s", "^")):
        selected = [item for item in details if item["condition"] == condition]
        rates = []
        for offset in offsets:
            hits = []
            for item in selected:
                for event_index, target in item["events"]:
                    index = event_index + offset
                    hits.append(item["adds"][index] == target)
            rates.append(mean(hits))
        axis.plot(offsets, rates, marker=marker, linewidth=2, label=condition)
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Action offset from stress-reward index (hours)")
    axis.set_ylabel("Fraction adding to stressed node")
    axis.set_title("Targeted add actions around configured stress events")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_timelines(details: list[dict], output: Path) -> None:
    selected = []
    for condition in CONDITIONS:
        group = [item for item in details if item["condition"] == condition]
        if group:
            selected.append(max(group, key=lambda item: item["total_gap"]))
            selected.append(min(group, key=lambda item: item["total_gap"]))
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 2.6 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for axis, item in zip(axes, selected):
        x = list(range(len(item["gaps"])))
        axis.plot(x, item["gaps"], color="tab:red", linewidth=1.3, label="remaining gap")
        for event_index, target in item["events"]:
            axis.axvline(event_index, color="black", linestyle="--", linewidth=1)
            axis.scatter(event_index, item["gaps"][event_index], s=55,
                         color="tab:green" if item["adds"][event_index] == target else "tab:orange",
                         zorder=3)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set_ylabel("Gap")
        axis.set_title(f"{item['condition']} seed {item['seed']}")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("History index (stress at configured hour h appears at h-1)")
    fig.suptitle("Successful and severe-failure timelines; event dot green = targeted add")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_report(rows: list[dict], details: list[dict], output: Path) -> None:
    grouped = {condition: [row for row in rows if row["condition"] == condition]
               for condition in CONDITIONS}
    paired = {condition: {row["training_seed"]: row for row in values}
              for condition, values in grouped.items()}
    seeds = sorted(set.intersection(*(set(paired[condition]) for condition in CONDITIONS)))
    utility_diff = [paired["Original"][seed]["utility"] - paired["No-Time"][seed]["utility"] for seed in seeds]
    deadline_diff = [paired["Original"][seed]["deadline_target_rate"]
                     - paired["No-Time"][seed]["deadline_target_rate"] for seed in seeds]
    allocation_diff = [paired["Original"][seed]["mean_target_allocation_at_event"]
                       - paired["No-Time"][seed]["mean_target_allocation_at_event"] for seed in seeds]
    calendar_utility_vs_original = [paired["Calendar-Only"][seed]["utility"]
                                    - paired["Original"][seed]["utility"] for seed in seeds]
    calendar_utility_vs_no_time = [paired["Calendar-Only"][seed]["utility"]
                                   - paired["No-Time"][seed]["utility"] for seed in seeds]
    calendar_deadline_vs_no_time = [paired["Calendar-Only"][seed]["deadline_target_rate"]
                                    - paired["No-Time"][seed]["deadline_target_rate"] for seed in seeds]
    calendar_allocation_vs_no_time = [paired["Calendar-Only"][seed]["mean_target_allocation_at_event"]
                                      - paired["No-Time"][seed]["mean_target_allocation_at_event"]
                                      for seed in seeds]
    utility_ci_half = 2.262 * stdev(utility_diff) / math.sqrt(len(utility_diff))

    lines = [
        "# PRORL proactivity ablation: dynamics analysis",
        "",
        f"Analyzed {len(rows)} runs ({len(grouped['Original'])} Original, {len(grouped['No-Time'])} No-Time, "
        f"{len(grouped['Calendar-Only'])} Calendar-Only), "
        f"paired on {len(seeds)} training seeds. All runs use evaluation seed 1000.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Original | No-Time | Calendar-Only |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in (("utility", "Utility"), ("remaining_gap", "Remaining gap"),
                          ("gap_hours", "Hours with negative gap"),
                          ("action_substep_rate", "Active add/remove subaction rate"),
                          ("deadline_target_rate", "Correct targeted add at stress deadline"),
                          ("mean_target_allocation_at_event", "Target allocation at stress event")):
        values = []
        for condition in CONDITIONS:
            sample = [row[metric] for row in grouped[condition]]
            values.append(f"{mean(sample):.3f} ± {stdev(sample):.3f}")
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    severe_original = [row["training_seed"] for row in grouped["Original"] if row["severe_failure"]]
    severe_no_time = [row["training_seed"] for row in grouped["No-Time"] if row["severe_failure"]]
    severe_calendar = [row["training_seed"] for row in grouped["Calendar-Only"] if row["severe_failure"]]
    lines.extend([
        "",
        "## Paired utility result",
        "",
        f"Original − No-Time mean difference: {mean(utility_diff):.3f}; "
        f"95% CI [{mean(utility_diff)-utility_ci_half:.3f}, {mean(utility_diff)+utility_ci_half:.3f}]; "
        f"exact paired sign-flip p={exact_signflip_p(utility_diff):.4f}.",
        "",
        "## Failure modes",
        "",
        f"- Severe Original runs (remaining gap < -50): {severe_original}.",
        f"- Severe No-Time runs (remaining gap < -50): {severe_no_time}.",
        f"- Severe Calendar-Only runs (remaining gap < -50): {severe_calendar}.",
        "- Severe runs have negative gap across most of the week, not only at the two stress events. "
        "They therefore indicate a learned allocation-cycle failure, rather than merely missing a peak.",
        "- All three agents activate almost every add/remove substep. A random targeted add inside a six-hour "
        "window is therefore not valid evidence of anticipation.",
        "",
        "## Timing interpretation",
        "",
        "`EnvWrapper.step()` applies the selected action, advances to the next demand state, and then computes "
        "the reward. Consequently, configured simulation hour h is represented by action/reward history index h-1. "
        "The earlier lead-time analysis used h directly and counted any targeted add in a six-hour window, so its "
        "reported 4–5 hour lead should not be interpreted as forecast horizon.",
        "",
        "At the correctly aligned stress deadline, Original adds to the stressed node more often than No-Time. "
        f"The paired difference is {mean(deadline_diff):.3f} (exact sign-flip p="
        f"{exact_signflip_p(deadline_diff):.4f}); its stressed-node allocation is higher by "
        f"{mean(allocation_diff):.3f} units (p={exact_signflip_p(allocation_diff):.4f}). "
        "This supports the narrower claim that calendar input affects action timing. It does not establish robust "
        "proactivity: mean utility is unchanged, policies move resources almost continuously, and the fixed episode "
        "start/allocation state can itself act as a clock for a feed-forward policy.",
        "",
        "Calendar-Only isolates calendar memorization: it retains `time-encoded` while removing both "
        "`node-demand` and `node-delta`. Its aggregate results show whether a memorized fixed schedule is "
        "sufficient without observing demand; interpretation should still be restricted to the single fixed "
        "evaluation schedule.",
        "",
        "## Calendar-Only paired results",
        "",
        f"Calendar-Only − Original utility: {mean(calendar_utility_vs_original):.3f} "
        f"(p={exact_signflip_p(calendar_utility_vs_original):.4f}). Calendar-Only − No-Time utility: "
        f"{mean(calendar_utility_vs_no_time):.3f} (p={exact_signflip_p(calendar_utility_vs_no_time):.4f}). "
        "Neither utility comparison is statistically distinguishable with ten seeds.",
        "",
        f"Against No-Time, Calendar-Only improves correctly targeted deadline actions by "
        f"{mean(calendar_deadline_vs_no_time):.3f} (p={exact_signflip_p(calendar_deadline_vs_no_time):.4f}) "
        f"and target allocation by {mean(calendar_allocation_vs_no_time):.3f} units "
        f"(p={exact_signflip_p(calendar_allocation_vs_no_time):.4f}). Calendar-Only and Original have "
        "nearly identical deadline targeting. This is evidence of fixed-schedule memorization, not evidence "
        "that calendar-only control generalizes to shifted or unseen peaks.",
        "",
        "## Figures",
        "",
        "- `figures/paired_metrics.png`",
        "- `figures/deadline_target_actions.png`",
        "- `figures/failure_timelines.png`",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = find_results(args.results)
    if not paths:
        raise SystemExit("No full_data.json files found")
    rows: list[dict] = []
    details: list[dict] = []
    for path in paths:
        row, detail = summarize(path)
        rows.append(row)
        details.append(detail)
    rows.sort(key=lambda row: (row["condition"], row["training_seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_summary(rows, args.output_dir / "dynamics_summary.csv")
    plot_seed_metrics(rows, figures / "paired_metrics.png")
    plot_deadline_actions(details, figures / "deadline_target_actions.png")
    plot_timelines(details, figures / "failure_timelines.png")
    build_report(rows, details, args.output_dir / "dynamics_report.md")
    print(f"Analyzed {len(rows)} runs")
    print(f"Wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
