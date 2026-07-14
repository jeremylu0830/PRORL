#!/usr/bin/env python3
"""Analyze paired frozen-policy movement-gating evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.proactivity_ablation.analyze_randomized_heldout import (
    event_specs,
    exact_signflip_p,
    first,
    mean_ci,
    reconstruct,
    write_csv,
)


MODES = ("baseline", "satisfied", "safe-remove", "combined")
SCENARIOS = tuple(f"heldout-{index:02d}" for index in range(4))
METRICS = (
    "utility", "remaining_gap", "surplus", "movement_cost", "gap_hours",
    "event_allocation", "event_allocation_lift", "prepeak_target_rate",
    "prepeak_any_target_rate", "proposed_add", "proposed_remove",
    "canceled_add", "canceled_remove", "executed_add", "executed_remove",
    "add_cancel_rate", "remove_cancel_rate", "active_actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path)
    return parser.parse_args()


def parse_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data["movement_gate_evaluation"]
    if metadata["learning_enabled"] or metadata["weights_changed"]:
        raise ValueError(f"Non-frozen movement-gate evaluation in {path}")
    config = data["config"]
    specs = event_specs(metadata["scenario"])
    target_nodes = sorted({event["target"] for event in specs})
    rows = []
    for evaluation_seed in metadata["evaluation_seeds"]:
        prefix = f"evaluation-{evaluation_seed}"
        adds = [int(value) for value in data[f"{prefix}/action_history/add"]]
        removes = [int(value) for value in data[f"{prefix}/action_history/remove"]]
        gaps = [float(value) for value in data[f"{prefix}/reward_history/remaining_gap/history"]]
        allocations = reconstruct(config, adds, removes)
        event_allocations, pre_rates, pre_any = [], [], []
        for event in specs:
            indices = range(event["start"], event["start"] + event["duration"])
            pre_indices = range(max(0, event["start"] - 6), event["start"])
            event_allocations.extend(allocations[index][event["target"]] for index in indices)
            pre_values = [adds[index] == event["target"] for index in pre_indices]
            pre_rates.extend(pre_values)
            pre_any.append(any(pre_values))
        background = mean(
            allocation[target] for allocation in allocations for target in target_nodes)
        counts = metadata["gate_counts"][str(evaluation_seed)]
        proposed_add = int(counts.get("proposed_add", 0))
        proposed_remove = int(counts.get("proposed_remove", 0))
        canceled_add = int(counts.get("canceled_add", 0))
        canceled_remove = int(counts.get("canceled_remove", 0))
        executed_add = int(counts.get("executed_add", 0))
        executed_remove = int(counts.get("executed_remove", 0))
        event_allocation = mean(event_allocations)
        rows.append({
            "movement_gate": metadata["movement_gate"],
            "training_seed": int(metadata["training_seed"]),
            "evaluation_seed": int(evaluation_seed),
            "scenario": metadata["scenario"]["id"],
            "utility": float(first(data[f"{prefix}/reward/utility/total"])),
            "remaining_gap": float(first(data[f"{prefix}/reward/remaining_gap/total"])),
            "surplus": float(first(data[f"{prefix}/reward/surplus/total"])),
            "movement_cost": float(first(data[f"{prefix}/reward/cost/total"])),
            "gap_hours": sum(value < 0 for value in gaps),
            "event_allocation": event_allocation,
            "event_allocation_lift": event_allocation - background,
            "prepeak_target_rate": mean(pre_rates),
            "prepeak_any_target_rate": mean(pre_any),
            "proposed_add": proposed_add,
            "proposed_remove": proposed_remove,
            "canceled_add": canceled_add,
            "canceled_remove": canceled_remove,
            "executed_add": executed_add,
            "executed_remove": executed_remove,
            "add_cancel_rate": canceled_add / proposed_add if proposed_add else 0.0,
            "remove_cancel_rate": canceled_remove / proposed_remove if proposed_remove else 0.0,
            "active_actions": executed_add + executed_remove,
            "result": str(path),
        })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["movement_gate"], row["training_seed"], row["scenario"])].append(row)
    result = []
    for (mode, training_seed, scenario), sample in sorted(grouped.items()):
        row = {"movement_gate": mode, "training_seed": training_seed, "scenario": scenario}
        row.update({metric: mean(item[metric] for item in sample) for metric in METRICS})
        result.append(row)
    return result


def validate_baseline(results: Path, reference: Path) -> int:
    """Require baseline to reproduce the earlier frozen evaluation except timing."""
    compared = 0
    for path in sorted((results / "baseline").glob("seed=*/*.json")):
        relative = path.relative_to(results / "baseline")
        old_path = reference / relative
        if not old_path.exists():
            raise ValueError(f"Missing baseline reference {old_path}")
        new = json.loads(path.read_text(encoding="utf-8"))
        old = json.loads(old_path.read_text(encoding="utf-8"))
        seeds = new["movement_gate_evaluation"]["evaluation_seeds"]
        for seed in seeds:
            prefix = f"evaluation-{seed}/"
            for key, value in old.items():
                if key.startswith(prefix) and "/times/" not in key:
                    compared += 1
                    if new.get(key) != value:
                        raise ValueError(f"Baseline regression at {relative}: {key}")
    return compared


def build_report(rows: list[dict], output: Path, baseline_fields: int | None) -> None:
    row_map = {(row["movement_gate"], row["training_seed"], row["scenario"]): row for row in rows}
    lines = [
        "# Frozen-policy movement-gating ablation",
        "",
        "The same ten randomized Forecast-No-Time best checkpoints are evaluated with no gate, an add-only "
        "satisfaction gate, a remove-only safety gate, and both gates. Each checkpoint uses four held-out "
        "schedules and five evaluation seeds. No learning occurs. Required capacity is the per-node maximum "
        "over current demand and the six-hour oracle forecast.",
        "",
    ]
    if baseline_fields is not None:
        lines.append(f"Baseline regression check passed for {baseline_fields} non-timing evaluation fields.")
        lines.append("")
    lines.extend([
        "## Absolute outcomes",
        "",
        "| Gate | Utility | Remaining gap | Surplus | Movement cost | Active sub-actions | Add cancel | Remove cancel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        sample = [row for row in rows if row["movement_gate"] == mode]
        values = {metric: mean(row[metric] for row in sample) for metric in METRICS}
        lines.append(
            f"| {mode} | {values['utility']:.3f} | {values['remaining_gap']:.3f} | "
            f"{values['surplus']:.3f} | {values['movement_cost']:.3f} | {values['active_actions']:.1f} | "
            f"{values['add_cancel_rate']:.3f} | {values['remove_cancel_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Overall paired contrasts versus baseline",
        "",
        "Differences are averaged over the four scenarios within each training seed. Confidence intervals and "
        "exact two-sided sign-flip p-values use the ten paired training seeds.",
        "",
        "| Gate | Metric | Mean difference | 95% CI | Exact p |",
        "|---|---|---:|---:|---:|",
    ])
    primary = ("utility", "remaining_gap", "surplus", "movement_cost", "event_allocation_lift", "active_actions")
    contrasts = {}
    for mode in MODES[1:]:
        for metric in primary:
            differences = []
            for seed in range(10, 20):
                gated = mean(row_map[(mode, seed, scenario)][metric] for scenario in SCENARIOS)
                baseline = mean(row_map[("baseline", seed, scenario)][metric] for scenario in SCENARIOS)
                differences.append(gated - baseline)
            center, lower, upper = mean_ci(differences)
            p_value = exact_signflip_p(differences)
            contrasts[(mode, metric)] = (center, p_value)
            lines.append(f"| {mode} | {metric} | {center:.3f} | [{lower:.3f}, {upper:.3f}] | {p_value:.4f} |")
    best = max(MODES[1:], key=lambda mode: contrasts[(mode, "utility")][0])
    lines.extend([
        "",
        "## Main finding",
        "",
        f"The largest utility improvement is produced by **{best}**: "
        f"{contrasts[(best, 'utility')][0]:.3f} versus baseline "
        f"(p={contrasts[(best, 'utility')][1]:.4f}). Its movement-cost difference is "
        f"{contrasts[(best, 'movement_cost')][0]:.3f} "
        f"(p={contrasts[(best, 'movement_cost')][1]:.4f}), while remaining-gap difference is "
        f"{contrasts[(best, 'remaining_gap')][0]:.3f} "
        f"(p={contrasts[(best, 'remaining_gap')][1]:.4f}).",
        "",
        "The separate add-only and remove-only gates identify whether redundant allocation, unsafe removal, or "
        "their interaction is responsible. This is an inference-time intervention, so it diagnoses policy/action "
        "conversion; it does not yet establish that the DQN can learn the gated behavior end to end.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = [path for mode in MODES for path in sorted((args.results / mode).glob("seed=*/*.json"))]
    expected = len(MODES) * len(SCENARIOS) * 10
    if len(files) != expected:
        raise SystemExit(f"Expected {expected} movement-gate job files, found {len(files)}")
    baseline_fields = None
    if args.baseline_reference is not None:
        baseline_fields = validate_baseline(args.results, args.baseline_reference)
    per_evaluation = [row for path in files for row in parse_file(path)]
    summary = aggregate(per_evaluation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_evaluation, args.output_dir / "per_evaluation_seed.csv")
    write_csv(summary, args.output_dir / "movement_gate_summary.csv")
    build_report(summary, args.output_dir / "movement_gate_report.md", baseline_fields)
    print(f"Analyzed {len(files)} jobs and {len(per_evaluation)} episodes")


if __name__ == "__main__":
    main()
