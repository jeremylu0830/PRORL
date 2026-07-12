#!/usr/bin/env python3
"""Summarize anticipation around configured peaks from PRORL full_data.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+", help="Result folders or full_data.json files")
    parser.add_argument("--window", type=int, default=6, help="Hours before each peak to inspect")
    parser.add_argument("--output", type=Path, default=Path("proactivity_summary.csv"))
    return parser.parse_args()


def result_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        else:
            files.extend(path.rglob("full_data.json"))
    return sorted(set(files))


def history(data: dict, seed: int, metric: str) -> list[float]:
    value = data.get(f"evaluation-{seed}/{metric}", [])
    if isinstance(value, dict):
        return value.get("history", [])
    return value


def configured_peaks(config: dict) -> list[tuple[int, int]]:
    synthetic = config["emulator"]["model"]["synthetic_model"]
    peaks: list[tuple[int, int]] = []
    for couple_index, couple in enumerate(synthetic["couples_config"]):
        schedule = couple["stress_every"]
        days = schedule["week_day"]
        hours = schedule["hour"]
        if not isinstance(days, list):
            days = [days]
        if not isinstance(hours, list):
            hours = [hours]
        # In the synthetic model, each config controls nodes (2i, 2i+1).
        # swap_stress toggles on the first peak, making node 2i+1 the high-demand node.
        target_node = 2 * couple_index + (1 if couple.get("swap_stress", False) else 0)
        peaks.extend((day * 24 + hour, target_node) for day in days for hour in hours)
    return sorted(peaks)


def first_value(value):
    return value[0] if isinstance(value, list) and value else value


def summarize_file(path: Path, window: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    config = data["config"]
    seeds = config["random_seeds"]["evaluation"]
    if not isinstance(seeds, list):
        seeds = [seeds]
    n_nodes = config["environment"]["nodes"]["n_nodes"]
    features = config["environment"]["state"]["base_features"]
    peak_steps = configured_peaks(config)

    leads: list[int] = []
    pre_peak_moves = 0
    for seed in seeds:
        add = history(data, seed, "action_history/add")
        if not add:
            add = history(data, seed, "action_history/add_node")
        for peak, target_node in peak_steps:
            start = max(0, peak - window)
            actions = add[start:peak]
            active = [start + i for i, node in enumerate(actions) if int(node) == target_node]
            if active:
                pre_peak_moves += len(active)
                leads.append(peak - active[0])

    utility = []
    gap = []
    surplus = []
    cost = []
    for seed in seeds:
        utility.append(first_value(data.get(f"evaluation-{seed}/reward/utility/total")))
        gap.append(first_value(data.get(f"evaluation-{seed}/reward/remaining_gap/total")))
        surplus.append(first_value(data.get(f"evaluation-{seed}/reward/surplus/total")))
        cost.append(first_value(data.get(f"evaluation-{seed}/reward/cost/total")))

    def avg(values):
        clean = [float(value) for value in values if value is not None]
        return mean(clean) if clean else ""

    return {
        "result": str(path),
        "training_seed": config["random_seeds"]["training"],
        "time_encoded": "time-encoded" in features,
        "peaks": ";".join(f"{step}:n{node}" for step, node in peak_steps),
        "pre_peak_moves": pre_peak_moves,
        "mean_lead_hours": mean(leads) if leads else 0,
        "utility": avg(utility),
        "remaining_gap": avg(gap),
        "surplus": avg(surplus),
        "movement_cost": avg(cost),
    }


def main() -> None:
    args = parse_args()
    files = result_files(args.results)
    if not files:
        raise SystemExit("No full_data.json files found")
    rows = [summarize_file(path, args.window) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for label in (True, False):
        group = [row for row in rows if row["time_encoded"] is label]
        if group:
            utilities = [row["utility"] for row in group if row["utility"] != ""]
            spread = stdev(utilities) if len(utilities) > 1 else 0
            print(
                f"time_encoded={label}: n={len(group)}, utility={mean(utilities):.3f} ± {spread:.3f}, "
                f"lead={mean(row['mean_lead_hours'] for row in group):.3f} h"
            )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
