#!/usr/bin/env python3
"""Analyze the paired TIM real-data baseline necessity audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


# The remote harness retained historical column labels, but the reward receives
# the tuple in [remaining-gap, surplus, movement-cost] order.
INPUT_WEIGHT_KEYS = ("weight_utility", "weight_gap", "weight_cost")
OUTPUT_WEIGHT_KEYS = ("gap_weight", "surplus_weight", "cost_weight")
METRICS = (
    "utility_total",
    "remaining_gap_total",
    "surplus_total",
    "movement_cost_total",
    "movement_count",
    "satisfied_hours",
    "sla_violation_hours",
)
AGENTS = ("wait", "random", "greedy-optimal", "sampling-optimal", "exhaustive-search")
SEEDS = tuple(range(1000, 2000, 100))
T_95_DF9 = 2.262


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = set(INPUT_WEIGHT_KEYS) | set(METRICS) | {"evaluation_seed", "episode_length", "elapsed_seconds"}
    for row in rows:
        for field in numeric:
            row[field] = float(row[field])
        row["evaluation_seed"] = int(row["evaluation_seed"])
        row["episode_length"] = int(row["episode_length"])
    return rows


def validate(rows: list[dict]) -> None:
    if len(rows) != 200:
        raise ValueError(f"expected 200 rows, found {len(rows)}")
    observed = set()
    for row in rows:
        if row["status"] != "passed" or row["episode_length"] != 168:
            raise ValueError(f"invalid job row: {row}")
        key = (*[row[field] for field in INPUT_WEIGHT_KEYS], row["requested_agent"], row["evaluation_seed"])
        if key in observed:
            raise ValueError(f"duplicate job: {key}")
        observed.add(key)
    weights = {tuple(row[field] for field in INPUT_WEIGHT_KEYS) for row in rows}
    expected = set(itertools.product(weights, AGENTS, SEEDS))
    flattened = {(tuple(key[:3]), key[3], key[4]) for key in observed}
    if flattened != expected:
        raise ValueError("weight/agent/seed coverage is incomplete")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_signflip_p(values: list[float]) -> float:
    if all(abs(value) < 1e-12 for value in values):
        return 1.0
    observed = abs(mean(values))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        permuted = abs(mean(value * sign for value, sign in zip(values, signs)))
        extreme += permuted >= observed - 1e-12
    return extreme / (2 ** len(values))


def outcome_count(rows: list[dict]) -> int:
    return len({tuple(row[metric] for metric in METRICS) for row in rows})


def analyze(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        weight = tuple(row[field] for field in INPUT_WEIGHT_KEYS)
        groups[(weight, row["requested_agent"])].append(row)

    summaries = []
    for (weight, agent), sample in sorted(groups.items()):
        output = {field: value for field, value in zip(OUTPUT_WEIGHT_KEYS, weight)}
        output.update({"agent": agent, "n_seeds": len(sample), "unique_outcomes": outcome_count(sample)})
        for metric in METRICS:
            values = [row[metric] for row in sample]
            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_sd"] = stdev(values)
        summaries.append(output)

    contrasts = []
    for weight in sorted({tuple(row[field] for field in INPUT_WEIGHT_KEYS) for row in rows}):
        control_rows = groups[(weight, "wait")]
        control = {row["evaluation_seed"]: row for row in control_rows}
        for agent in AGENTS[1:]:
            treatment_rows = groups[(weight, agent)]
            treatment = {row["evaluation_seed"]: row for row in treatment_rows}
            deterministic_repeat = outcome_count(control_rows) == outcome_count(treatment_rows) == 1
            for metric in METRICS:
                differences = [treatment[seed][metric] - control[seed][metric] for seed in SEEDS]
                center = mean(differences)
                spread = stdev(differences)
                half = T_95_DF9 * spread / math.sqrt(len(differences))
                output = {field: value for field, value in zip(OUTPUT_WEIGHT_KEYS, weight)}
                output.update({
                    "treatment": agent,
                    "control": "wait",
                    "metric": metric,
                    "mean_difference": center,
                    "ci95_lower": center - half,
                    "ci95_upper": center + half,
                    "exact_signflip_p": "" if deterministic_repeat else exact_signflip_p(differences),
                    "inference_scope": (
                        "duplicated deterministic outcome; descriptive only"
                        if deterministic_repeat
                        else "policy RNG on one fixed traffic week"
                    ),
                    "wins": sum(value > 1e-12 for value in differences),
                    "ties": sum(abs(value) <= 1e-12 for value in differences),
                    "losses": sum(value < -1e-12 for value in differences),
                })
                contrasts.append(output)
    return summaries, contrasts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("all_runs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.all_runs)
    validate(rows)
    summaries, contrasts = analyze(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "absolute_summary.csv", summaries)
    write_csv(args.output_dir / "paired_contrasts_vs_wait.csv", contrasts)
    print(f"Analyzed {len(rows)} jobs into {args.output_dir}")


if __name__ == "__main__":
    main()
