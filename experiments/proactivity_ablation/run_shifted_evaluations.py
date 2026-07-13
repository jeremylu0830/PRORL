#!/usr/bin/env python3
"""Evaluate trained PRORL checkpoints on shifted synthetic demand peaks.

This script never enters a training loop.  It uses TestRunner in evaluation
mode, loads each saved best-validation checkpoint, and writes tracker output
to one JSON file per condition/training-seed/schedule.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Importing prorl normally parses the CLI. Test mode gives this standalone
# experiment access to the project classes without consuming our arguments.
os.environ.setdefault("ENV", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prorl.common.data_structure import RunMode  # noqa: E402
from prorl.common.encoders import NumpyEncoder  # noqa: E402
from prorl.common.filesystem import ROOT_DIR  # noqa: E402
from prorl.run.config import SingleRunConfig  # noqa: E402
from prorl.run.runner import TestRunner  # noqa: E402


SCHEDULES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "control": ((3, 9), (5, 12)),
    "shift_plus_6h": ((3, 15), (5, 18)),
    "shift_plus_1d": ((4, 9), (6, 12)),
    "unseen": ((1, 17), (6, 4)),
}
CONDITIONS = ("Original", "No-Time", "Calendar-Only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+", help="Training result roots")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=list(range(10, 20)))
    parser.add_argument("--evaluation-seeds", nargs="+", type=int,
                        default=[1000, 1100, 1200, 1300, 1400])
    parser.add_argument("--schedules", nargs="+", choices=tuple(SCHEDULES), default=list(SCHEDULES))
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def classify(features: list[str]) -> str:
    has_time = "time-encoded" in features
    has_demand = "node-demand" in features or "node-delta" in features
    if has_time and not has_demand:
        return "Calendar-Only"
    if has_time:
        return "Original"
    return "No-Time"


def discover(paths: list[Path], checkpoint_choice: str) -> list[dict[str, Any]]:
    records = []
    seen: set[tuple[str, int]] = set()
    checkpoint_name = "best_validation_agent_state.pth" if checkpoint_choice == "best" else "agent_state.pth"
    for root in paths:
        for full_data in root.rglob("full_data.json"):
            data = json.loads(full_data.read_text(encoding="utf-8"))
            config = data["config"]
            condition = classify(config["environment"]["state"]["base_features"])
            seed = int(config["random_seeds"]["training"])
            key = (condition, seed)
            if key in seen:
                raise RuntimeError(f"Duplicate checkpoint for {condition} seed {seed}: {full_data}")
            checkpoint = full_data.parent / checkpoint_name
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            seen.add(key)
            records.append({
                "condition": condition,
                "training_seed": seed,
                "config": config,
                "full_data": full_data.resolve(),
                "checkpoint": checkpoint.resolve(),
            })
    return sorted(records, key=lambda item: (item["condition"], item["training_seed"]))


def set_schedule(config: dict, schedule_name: str) -> None:
    couples = config["emulator"]["model"]["synthetic_model"]["couples_config"]
    schedule = SCHEDULES[schedule_name]
    if len(couples) != len(schedule):
        raise ValueError(f"Expected {len(schedule)} demand couples, got {len(couples)}")
    for couple, (week_day, hour) in zip(couples, schedule):
        couple["stress_every"] = {"week_day": [week_day], "hour": [hour]}


def evaluation_config(record: dict[str, Any], schedule: str, evaluation_seeds: list[int]) -> SingleRunConfig:
    raw = copy.deepcopy(record["config"])
    set_schedule(raw, schedule)
    raw["run"]["run_mode"] = RunMode.Eval.value
    raw["run"]["evaluation_episode_length"] = 168
    raw["run"]["validation_run"]["enabled"] = False
    raw["random_seeds"]["evaluation"] = evaluation_seeds
    raw["environment"]["agent"]["model_load"] = {
        "load_model": True,
        "model_load_options": {
            "path": str(record["checkpoint"]),
            "mode": "disk",
            "base_path": "",
            "use_ssh_tunnel": False,
        },
    }
    raw["redis"]["enabled"] = False
    # Disk loading and saving share the same object-handler switch in PRORL.
    # Keep the handler enabled so the checkpoint can be read; TestRunner does
    # not invoke RunStatsManager, so no training result/model is saved here.
    raw["saver"]["enabled"] = True
    raw["saver"]["save_agent"] = False
    raw["saver"]["stats_condensed"] = False
    raw["logger"]["level"] = 30
    return SingleRunConfig(root_dir=ROOT_DIR, **raw)


def forbid_learning(runner: TestRunner) -> None:
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Learning was called during shifted-peak evaluation")

    runner.agent.learn = forbidden
    # Split PRORL exposes component learners too. Guard them as an additional
    # assertion that evaluation cannot mutate the networks.
    for name in ("add_node_agent", "remove_node_agent", "quantity_agent", "movement_agent"):
        component = getattr(runner.agent, name, None)
        if component is not None and hasattr(component, "learn"):
            component.learn = forbidden


def run_one(record: dict[str, Any], schedule: str, evaluation_seeds: list[int], output: Path) -> None:
    config = evaluation_config(record, schedule, evaluation_seeds)
    if config.run.run_mode != RunMode.Eval:
        raise RuntimeError(f"Refusing non-evaluation run mode: {config.run.run_mode}")
    code = f"shifted-{record['condition'].lower()}-seed{record['training_seed']}-{schedule}"
    runner = TestRunner(run_code=code, config=config)
    runner._init()
    forbid_learning(runner)
    started = time.time()
    runner.run()
    elapsed = time.time() - started
    payload = runner.stats_tracker.disk_stats()
    payload["shifted_evaluation"] = {
        "condition": record["condition"],
        "training_seed": record["training_seed"],
        "schedule": schedule,
        "peaks": [list(value) for value in SCHEDULES[schedule]],
        "checkpoint": str(record["checkpoint"]),
        "source_full_data": str(record["full_data"]),
        "evaluation_seeds": evaluation_seeds,
        "elapsed_seconds": elapsed,
        "learning_enabled": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, cls=NumpyEncoder)


def main() -> None:
    args = parse_args()
    records = discover(args.results, args.checkpoint)
    wanted = [record for record in records
              if record["condition"] in args.conditions and record["training_seed"] in args.training_seeds]
    expected = len(args.conditions) * len(args.training_seeds)
    if len(wanted) != expected:
        found = {(record["condition"], record["training_seed"]) for record in wanted}
        missing = [(condition, seed) for condition in args.conditions for seed in args.training_seeds
                   if (condition, seed) not in found]
        raise SystemExit(f"Missing requested checkpoints: {missing}")

    jobs = [(record, schedule) for record in wanted for schedule in args.schedules]
    print(f"Evaluation-only jobs: {len(jobs)}; episodes per job: {len(args.evaluation_seeds)}")
    for index, (record, schedule) in enumerate(jobs, start=1):
        output = args.output_dir / record["condition"] / f"seed={record['training_seed']}" / f"{schedule}.json"
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing {output}")
            continue
        print(f"[{index}/{len(jobs)}] {record['condition']} seed={record['training_seed']} {schedule}")
        run_one(record, schedule, args.evaluation_seeds, output)
    print(f"Completed evaluation-only jobs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
