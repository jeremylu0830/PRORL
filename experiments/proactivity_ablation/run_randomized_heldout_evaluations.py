#!/usr/bin/env python3
"""Evaluate frozen randomized-training checkpoints on fixed held-out scenarios."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("ENV", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prorl.common.data_structure import RunMode  # noqa: E402
from prorl.common.encoders import NumpyEncoder  # noqa: E402
from prorl.common.filesystem import ROOT_DIR  # noqa: E402
from prorl.run.config import SingleRunConfig  # noqa: E402
from prorl.run.runner import TestRunner  # noqa: E402

from experiments.proactivity_ablation.run_forecast_inference_ablation import agent_tensors  # noqa: E402
from experiments.proactivity_ablation.run_shifted_evaluations import discover, forbid_learning  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "randomized_schedule_scenarios.json"
CONDITION_NAMES = {
    "No-Time": "Randomized-No-Time",
    "Forecast-No-Time": "Randomized-Forecast-No-Time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+", help="Randomized training result roots")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=list(range(10, 20)))
    parser.add_argument("--evaluation-seeds", nargs="+", type=int,
                        default=[1000, 1100, 1200, 1300, 1400])
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_fixed_scenario(raw: dict, scenario: dict) -> None:
    synthetic = raw["emulator"]["model"]["synthetic_model"]
    couples = synthetic["couples_config"]
    if len(couples) != len(scenario["couples"]):
        raise ValueError(f"Scenario {scenario['id']} has the wrong number of couples")
    synthetic["schedule_randomization"] = {"enabled": False, "scenarios": []}
    for couple, overrides in zip(couples, scenario["couples"]):
        couple.update(copy.deepcopy(overrides))


def evaluation_config(record: dict, scenario: dict,
                      evaluation_seeds: list[int]) -> SingleRunConfig:
    raw = copy.deepcopy(record["config"])
    apply_fixed_scenario(raw, scenario)
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
    raw["saver"]["enabled"] = True
    raw["saver"]["save_agent"] = False
    raw["saver"]["stats_condensed"] = False
    raw["logger"]["level"] = 30
    return SingleRunConfig(root_dir=ROOT_DIR, **raw)


def run_one(record: dict, scenario: dict, evaluation_seeds: list[int], output: Path) -> None:
    condition = CONDITION_NAMES[record["condition"]]
    config = evaluation_config(record, scenario, evaluation_seeds)
    runner = TestRunner(
        run_code=f"heldout-{condition.lower()}-seed{record['training_seed']}-{scenario['id']}",
        config=config,
    )
    runner._init()
    forbid_learning(runner)
    before = list(agent_tensors(runner.agent.get_agent_state()))
    started = time.time()
    runner.run()
    after = list(agent_tensors(runner.agent.get_agent_state()))
    weights_changed = len(before) != len(after) or any(
        old.shape != new.shape or not torch.equal(old, new)
        for old, new in zip(before, after)
    )
    if weights_changed:
        raise RuntimeError("Held-out evaluation changed frozen agent tensors")
    payload = runner.stats_tracker.disk_stats()
    payload["randomized_heldout_evaluation"] = {
        "condition": condition,
        "training_seed": record["training_seed"],
        "scenario": scenario,
        "checkpoint": str(record["checkpoint"]),
        "source_full_data": str(record["full_data"]),
        "evaluation_seeds": evaluation_seeds,
        "elapsed_seconds": time.time() - started,
        "learning_enabled": False,
        "weights_changed": False,
        "agent_tensor_count": len(after),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, cls=NumpyEncoder)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    scenarios = manifest["held_out"]
    records = discover(args.results, args.checkpoint)
    records = [record for record in records
               if record["condition"] in CONDITION_NAMES
               and record["training_seed"] in args.training_seeds
               and record["config"]["emulator"]["model"]["synthetic_model"]
               .get("schedule_randomization", {}).get("enabled", False)]
    expected = len(CONDITION_NAMES) * len(args.training_seeds)
    if len(records) != expected:
        found = {(record["condition"], record["training_seed"]) for record in records}
        raise SystemExit(f"Expected {expected} randomized checkpoints, found {len(records)}: {sorted(found)}")
    jobs = [(record, scenario) for record in records for scenario in scenarios]
    print(f"Held-out evaluation jobs: {len(jobs)}; episodes per job: {len(args.evaluation_seeds)}")
    for index, (record, scenario) in enumerate(jobs, start=1):
        condition = CONDITION_NAMES[record["condition"]]
        output = args.output_dir / condition / f"seed={record['training_seed']}" / f"{scenario['id']}.json"
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing {output}")
            continue
        print(f"[{index}/{len(jobs)}] {condition} seed={record['training_seed']} {scenario['id']}")
        run_one(record, scenario, args.evaluation_seeds, output)
    print(f"Completed held-out evaluation in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
