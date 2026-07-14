#!/usr/bin/env python3
"""Evaluate frozen randomized Forecast-No-Time checkpoints with movement gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

os.environ.setdefault("ENV", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prorl.common.encoders import NumpyEncoder  # noqa: E402
from prorl.run.runner import TestRunner  # noqa: E402

from experiments.proactivity_ablation.movement_gate import MovementGate, apply_movement_gate  # noqa: E402
from experiments.proactivity_ablation.run_forecast_inference_ablation import agent_tensors  # noqa: E402
from experiments.proactivity_ablation.run_randomized_heldout_evaluations import (  # noqa: E402
    DEFAULT_MANIFEST,
    evaluation_config,
)
from experiments.proactivity_ablation.run_shifted_evaluations import discover, forbid_learning  # noqa: E402


MODES = tuple(item.value for item in MovementGate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+", help="Randomized training result roots")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=list(range(10, 20)))
    parser.add_argument("--evaluation-seeds", nargs="+", type=int,
                        default=[1000, 1100, 1200, 1300, 1400])
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class MovementGateRunner(TestRunner):

    def __init__(self, *args, movement_gate: str, **kwargs):
        self.movement_gate = MovementGate(movement_gate)
        self.gate_counts: dict[int, Counter] = {}
        super().__init__(*args, **kwargs)

    def _return_new_env(self, *args, **kwargs):
        env = super()._return_new_env(*args, **kwargs)
        seed = int(kwargs.get("random_seed", env._initial_seed))
        counts = self.gate_counts.setdefault(seed, Counter())
        original_step = env.step

        def gated_step(actions, resource):
            decision = apply_movement_gate(actions, env, resource, self.movement_gate)
            for field in (
                    "proposed_add", "proposed_remove", "canceled_add", "canceled_remove",
                    "executed_add", "executed_remove"):
                counts[field] += int(getattr(decision, field))
            return original_step(actions, resource)

        env.step = gated_step
        return env


def run_one(record: dict, scenario: dict, mode: str,
            evaluation_seeds: list[int], output: Path) -> None:
    config = evaluation_config(record, scenario, evaluation_seeds)
    runner = MovementGateRunner(
        run_code=f"movement-gate-{mode}-seed{record['training_seed']}-{scenario['id']}",
        config=config,
        movement_gate=mode,
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
        raise RuntimeError("Movement-gate evaluation changed frozen agent tensors")
    missing = set(evaluation_seeds) - set(runner.gate_counts)
    if missing:
        raise RuntimeError(f"Missing movement-gate counters for evaluation seeds {sorted(missing)}")

    payload = runner.stats_tracker.disk_stats()
    payload["movement_gate_evaluation"] = {
        "condition": "Randomized-Forecast-No-Time",
        "training_seed": record["training_seed"],
        "movement_gate": mode,
        "scenario": scenario,
        "checkpoint": str(record["checkpoint"]),
        "source_full_data": str(record["full_data"]),
        "evaluation_seeds": evaluation_seeds,
        "gate_counts": {str(seed): dict(runner.gate_counts[seed]) for seed in evaluation_seeds},
        "required_demand": "per-node max(current, oracle forecast t+1..t+6)",
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
    records = [record for record in discover(args.results, args.checkpoint)
               if record["condition"] == "Forecast-No-Time"
               and record["training_seed"] in args.training_seeds
               and record["config"]["emulator"]["model"]["synthetic_model"]
               .get("schedule_randomization", {}).get("enabled", False)]
    if len(records) != len(args.training_seeds):
        found = {record["training_seed"] for record in records}
        raise SystemExit(f"Missing randomized Forecast-No-Time checkpoints: "
                         f"{sorted(set(args.training_seeds) - found)}")

    jobs = [(record, mode, scenario) for record in records
            for mode in args.modes for scenario in scenarios]
    print(f"Movement-gate jobs: {len(jobs)}; episodes per job: {len(args.evaluation_seeds)}")
    for index, (record, mode, scenario) in enumerate(jobs, start=1):
        output = (args.output_dir / mode / f"seed={record['training_seed']}"
                  / f"{scenario['id']}.json")
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing {output}")
            continue
        print(f"[{index}/{len(jobs)}] seed={record['training_seed']} {mode} {scenario['id']}")
        run_one(record, scenario, mode, args.evaluation_seeds, output)
    print(f"Completed movement-gate evaluation in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
