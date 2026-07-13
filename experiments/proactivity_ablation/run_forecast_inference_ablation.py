#!/usr/bin/env python3
"""Counterfactual forecast-input evaluation using frozen Forecast-No-Time checkpoints."""

from __future__ import annotations

import argparse
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

from prorl.common.encoders import NumpyEncoder  # noqa: E402
from prorl.run.runner import TestRunner  # noqa: E402

from experiments.proactivity_ablation.forecast_perturbation import (  # noqa: E402
    ForecastPerturbation,
    perturb_forecast,
)
from experiments.proactivity_ablation.run_shifted_evaluations import (  # noqa: E402
    SCHEDULES,
    discover,
    evaluation_config,
    forbid_learning,
)


MODES = tuple(mode.value for mode in ForecastPerturbation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+", help="Forecast-No-Time training result roots")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=list(range(10, 20)))
    parser.add_argument("--evaluation-seeds", nargs="+", type=int,
                        default=[1000, 1100, 1200, 1300, 1400])
    parser.add_argument("--schedules", nargs="+", choices=tuple(SCHEDULES), default=list(SCHEDULES))
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ForecastPerturbationRunner(TestRunner):

    def __init__(self, *args, forecast_mode: str, **kwargs):
        self.forecast_mode = ForecastPerturbation(forecast_mode)
        super().__init__(*args, **kwargs)

    def _return_new_env(self, *args, **kwargs):
        env = super()._return_new_env(*args, **kwargs)
        oracle_forecast = env.load_generator.forecast

        def counterfactual_forecast(current_step, horizon):
            oracle = oracle_forecast(current_step, horizon)
            return perturb_forecast(oracle, self.forecast_mode)

        env.load_generator.forecast = counterfactual_forecast
        return env


def agent_tensors(value):
    if torch.is_tensor(value):
        yield value.detach().cpu().clone()
    elif isinstance(value, dict):
        for nested in value.values():
            yield from agent_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from agent_tensors(nested)


def run_one(record: dict, mode: str, schedule: str, evaluation_seeds: list[int], output: Path) -> None:
    config = evaluation_config(record, schedule, evaluation_seeds)
    code = f"forecast-ablation-{mode}-seed{record['training_seed']}-{schedule}"
    runner = ForecastPerturbationRunner(
        run_code=code, config=config, forecast_mode=mode)
    runner._init()
    forbid_learning(runner)
    tensors_before = list(agent_tensors(runner.agent.get_agent_state()))
    started = time.time()
    runner.run()
    tensors_after = list(agent_tensors(runner.agent.get_agent_state()))
    weights_changed = len(tensors_before) != len(tensors_after) or any(
        before.shape != after.shape or not torch.equal(before, after)
        for before, after in zip(tensors_before, tensors_after))
    if weights_changed:
        raise RuntimeError("Frozen-checkpoint evaluation changed agent tensors")
    payload = runner.stats_tracker.disk_stats()
    payload["forecast_inference_ablation"] = {
        "condition": "Forecast-No-Time",
        "training_seed": record["training_seed"],
        "forecast_mode": mode,
        "schedule": schedule,
        "peaks": [list(value) for value in SCHEDULES[schedule]],
        "checkpoint": str(record["checkpoint"]),
        "source_full_data": str(record["full_data"]),
        "evaluation_seeds": evaluation_seeds,
        "elapsed_seconds": time.time() - started,
        "learning_enabled": False,
        "weights_changed": weights_changed,
        "agent_tensor_count": len(tensors_after),
        "perturbation_stage": "oracle-output-before-state-normalization",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, cls=NumpyEncoder)


def main() -> None:
    args = parse_args()
    records = [record for record in discover(args.results, args.checkpoint)
               if record["condition"] == "Forecast-No-Time"
               and record["training_seed"] in args.training_seeds]
    if len(records) != len(args.training_seeds):
        found = {record["training_seed"] for record in records}
        raise SystemExit(f"Missing Forecast-No-Time checkpoints: {sorted(set(args.training_seeds)-found)}")
    jobs = [(record, mode, schedule) for record in records
            for mode in args.modes for schedule in args.schedules]
    print(f"Inference-only jobs: {len(jobs)}; episodes per job: {len(args.evaluation_seeds)}")
    for index, (record, mode, schedule) in enumerate(jobs, start=1):
        output = args.output_dir / mode / f"seed={record['training_seed']}" / f"{schedule}.json"
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing {output}")
            continue
        print(f"[{index}/{len(jobs)}] seed={record['training_seed']} {mode} {schedule}")
        run_one(record, mode, schedule, args.evaluation_seeds, output)
    print(f"Completed forecast inference ablation in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
