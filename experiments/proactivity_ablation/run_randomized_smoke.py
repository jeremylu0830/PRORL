#!/usr/bin/env python3
"""Bounded paired training smoke test for randomized-schedule configs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

os.environ.setdefault("ENV", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prorl import ROOT_DIR  # noqa: E402
from prorl.run.config import SingleRunConfig  # noqa: E402
from prorl.run.runner import TestRunner  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIGS = (
    HERE / "generated_randomized" / "randomized_no_time.yaml",
    HERE / "generated_randomized" / "randomized_forecast_no_time.yaml",
)
DEFAULT_RESULT = Path("/tmp/prorl_randomized_smoke_result.json")


class SmokeRunner(TestRunner):
    def _after_run(self):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs=2, default=DEFAULT_CONFIGS)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--steps", type=int, default=192)
    return parser.parse_args()


def tensors(value):
    if torch.is_tensor(value):
        yield value.detach().cpu().clone()
    elif isinstance(value, dict):
        for nested in value.values():
            yield from tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from tensors(nested)


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def smoke_config(path: Path, steps: int) -> tuple[SingleRunConfig, str]:
    multi = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    raw = copy.deepcopy(multi["base_run_config"])
    scheduled_agent_type = next(iter(multi["hyperparameters"]))
    raw["environment"]["agent"]["type"] = scheduled_agent_type
    raw["run"]["training_iterations"] = steps
    raw["run"]["validation_run"]["enabled"] = False
    raw["run"]["info_frequency"] = max(steps // 2, 1)
    raw["environment"]["agent"]["double_dqn"]["bootstrap_steps"] = 16
    raw["environment"]["agent"]["double_dqn"]["batch_size"] = 16
    raw["environment"]["agent"]["replay_buffer"]["capacity"] = 512
    raw["redis"]["enabled"] = False
    raw["saver"]["enabled"] = False
    raw["saver"]["save_agent"] = False
    raw["logger"]["level"] = 20
    return SingleRunConfig(root_dir=ROOT_DIR, **raw), scheduled_agent_type


def run_condition(path: Path, steps: int) -> dict:
    config, scheduled_agent_type = smoke_config(path, steps)
    runner = SmokeRunner(run_code=f"randomized-smoke-{path.stem}", config=config)
    runner._init()
    before = list(tensors(runner.agent.get_agent_state()))
    started = time.time()
    runner.run()
    after = list(tensors(runner.agent.get_agent_state()))
    changed = sum(
        old.shape == new.shape and not torch.equal(old, new)
        for old, new in zip(before, after)
    )
    generator = runner.env.load_generator
    if changed == 0:
        raise RuntimeError(f"{path.name}: training did not change agent tensors")
    if not generator.randomize_schedule or not generator.schedule_scenario_history:
        raise RuntimeError(f"{path.name}: randomized schedules were not sampled")
    has_forecast = "forecast" in path.stem
    if has_forecast != (runner.env.current_demand_forecast is not None):
        raise RuntimeError(f"{path.name}: forecast-state presence does not match condition")
    return {
        "condition": path.stem,
        "config": str(path.resolve()),
        "elapsed_seconds": time.time() - started,
        "scheduled_agent_type": scheduled_agent_type,
        "actual_agent_type": runner.agent.name.value,
        "changed_agent_tensors": changed,
        "agent_tensor_count": len(after),
        "scenario_history": generator.schedule_scenario_history,
        "forecast_enabled": has_forecast,
    }


def main() -> None:
    args = parse_args()
    if args.steps < 32:
        raise ValueError("smoke training requires at least 32 steps")
    report = {"status": "running", "training_steps": args.steps, "started_at": time.time()}
    write_report(args.result, report)
    try:
        conditions = [run_condition(path, args.steps) for path in args.configs]
        histories = [condition["scenario_history"] for condition in conditions]
        if histories[0] != histories[1]:
            raise RuntimeError("paired conditions sampled different scenario histories")
        report.update({"status": "passed", "conditions": conditions, "finished_at": time.time()})
    except Exception as error:
        report.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": time.time(),
        })
        write_report(args.result, report)
        raise
    write_report(args.result, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
