#!/usr/bin/env python3
"""Run a short, self-contained Forecast-No-Time training smoke test."""

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
DEFAULT_CONFIG = HERE / "generated" / "forecast_no_time.yaml"
DEFAULT_RESULT = Path("/tmp/prorl_forecast_smoke_result.json")


class SmokeRunner(TestRunner):
    """Avoid the full post-training evaluation; this test only checks training."""

    def _after_run(self):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--steps", type=int, default=96)
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
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps < 32:
        raise ValueError("smoke training requires at least 32 steps")
    with args.config.resolve().open(encoding="utf-8") as stream:
        raw = copy.deepcopy(yaml.safe_load(stream)["base_run_config"])

    raw["run"]["training_iterations"] = args.steps
    raw["run"]["validation_run"]["enabled"] = False
    raw["run"]["info_frequency"] = max(args.steps // 2, 1)
    raw["environment"]["agent"]["double_dqn"]["bootstrap_steps"] = 16
    raw["environment"]["agent"]["double_dqn"]["batch_size"] = 16
    raw["environment"]["agent"]["replay_buffer"]["capacity"] = 256
    raw["redis"]["enabled"] = False
    raw["saver"]["enabled"] = False
    raw["saver"]["save_agent"] = False
    raw["logger"]["level"] = 20

    report = {
        "status": "running",
        "config": str(args.config.resolve()),
        "training_steps": args.steps,
        "bootstrap_steps": 16,
        "batch_size": 16,
        "started_at": time.time(),
    }
    write_report(args.result, report)
    try:
        config = SingleRunConfig(root_dir=ROOT_DIR, **raw)
        runner = SmokeRunner(run_code="forecast-no-time-smoke", config=config)
        runner._init()
        before = list(tensors(runner.agent.get_agent_state()))
        started = time.time()
        runner.run()
        elapsed = time.time() - started
        after = list(tensors(runner.agent.get_agent_state()))
        changed = sum(
            1 for old, new in zip(before, after)
            if old.shape == new.shape and not torch.equal(old, new)
        )
        forecast_size = runner.env.n_nodes * 6 * len(runner.env.resources_info)
        if changed == 0:
            raise RuntimeError("training completed but no agent tensor changed")
        if runner.env.current_demand_forecast is None:
            raise RuntimeError("training completed without producing forecasts")
        report.update({
            "status": "passed",
            "elapsed_seconds": elapsed,
            "forecast_horizon": len(runner.env.current_demand_forecast),
            "forecast_feature_size": forecast_size,
            "changed_agent_tensors": changed,
            "agent_tensor_count": len(after),
            "state_spaces": {key.value: value for key, value in runner.env.state_spaces.items()},
            "finished_at": time.time(),
        })
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
