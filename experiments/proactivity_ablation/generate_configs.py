#!/usr/bin/env python3
"""Generate PRORL proactivity-ablation configs without editing the originals."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "config/sensitivity_analysis/double-dqn-full-space-longer/2-single_peak.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    return parser.parse_args()


def without_time_feature(config: dict) -> None:
    state = config["base_run_config"]["environment"]["state"]
    for key in ("base_features", "features"):
        state[key] = [feature for feature in state[key] if feature != "time-encoded"]


def add_oracle_forecast(config: dict, horizon: int = 6) -> None:
    state = config["base_run_config"]["environment"]["state"]
    for key in ("base_features", "features"):
        features = state[key]
        insert_at = features.index("node-delta") + 1
        features.insert(insert_at, "node-demand-forecast")
    state["additional_properties"]["forecast_horizon"] = horizon
    state["additional_properties"]["forecast_type"] = "oracle"


def use_constrained_reward(config: dict, sla_target_rate: float = 0.1) -> None:
    reward = config["base_run_config"]["environment"]["reward"]
    reward["type"] = "constrained-gap-surplus-cost"
    parameters = reward["parameters"]
    # A common [0, 1] scale keeps the learned multiplier comparable in train,
    # validation and evaluation environments.
    parameters["training_normalization_range"] = [0, 1]
    parameters["val_eval_normalization_range"] = [0, 1]
    parameters["objective_weights"] = [0.5, 0.5]
    parameters["sla_target_rate"] = sla_target_rate
    parameters["dual_learning_rate"] = 0.05
    parameters["dual_initial_lambda"] = 1.0
    parameters["dual_max_lambda"] = 20.0
    parameters["dual_update_interval"] = 168


def write_variant(base: dict, output: Path, name: str, with_time: bool, seeds: list[int],
                  oracle_horizon: Optional[int] = None,
                  constrained_sla_target: Optional[float] = None) -> None:
    config = copy.deepcopy(base)
    if not with_time:
        without_time_feature(config)
    if oracle_horizon is not None:
        add_oracle_forecast(config, horizon=oracle_horizon)
    if constrained_sla_target is not None:
        use_constrained_reward(config, sla_target_rate=constrained_sla_target)
    config["multi_run_name"] = f"proactivity-ablation-{name}"
    config["random_seeds"]["run"] = seeds
    output_path = output / f"{name}.yaml"
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def main() -> None:
    args = parse_args()
    with args.base.resolve().open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    args.output.mkdir(parents=True, exist_ok=True)
    write_variant(base, args.output, "original", with_time=True, seeds=args.seeds)
    write_variant(base, args.output, "no_time", with_time=False, seeds=args.seeds)
    write_variant(base, args.output, "forecast_no_time", with_time=False,
                  seeds=args.seeds, oracle_horizon=6)
    write_variant(base, args.output, "constrained_forecast_no_time", with_time=False,
                  seeds=args.seeds, oracle_horizon=6, constrained_sla_target=0.1)
    print(f"Generated configs in {args.output.resolve()}")


if __name__ == "__main__":
    main()
