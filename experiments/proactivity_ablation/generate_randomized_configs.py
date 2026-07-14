#!/usr/bin/env python3
"""Generate paired randomized-schedule training configs from a fixed manifest."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.proactivity_ablation.generate_configs import (
    DEFAULT_BASE,
    add_oracle_forecast,
    without_time_feature,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "randomized_schedule_scenarios.json"
DEFAULT_OUTPUT = HERE / "generated_randomized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    return parser.parse_args()


def randomized_variant(base: dict, scenarios: list[dict], condition: str,
                       seeds: list[int], with_forecast: bool) -> dict:
    config = copy.deepcopy(base)
    without_time_feature(config)
    if with_forecast:
        add_oracle_forecast(config, horizon=6)
    synthetic = config["base_run_config"]["emulator"]["model"]["synthetic_model"]
    synthetic["schedule_randomization"] = {
        "enabled": True,
        "scenarios": copy.deepcopy(scenarios),
    }
    config["multi_run_name"] = f"proactivity-randomized-{condition}"
    config["random_seeds"]["run"] = seeds
    return config


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.base.resolve().read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    training = manifest["training"]
    args.output.mkdir(parents=True, exist_ok=True)
    variants = (
        ("randomized_no_time", "no_time", False),
        ("randomized_forecast_no_time", "forecast_no_time", True),
    )
    for filename, condition, with_forecast in variants:
        config = randomized_variant(base, training, condition, args.seeds, with_forecast)
        with (args.output / f"{filename}.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Generated paired randomized configs in {args.output.resolve()}")


if __name__ == "__main__":
    main()
