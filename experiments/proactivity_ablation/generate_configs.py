#!/usr/bin/env python3
"""Generate PRORL proactivity-ablation configs without editing the originals."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

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


def write_variant(base: dict, output: Path, name: str, with_time: bool, seeds: list[int]) -> None:
    config = copy.deepcopy(base)
    if not with_time:
        without_time_feature(config)
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
    print(f"Generated configs in {args.output.resolve()}")


if __name__ == "__main__":
    main()
