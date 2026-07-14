#!/usr/bin/env python3
"""Generate isolated TIM real-data necessity-audit multi-run configs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml


CONFIGS = (
    "1-12_nodes-weights_0.6_0.3_0.1-baselines.yaml",
    "3-12_nodes-weights_0.75_0.2_0.05-baselines.yaml",
    "5-12_nodes-weights_0.9_0.1_0-baselines.yaml",
    "7-12_nodes-weights_0.35_0.6_0.05-baselines.yaml",
)
DEFAULT_AGENTS = (
    "wait",
    "random",
    "greedy-optimal",
    "sampling-optimal",
    "exhaustive-search",
)


def _seed_parameter(config: dict) -> list[dict]:
    """Reuse the paper config's exact ten evaluation seeds."""
    try:
        return deepcopy(config["hyperparameters"]["random"])
    except KeyError as exc:
        raise ValueError("source config has no random seed schedule") from exc


def build_audit_config(source: dict, load_ratio: float, agents: Iterable[str]) -> dict:
    output = deepcopy(source)
    selected = tuple(agents)
    unknown = set(selected) - set(DEFAULT_AGENTS)
    if unknown:
        raise ValueError(f"unsupported audit agents: {sorted(unknown)}")

    existing = source["hyperparameters"]
    seed_parameter = _seed_parameter(source)
    output["hyperparameters"] = {
        agent: deepcopy(existing.get(agent, seed_parameter))
        for agent in selected
    }
    model = output["base_run_config"]["emulator"]["model"]["tim_dataset_model_options"]
    model["loads_with_respect_to_capacity"]["evaluation"] = [load_ratio]
    source_name = source.get("multi_run_name", "tim-12-nodes")
    output["multi_run_name"] = f"{source_name}-necessity-audit-load_{load_ratio:g}"
    output["skip_name_date"] = True
    return output


def generate(source_dir: Path, output_dir: Path, loads: Iterable[float], agents: Iterable[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for source_name in CONFIGS:
        source_path = source_dir / source_name
        with source_path.open(encoding="utf-8") as handle:
            source = yaml.safe_load(handle)
        for load in loads:
            audit = build_audit_config(source, load, agents)
            destination = output_dir / f"load_{load:g}-{source_name}"
            with destination.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(audit, handle, sort_keys=False)
            generated.append(str(destination))

    manifest = {
        "protocol": "tim-realdata-necessity-audit-v1",
        "source_configs": [str(source_dir / name) for name in CONFIGS],
        "loads": list(loads),
        "agents": list(agents),
        "generated_configs": generated,
        "note": "wait preserves the configured initial allocation for the full episode",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("config/tim_dataset/12-nodes"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/proactivity_ablation/generated_tim_realdata_audit"),
    )
    parser.add_argument("--loads", nargs="+", type=float, default=[0.8])
    parser.add_argument("--agents", nargs="+", choices=DEFAULT_AGENTS, default=list(DEFAULT_AGENTS))
    args = parser.parse_args()
    manifest = generate(args.source_dir, args.output_dir, args.loads, args.agents)
    print(f"Generated {len(manifest['generated_configs'])} configs in {args.output_dir}")


if __name__ == "__main__":
    main()
