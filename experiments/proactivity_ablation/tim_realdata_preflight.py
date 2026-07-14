#!/usr/bin/env python3
"""Check whether the original 12-node TIM audit can run reproducibly."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


REQUIRED_COLUMNS = {
    "aggregated_bs_id", "internet", "idx", "hour", "weekday", "week", "month"
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _resolve_data_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / "data" / path


def inspect(repo: Path, config_path: Path, checkpoint_dir: Optional[Path] = None) -> dict:
    checks: list[Check] = []
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    options = config["base_run_config"]["emulator"]["model"]["tim_dataset_model_options"]

    bs_ids_value = options["bs_ids"]
    paths = {
        "full_data": _resolve_data_path(repo, options["full_data_path"]),
        "index": _resolve_data_path(repo, options["index_data_path"]),
        "ran_configuration": _resolve_data_path(repo, options["ran_configurations_path"]),
        "base_station_metadata": _resolve_data_path(repo, options["bs_data_path"]),
    }
    if isinstance(bs_ids_value, str):
        paths["base_station_ids"] = _resolve_data_path(repo, bs_ids_value)
    for name, path in paths.items():
        checks.append(Check(name, "ready" if path.is_file() else "missing", str(path)))

    full_data = paths["full_data"]
    if full_data.is_file():
        with full_data.open(encoding="utf-8", newline="") as handle:
            header = set(next(csv.reader(handle)))
        missing = sorted(REQUIRED_COLUMNS - header)
        checks.append(Check(
            "full_data_schema",
            "ready" if not missing else "invalid",
            "all required columns present" if not missing else f"missing columns: {missing}",
        ))

    if isinstance(bs_ids_value, list):
        bs_ids = bs_ids_value
    else:
        bs_ids_path = paths["base_station_ids"]
        bs_ids = []
        if bs_ids_path.is_file():
            with bs_ids_path.open(encoding="utf-8") as handle:
                bs_ids = json.load(handle).get("aggregated_bs_id", [])
    if bs_ids:
        checks.append(Check(
            "node_count",
            "ready" if len(bs_ids) == 12 else "invalid",
            f"found {len(bs_ids)} node ids",
        ))

    free_bytes = shutil.disk_usage(repo).free
    # Preparing the public raw trace transfers about 20.8 GB. A streaming
    # pipeline needs less peak space, but this margin prevents a nearly-full
    # filesystem from corrupting a long preparation job.
    safe_margin = 12 * 1024 ** 3
    checks.append(Check(
        "free_disk",
        "ready" if free_bytes >= safe_margin else "warning",
        f"{free_bytes / 1024 ** 3:.1f} GiB free; 12 GiB safety margin recommended",
    ))

    checkpoint_count = 0
    if checkpoint_dir is not None:
        checkpoint_count = len(list(checkpoint_dir.rglob("*.pth"))) if checkpoint_dir.exists() else 0
        checks.append(Check(
            "prorl_checkpoints",
            "ready" if checkpoint_count else "missing",
            f"{checkpoint_count} .pth files under {checkpoint_dir}",
        ))

    blocking = [check.name for check in checks if check.status in {"missing", "invalid"}]
    return {
        "protocol": "tim-realdata-necessity-audit-v1",
        "config": str(config_path),
        "ready_for_baselines": not any(
            name != "prorl_checkpoints" for name in blocking
        ),
        "ready_for_prorl_comparison": not blocking if checkpoint_dir is not None else False,
        "blocking_checks": blocking,
        "checks": [asdict(check) for check in checks],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/tim_dataset/12-nodes/1-12_nodes-weights_0.6_0.3_0.1-baselines.yaml"),
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    report = inspect(repo, config, args.checkpoint_dir)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
