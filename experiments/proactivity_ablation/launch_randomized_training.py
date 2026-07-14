#!/usr/bin/env python3
"""Schedule and run the paired randomized experiment on a remote worker host."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import redis
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "generated_randomized"
GENERATOR = HERE / "generate_randomized_configs.py"
DEFAULT_STATUS = Path("/tmp/prorl_randomized_training_status.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=2,
                        help="Concurrent training subprocesses managed by the run worker")
    parser.add_argument("--validation-processes", type=int, default=1)
    parser.add_argument("--queue", default="proactivity-randomized-v1")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    return parser.parse_args()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def command(*parts: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / "prorl.py"), *parts]


def redis_connection() -> redis.Redis:
    load_dotenv(PROJECT_ROOT / ".env")
    return redis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        db=int(os.environ["REDIS_DB"]),
        password=os.getenv("REDIS_PASSWORD"),
    )


def main() -> None:
    args = parse_args()
    if args.processes < 1 or args.validation_processes < 1:
        raise ValueError("process counts must be positive")
    os.chdir(PROJECT_ROOT)
    status = {
        "status": "starting",
        "queue": args.queue,
        "training_processes": args.processes,
        "validation_processes": args.validation_processes,
        "started_at": time.time(),
    }
    write_status(args.status, status)
    validation_worker = None
    connection = None

    def stop_validation_worker():
        if validation_worker is not None and validation_worker.poll() is None:
            validation_worker.terminate()
            try:
                validation_worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                validation_worker.kill()
                validation_worker.wait()

    try:
        connection = redis_connection()
        connection.ping()
        queue_names = (args.queue, f"{args.queue}-validation", f"{args.queue}-failed")
        nonempty = {name: connection.llen(name) for name in queue_names if connection.llen(name)}
        if nonempty:
            raise RuntimeError(f"refusing to reuse non-empty Redis queues: {nonempty}")
        subprocess.run([sys.executable, str(GENERATOR)], check=True)
        status["status"] = "scheduling"
        write_status(args.status, status)
        subprocess.run(command(
            "run", "scheduler", "multi-runs",
            "--from-folder", str(CONFIG_DIR.relative_to(PROJECT_ROOT)),
            "-q", args.queue,
        ), check=True)
        queued = connection.llen(args.queue)
        if queued != 20:
            raise RuntimeError(f"expected 20 scheduled training runs, found {queued}")
        status["scheduled_runs"] = queued

        status["status"] = "training"
        write_status(args.status, status)
        validation_worker = subprocess.Popen(command(
            "run", "worker", "val-run-worker",
            "-p", str(args.validation_processes),
            "-q", args.queue,
        ))
        training = subprocess.run(command(
            "run", "worker", "run-worker",
            "-p", str(args.processes),
            "--stop-empty",
            "-q", args.queue,
        ))
        if training.returncode != 0:
            raise RuntimeError(f"training worker exited with code {training.returncode}")
        try:
            validation_code = validation_worker.wait(timeout=300)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("validation worker did not stop after training queue emptied") from error
        if validation_code != 0:
            raise RuntimeError(f"validation worker exited with code {validation_code}")
        failed = connection.llen(f"{args.queue}-failed")
        remaining = connection.llen(args.queue)
        if failed or remaining:
            raise RuntimeError(f"queue audit failed: remaining={remaining}, failed={failed}")
        status.update({"status": "passed", "finished_at": time.time()})
    except BaseException as error:
        stop_validation_worker()
        status.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": time.time(),
        })
        write_status(args.status, status)
        raise
    finally:
        if connection is not None:
            connection.close()
    write_status(args.status, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
