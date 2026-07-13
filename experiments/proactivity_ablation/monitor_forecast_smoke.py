#!/usr/bin/env python3
"""Background monitor for the tmux-based forecast smoke run."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="forecast-smoke")
    parser.add_argument("--tmux-bin", default="tmux")
    parser.add_argument("--tmux-socket")
    parser.add_argument("--result", type=Path,
                        default=Path("/tmp/prorl_forecast_smoke_result.json"))
    parser.add_argument("--monitor-result", type=Path,
                        default=Path("/tmp/prorl_forecast_smoke_monitor.json"))
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def tmux_running(tmux_bin: str, tmux_socket: Optional[str], session: str) -> bool:
    command = [tmux_bin]
    if tmux_socket:
        command += ["-L", tmux_socket]
    command += ["has-session", "-t", session]
    return subprocess.run(
        command,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    ).returncode == 0


def main() -> None:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    payload = {"status": "monitoring", "session": args.session}
    while time.monotonic() < deadline:
        if args.result.exists():
            try:
                result = json.loads(args.result.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                result = {}
            if result.get("status") in {"passed", "failed"}:
                payload = {"status": result["status"], "session": args.session,
                           "smoke_result": str(args.result)}
                break
        if not tmux_running(args.tmux_bin, args.tmux_socket, args.session):
            payload = {"status": "failed", "session": args.session,
                       "error": "tmux session exited without a final smoke result"}
            break
        time.sleep(2)
    else:
        payload = {"status": "failed", "session": args.session,
                   "error": f"monitor timed out after {args.timeout} seconds"}
    args.monitor_result.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
