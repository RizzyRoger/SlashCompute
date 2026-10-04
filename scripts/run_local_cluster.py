#!/usr/bin/env python3
"""Start a coordinator and N agents on this machine (no second Mac required).

Example::

    uv run python scripts/run_local_cluster.py --agents 2

Then in another terminal::

    uv run slashcompute-coordinator submit job.json --url http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main() -> None:
    p = argparse.ArgumentParser(description="Local /compute coordinator + agents")
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--data-port", type=int, default=9700)
    p.add_argument("--gpu-percent", type=int, default=100)
    p.add_argument("--max-memory-gb", type=float, default=4.0)
    p.add_argument("--home", type=Path, default=Path.home() / ".slashcompute" / "cluster")
    p.add_argument("--sandbox", action="store_true")
    args = p.parse_args()

    home: Path = args.home
    home.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    procs: list[subprocess.Popen] = []

    def stop(code: int | str = 0):
        # sys.exit with a message prints it to stderr and exits 1.
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in procs:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(code)

    signal.signal(signal.SIGINT, lambda *_: stop())
    signal.signal(signal.SIGTERM, lambda *_: stop())

    coord_home = home / "coordinator"
    env = os.environ.copy()
    env.setdefault("SLASHCOMPUTE_VERIFY_RATE", "0")
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "slashcompute.coordinator.main", "serve",
         "--host", "127.0.0.1", "--port", str(args.port),
         "--home", str(coord_home), "--no-mdns"],
        env=env,
    ))
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if procs[0].poll() is not None:
            raise SystemExit("coordinator exited before becoming healthy")
        time.sleep(0.2)
    else:
        stop("coordinator did not start")

    print(f"coordinator {url}  (state {coord_home})")
    for i in range(args.agents):
        agent_home = home / f"agent-{i}"
        cmd = [
            sys.executable, "-m", "slashcompute.agent.main", "start",
            "--url", url, "--localhost",
            "--data-port", str(args.data_port + i),
            "--gpu-percent", str(args.gpu_percent),
            "--max-memory-gb", str(args.max_memory_gb),
            "--home", str(agent_home),
            "--name", f"local-{i}",
        ]
        if not args.sandbox:
            cmd.append("--no-sandbox")
        procs.append(subprocess.Popen(cmd, env=env))
        print(f"agent {i} data-port {args.data_port + i}  (state {agent_home})")

    print("cluster running. Ctrl-C to stop.")
    print(f"  uv run slashcompute-coordinator nodes --url {url}")
    print(f"  uv run slashcompute-coordinator submit job.json --url {url}")
    while True:
        for proc in procs:
            if proc.poll() is not None:
                stop(f"process exited with {proc.returncode}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
