"""SIGTERM stops a busy coordinator promptly and cleanly."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from slashcompute.common import protocol as P
from slashcompute.inference.coordinator.bus import CommandBus


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait(cond, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if cond():
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError("condition not met")


async def test_close_releases_long_polls_without_eating_commands():
    bus = CommandBus()
    assert await bus.poll("n", 0.05) == []  # timed out: its get() must not linger
    bus.post("n", "noop", {})
    assert [c["kind"] for c in await bus.poll("n", 1)] == ["noop"]

    waiting = asyncio.create_task(bus.poll("n", 30))
    await asyncio.sleep(0.05)
    bus.close()
    assert await asyncio.wait_for(waiting, 1) == []


@pytest.mark.integration
def test_sigterm_exits_promptly_with_nodes_connected(tmp_path):
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "SLASHCOMPUTE_LOG": "INFO"}
    procs, logs = [], []

    def spawn(name, args):
        fh = (tmp_path / f"{name}.log").open("w")
        logs.append(fh)
        procs.append(subprocess.Popen([sys.executable, "-m", *args], env=env,
                                      stdout=fh, stderr=subprocess.STDOUT))
        return procs[-1]

    try:
        coord = spawn("coord", ["slashcompute.coordinator.main", "serve", "--host", "127.0.0.1",
                                "--port", str(port), "--home", str(tmp_path / "coord"), "--no-mdns"])
        _wait(lambda: httpx.get(f"{url}/health", timeout=1).json()["ok"])
        (tmp_path / "models").mkdir()
        spawn("node", ["slashcompute.inference.node", "start", "--url", url, "--localhost", "--fake",
                       "--memory-gb", "2", "--models-dir", str(tmp_path / "models"),
                       "--home", str(tmp_path / "node")])
        _wait(lambda: httpx.get(f"{url}/health", timeout=1).json()["inference_nodes"] == 1)

        mem = 8 * 1024**3
        with connect(f"ws://127.0.0.1:{port}/ws/agent") as ws:
            ws.send(P.dump(P.Register(
                node_id="t-1", name="t-1", data_host="127.0.0.1", data_port=1, gpu_percent=100,
                device=P.DeviceProfile(chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
                                       working_set_bytes=mem, memory_contrib_bytes=mem,
                                       matmul_tflops=1.0, mem_bandwidth_gbps=100.0))))
            assert isinstance(P.parse_coordinator_message(ws.recv(timeout=10)), P.Welcome)
            time.sleep(1.0)  # the node is now parked in its `/agent/commands` long-poll

            started = time.monotonic()
            coord.send_signal(signal.SIGTERM)
            with pytest.raises(ConnectionClosed):
                while True:
                    ws.recv(timeout=5)
            assert ws.protocol.close_rcvd is not None and ws.protocol.close_rcvd.code in (1001, 1012)
            coord.wait(timeout=5)
            assert time.monotonic() - started < 3.0
        assert coord.returncode in (0, -signal.SIGTERM)
        assert "Traceback" not in (tmp_path / "coord.log").read_text()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
        for fh in logs:
            fh.close()
