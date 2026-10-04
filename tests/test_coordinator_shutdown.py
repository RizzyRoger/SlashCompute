"""SIGTERM stops a busy coordinator promptly and cleanly."""

from __future__ import annotations

import asyncio
import json
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
<<<<<<< HEAD
from slashcompute.coordinator.db import Database, Job
from slashcompute.inference.coordinator.bus import CommandBus
from slashcompute.jobs import LoraFinetuneSpec
=======
from slashcompute.inference import PREFIX
from slashcompute.inference.config import InferenceSettings
from slashcompute.inference.coordinator.bus import CommandBus
from slashcompute.inference.coordinator.service import create_inference_app
>>>>>>> 7ac0ca3 (fix: require the shell's own port on write Origins and harden the command bus poll)


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


async def test_poll_cancelled_after_dequeue_keeps_the_command_in_order():
    bus = CommandBus()
    waiting = asyncio.create_task(bus.poll("n", 30))
    await asyncio.sleep(0.05)
    bus.post("n", "first", {})
    bus.post("n", "second", {})
    while bus.queues["n"].qsize() == 2:  # the poll's get() has taken "first"; the poll has not returned
        await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert [c["kind"] for c in await bus.poll("n", 1)] == ["first", "second"]


async def test_posts_to_a_dropped_node_are_not_queued_until_it_polls_again():
    bus = CommandBus()
    bus.post("n", "run_job", {})
    bus.fail_node("n", "went offline")
    bus.post("n", "cancel_job", {})
    assert "n" not in bus.queues  # no queue resurrected for a node that is gone
    assert await bus.poll("n", 0.01) == []
    bus.post("n", "noop", {})
    assert [c["kind"] for c in await bus.poll("n", 1)] == ["noop"]


async def test_command_poll_during_shutdown_tells_the_agent_to_back_off():
    app = create_inference_app(InferenceSettings(DB_PATH=":memory:", BACKGROUND_TASKS=False))
    svc = app.state.inference
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://test{PREFIX}") as c:
        r = await c.post("/nodes/register", json={"name": "a", "commitment": {},
                                                  "llama_build": svc.s.PINNED_LLAMA_BUILD})
        auth = {"authorization": f"Bearer {r.json()['token']}"}
        svc.bus.close()
        r = await c.get("/agent/commands", params={"wait": 20}, headers=auth)
        assert r.status_code == 503 and r.headers["retry-after"] == "1"


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


@pytest.mark.integration
def test_port_taken_exits_3_without_touching_state(tmp_path, tiny_model, tiny_dataset):
    spec = LoraFinetuneSpec(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=10,
                            batch_size=2, microbatches=1, lora_rank=4, min_stages=1)
    db_path = tmp_path / "home" / "coordinator" / "coordinator.db"
    db_path.parent.mkdir(parents=True)
    db = Database(db_path)
    db.save(Job(id="j1", kind=spec.kind, spec_json=json.dumps(spec.model_dump(mode="json")),
                status="running"))

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        out = subprocess.run(
            [sys.executable, "-m", "slashcompute.coordinator.main", "serve", "--host", "127.0.0.1",
             "--port", str(port), "--home", str(tmp_path / "home")],
            env={**os.environ, "SLASHCOMPUTE_LOG": "INFO"}, capture_output=True, text=True, timeout=60)

    assert out.returncode == 3
    assert "address already in use" in out.stderr.lower()
    assert "advertising" not in out.stdout + out.stderr
    with db.session() as s:
        assert s.get(Job, "j1").status == "running"
