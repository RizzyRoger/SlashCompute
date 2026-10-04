import asyncio
from pathlib import Path

import pytest

from slashcompute.agent.paths import resolve_data_host
from slashcompute.agent.throttle import sleep_s
from slashcompute.transport import Frame, LinkServer, TcpLink


def test_sleep_s_duty_cycle():
    assert sleep_s(2.0, 100) == 0.0
    assert sleep_s(2.0, 50) == pytest.approx(2.0)
    assert sleep_s(1.0, 25) == pytest.approx(3.0)
    assert sleep_s(1.0, 0) == 3600.0


def test_resolve_data_host_localhost():
    assert resolve_data_host(True) == "127.0.0.1"
    assert resolve_data_host(False) != ""


def test_benchmark_profile_shape():
    from slashcompute.agent.benchmark import benchmark
    from slashcompute.common.protocol import DeviceProfile

    d = benchmark(max_memory_bytes=512 * 1024 * 1024)
    assert isinstance(d, DeviceProfile)
    assert d.memory_contrib_bytes <= 512 * 1024 * 1024
    assert d.matmul_tflops > 0 and d.mem_bandwidth_gbps > 0
    assert d.chip


async def test_hello_reject_does_not_leak_expect():
    server = await LinkServer("127.0.0.1", 0, {"job_id": "secret", "epoch": 9}).start()
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    link = TcpLink(reader, writer)
    await link.send(Frame("hello", {"job_id": "nope", "epoch": 0}))
    ack = await link.recv(5)
    assert ack.kind == "hello_reject"
    assert "expect" not in ack.meta and "secret" not in str(ack.meta)
    await link.close()
    await server.close()


def test_agent_options_localhost_host(tmp_path):
    from slashcompute.agent.daemon import AgentOptions

    opt = AgentOptions(url="http://127.0.0.1:8765", home=tmp_path, localhost=True)
    assert opt.data_host == "127.0.0.1"
    assert opt.node_id == AgentOptions(url="http://127.0.0.1:8765", home=tmp_path).node_id


# ------------------------------------------------------------ coordinator outages

def _fake_profile(_max_bytes=None):
    from slashcompute.common.protocol import DeviceProfile

    return DeviceProfile(chip="fake", memory_total_bytes=16 << 30, memory_available_bytes=8 << 30,
                         working_set_bytes=12 << 30, memory_contrib_bytes=8 << 30, matmul_tflops=1.0,
                         mem_bandwidth_gbps=100.0)


async def _fake_coordinator(on_register):
    """A /ws/agent endpoint that calls ``on_register(ws, n)`` for the n-th registration."""
    import websockets

    count = {"n": 0}

    async def handler(ws):
        await ws.recv()                                    # Register
        count["n"] += 1
        await on_register(ws, count["n"])

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}", count


async def test_agent_reconnects_after_the_coordinator_restarts(tmp_path, monkeypatch):
    """A coordinator restart (close 1012) used to end the agent with a traceback; it must rejoin."""
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import Welcome, dump

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    rejoined = asyncio.Event()

    async def on_register(ws, n):
        await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=30)))
        if n == 1:
            await ws.close(code=1012, reason="service restart")
            return
        rejoined.set()
        await ws.wait_closed()

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    running = asyncio.create_task(daemon.run())
    try:
        await asyncio.wait_for(rejoined.wait(), 10)
        assert count["n"] == 2 and not running.done()
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()
    assert daemon.status == "stopped"


async def test_agent_stops_when_the_coordinator_refuses_it(tmp_path, monkeypatch):
    from slashcompute.agent.daemon import AgentOptions, Daemon

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)

    async def on_register(ws, n):
        await ws.close(code=4003, reason="banned")

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    try:
        with pytest.raises(SystemExit, match="banned"):
            await asyncio.wait_for(daemon.run(), 10)
    finally:
        server.close()
    assert count["n"] == 1                                 # no retry loop against a refusal


async def test_agent_reconnects_when_handling_a_message_raises_an_http_error(tmp_path, monkeypatch):
    """A blob upload/download failing (httpx error, not OSError) used to kill the daemon outright."""
    import httpx

    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import Drain, Welcome, dump

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    rejoined = asyncio.Event()

    async def on_register(ws, n):
        await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=30)))
        if n == 1:
            await ws.send(dump(Drain(job_id="j", epoch=1)))
        else:
            rejoined.set()
        await ws.wait_closed()

    async def boom(_msg):
        raise httpx.ConnectError("coordinator blob store unreachable")

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    monkeypatch.setattr(daemon, "_handle", boom)
    running = asyncio.create_task(daemon.run())
    try:
        await asyncio.wait_for(rejoined.wait(), 10)
        assert count["n"] == 2 and not running.done()
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()


# ------------------------------------------------------------ verification

class _HeldBundle:
    def save_bundle(self, step, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bundle")
        return True


def _verify_daemon(tmp_path):
    from slashcompute.agent.daemon import AgentOptions, Daemon

    daemon = Daemon(AgentOptions(url="http://127.0.0.1:9930", home=tmp_path, localhost=True))
    sent = []

    async def send(msg):
        sent.append(msg)

    daemon.send = send
    return daemon, sent


@pytest.mark.parametrize("status", [None, 413])
async def test_failed_bundle_upload_is_reported_not_raised(tmp_path, status):
    """The stage must answer VerifyFetch with an error instead of crashing on a failed upload."""
    import httpx

    from slashcompute.common.protocol import VerifyBundleReady, VerifyFetch

    daemon, sent = _verify_daemon(tmp_path)
    daemon._session = _HeldBundle()

    def put_bytes(path, data, params=None):
        req = httpx.Request("POST", "http://127.0.0.1:9931" + path)
        if status is None:
            raise httpx.ConnectError("connection refused", request=req)
        httpx.Response(status, request=req).raise_for_status()

    daemon.opt.http.put_bytes = put_bytes
    await daemon._on_fetch(VerifyFetch(verify_id="v1", job_id="j", epoch=1, stage_idx=0, step=3))
    [reply] = sent
    assert isinstance(reply, VerifyBundleReady) and reply.verify_id == "v1"
    assert reply.path is None and "upload failed" in reply.error


async def test_replay_does_not_block_the_event_loop(tmp_path, monkeypatch):
    """A replay longer than the heartbeat timeout used to starve heartbeats and get the verifier dropped."""
    import time

    from slashcompute.common.protocol import VerifyRequest, VerifyResult

    daemon, sent = _verify_daemon(tmp_path)
    daemon.opt.http.get_file = lambda path, dest: (dest.write_bytes(b"bundle"), dest)[1]
    daemon.opt.http.put_bytes = lambda *a, **k: None

    def slow_replay(req, bundle, dest):
        time.sleep(0.5)
        dest.write_bytes(b"out")
        return {"wall_s": 0.5}

    monkeypatch.setattr("slashcompute.agent.daemon.run_replay", slow_replay)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    t = asyncio.create_task(ticker())
    try:
        await daemon._on_verify(VerifyRequest(verify_id="v2", kind="replay", model="m",
                                              bundle_url="/verify/v2/bundle"))
    finally:
        t.cancel()
    assert ticks >= 5                                     # the loop kept running during the replay
    [reply] = sent
    assert isinstance(reply, VerifyResult) and reply.error is None and reply.stats == {"wall_s": 0.5}


def test_chosen_memory_is_lent_even_above_what_is_free_but_not_past_the_working_set(monkeypatch):
    from slashcompute.agent import benchmark as bm

    monkeypatch.setattr(bm, "_memory", lambda: (16 << 30, 5 << 30))   # 16 GB Mac, 5 GB free
    monkeypatch.setattr(bm, "_matmul_tflops", lambda: 1.0)
    monkeypatch.setattr(bm, "_mem_bandwidth_gbps", lambda: 100.0)
    assert bm.benchmark().memory_contrib_bytes == 5 << 30            # automatic: what is free
    assert bm.benchmark(10 << 30).memory_contrib_bytes == 10 << 30   # the owner's choice
    assert bm.benchmark(3 << 30).memory_contrib_bytes == 3 << 30
    assert bm.benchmark(15 << 30).memory_contrib_bytes == 12 << 30   # capped at 75% of RAM


def test_sandbox_profile_ships_inside_the_package():
    import slashcompute.agent as agent_pkg
    from slashcompute.agent.sandbox import profile_path

    profile = profile_path()
    assert profile.is_file()
    assert profile.parent == Path(agent_pkg.__file__).resolve().parent
    assert "(deny default)" in profile.read_text()


def test_wrap_command_fails_closed_without_profile(monkeypatch, tmp_path):
    from slashcompute.agent import sandbox

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(sandbox, "profile_path", lambda: tmp_path / "missing.sb")
    with pytest.raises(RuntimeError, match="unsandboxed"):
        sandbox.wrap_command(["python", "-m", "worker"], tmp_path, tmp_path)


def test_wrap_command_fails_closed_without_sandbox_exec(monkeypatch, tmp_path):
    from slashcompute.agent import sandbox

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="unsandboxed"):
        sandbox.wrap_command(["python", "-m", "worker"], tmp_path, tmp_path)


def test_wrap_command_prefixes_sandbox_exec(monkeypatch, tmp_path):
    from slashcompute.agent import sandbox

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    cmd = sandbox.wrap_command(["python", "-m", "worker"], tmp_path / "job", tmp_path)
    assert cmd[:3] == ["/usr/bin/sandbox-exec", "-f", str(sandbox.profile_path())]
    assert cmd[-4:] == ["--", "python", "-m", "worker"]


async def test_bad_dataset_reports_a_fatal_stage_error(tmp_path, tiny_model):
    from slashcompute.agent.worker import WorkerContext, run_stage
    from slashcompute.common.protocol import StageAssignment, StageFinished
    from slashcompute.jobs import LoraFinetuneSpec

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"nope": 1}\n')

    class Http:
        def get_file(self, url, dest):
            dest.write_bytes(bad.read_bytes())
            return dest

    spec = LoraFinetuneSpec(model=str(tiny_model), dataset_path=str(bad), steps=2, batch_size=2,
                            microbatches=1, lora_rank=4)
    asg = StageAssignment(job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0,
                          layer_end=6, num_layers=6, spec=spec, dataset_url="/jobs/j/dataset",
                          checkpoint_every=10, verify_ring_size=2)
    ctx = WorkerContext(assignment=asg, http=Http(), job_dir=tmp_path / "job", data_bind="127.0.0.1",
                        data_port=0, gpu_percent=100, node_id="n")
    sent = []

    async def emit(msg):
        sent.append(msg)

    with pytest.raises(ValueError):
        await run_stage(ctx, emit)
    assert len(sent) == 1 and isinstance(sent[0], StageFinished)
    assert sent[0].reason == "error" and sent[0].fatal
    assert "unrecognised dataset row keys" in sent[0].detail
