import asyncio
import json

import httpx
import pytest

from inf_harness import FakeNode, chat, fast_settings, start_harness
from slashcompute.inference.coordinator.layers import synthetic_layout
from slashcompute.inference.coordinator.pipelines import TRANSITIONS, IllegalTransition, check_transition
from slashcompute.inference.node.config import Commitment

GB = 10 ** 9


QWEN = "Qwen3.8-27B-UD-Q4_K_XL.gguf"


def qwen_layout():
    return synthetic_layout(QWEN, n_layers=64, total_bytes=int(17.56 * GB), head_bytes=1 * GB)


def pipelines(h):
    return h.conn.execute("SELECT * FROM pipelines ORDER BY created_at").fetchall()


def members(h, pid):
    return h.conn.execute("SELECT * FROM pipeline_members WHERE pipeline_id=? ORDER BY position", (pid,)).fetchall()


# ------------------------------------------------------------ state machine


def test_transition_table():
    check_transition("planned", "starting")
    check_transition("loading", "active")
    check_transition("active", "broken")
    for old, new in [("active", "loading"), ("stopped", "active"), ("broken", "active"), ("planned", "active")]:
        with pytest.raises(IllegalTransition):
            check_transition(old, new)
    assert TRANSITIONS["stopped"] == set() and TRANSITIONS["broken"] == set()


# ------------------------------------------------------------ formation, reuse, teardown, failure


@pytest.fixture
async def two_node():
    h = await start_harness(fast_settings(PIPELINE_IDLE_SECONDS=30))
    h.add_model(qwen_layout())
    await h.add_nodes([FakeNode("head", 16, gen_score=1.0, may_be_head=True, files=(QWEN,)),
                       FakeNode("worker", 16, gen_score=1.2)])
    yield h
    await h.stop()


async def test_formation_split_and_reuse(two_node):
    h = two_node
    r1 = await chat(h, QWEN, max_tokens=16)
    assert r1.status_code == 200, r1.text
    r2 = await chat(h, QWEN, content="again", max_tokens=16)
    assert r2.status_code == 200
    assert r1.json()["network"]["pipeline_id"] == r2.json()["network"]["pipeline_id"]  # reused
    (p,) = pipelines(h)
    assert p["state"] == "active"
    ms = members(h, p["id"])
    assert [m["role"] for m in ms] == ["worker", "head"]
    assert ms[0]["endpoint"] and ms[0]["endpoint"].startswith("100.64.0.")
    assert sum(float(m["share"]) for m in ms) == pytest.approx(1.0)
    assert json.loads(p["tensor_split"])[-1] == ms[-1]["layer_end"] - ms[-1]["layer_start"] + 1
    # response carries the per-member split and their FLOP credit
    net = r1.json()["network"]
    assert {m["node"] for m in net["members"]} == {"head", "worker"}
    assert net["flops"] == pytest.approx(sum(m["flops"] for m in net["members"]))
    assert all(m["flops"] > 0 for m in net["members"])
    # every finished request is handed to the accounting hook, keyed by node
    rec = h.svc.accounting.records[-1]
    assert set(rec["per_node"]) == {h.ids["head"], h.ids["worker"]}


async def test_streaming_response(two_node):
    h = two_node
    r = await chat(h, QWEN, max_tokens=16, stream=True)
    assert r.status_code == 200
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    summary = json.loads(lines[-2][6:])
    assert summary["network"]["predicted_n"] == 16
    assert summary["network"]["gen_weight"] >= 1.0


async def test_idle_pipeline_is_torn_down():
    h = await start_harness(fast_settings(PIPELINE_IDLE_SECONDS=0.5))
    try:
        h.add_model(qwen_layout())
        await h.add_nodes([FakeNode("head", 16, may_be_head=True, files=(QWEN,)), FakeNode("worker", 16)])
        assert (await chat(h, QWEN)).status_code == 200
        for _ in range(40):
            await asyncio.sleep(0.1)
            if pipelines(h)[0]["state"] == "stopped":
                break
        assert pipelines(h)[0]["state"] == "stopped"
        assert h.agents["worker"].in_use == {} and h.agents["head"].in_use == {}
    finally:
        await h.stop()


async def test_node_failure_breaks_pipeline_replans_and_retries():
    h = await start_harness(fast_settings(), time_scale=1.0)
    try:
        h.add_model(qwen_layout())
        await h.add_nodes([FakeNode("head", 16, gen_score=1.0, may_be_head=True, files=(QWEN,)),
                           FakeNode("fast", 16, gen_score=2.0), FakeNode("spare", 16, gen_score=0.5)])
        assert (await chat(h, QWEN, max_tokens=4)).status_code == 200
        first = pipelines(h)[0]
        assert {m["node_id"] for m in members(h, first["id"])} == {h.ids["head"], h.ids["fast"]}

        task = asyncio.create_task(chat(h, QWEN, max_tokens=40, content="long one"))
        await asyncio.sleep(0.5)
        await h.drop("fast")
        r = await task
        assert r.status_code == 200, r.text
        net = r.json()["network"]
        assert h.ids["fast"] not in {m["node_id"] for m in net["members"]}

        failed = h.conn.execute("SELECT * FROM jobs WHERE state='failed'").fetchall()
        assert len(failed) == 1 and failed[0]["retryable"] == 1
        retry = h.conn.execute("SELECT * FROM jobs WHERE retry_of=?", (failed[0]["id"],)).fetchone()
        assert retry["state"] == "done" and retry["attempt"] == 2
        # nobody is credited for the failed attempt: one record, for the retry's members only
        (rec,) = h.svc.accounting.records[1:]
        assert h.ids["fast"] not in rec["per_node"]
        states = {p["id"]: p["state"] for p in pipelines(h)}
        assert states[first["id"]] == "broken"
        await asyncio.sleep(0.6)  # monitor notices the missed heartbeats
        rel = h.conn.execute("SELECT reliability FROM nodes WHERE id=?", (h.ids["fast"],)).fetchone()["reliability"]
        assert rel < 1.0
    finally:
        await h.stop()


async def test_lowering_commitment_drains_then_splits():
    h = await start_harness(fast_settings())
    try:
        h.add_model(qwen_layout())
        await h.add_nodes([FakeNode("studio", 64, may_be_head=True, files=(QWEN,)),
                           FakeNode("pc", 32, gen_score=0.8)])
        r = await chat(h, QWEN)
        assert len(r.json()["network"]["members"]) == 1
        await h.agents["studio"].update_commitment(Commitment(memory_gb=12, may_be_head=True))
        for _ in range(30):
            await asyncio.sleep(0.05)
            if pipelines(h)[0]["state"] == "stopped":
                break
        assert pipelines(h)[0]["state"] == "stopped"
        r2 = await chat(h, QWEN)
        assert r2.status_code == 200
        assert {m["node"] for m in r2.json()["network"]["members"]} == {"studio", "pc"}
    finally:
        await h.stop()


async def test_wrong_build_is_refused_and_plan_shows_members():
    h = await start_harness(fast_settings(PINNED_LLAMA_BUILD="b11160"))
    try:
        h.add_model(qwen_layout())
        await h.add_nodes([FakeNode("head", 16, may_be_head=True, files=(QWEN,)), FakeNode("worker", 16)])
        async with httpx.AsyncClient(base_url=h.node_url) as c:
            r = await c.post("/nodes/register", json={"name": "old", "llama_build": "b9000-abc", "commitment": {}})
            assert r.status_code == 409
            q = await c.get("/plan", params={"model": QWEN, "max_tokens": 100})
            assert q.status_code == 200
            body = q.json()
            assert body["plan"]["source"] in ("planner dry run", "active pipeline")
            assert {m["node"] for m in body["plan"]["members"]} == {"head", "worker"}
            assert body["gen_weight"] >= 1.0
        async with httpx.AsyncClient(base_url=h.url) as c:
            models = (await c.get("/v1/models")).json()["data"]
            assert [m["id"] for m in models] == [QWEN]
            assert models[0]["heads"] == ["head"]
    finally:
        await h.stop()


async def test_unknown_model_is_404(two_node):
    r = await chat(two_node, "nope.gguf")
    assert r.status_code == 404


async def test_idle_pipeline_of_the_same_model_is_evicted_for_a_larger_ctx(two_node):
    h = two_node
    small = await chat(h, QWEN, max_tokens=16)
    assert small.status_code == 200, small.text
    (first,) = pipelines(h)
    assert first["ctx"] == 4096
    # needs ctx 8192, which only fits once the idle ctx-4096 pipeline gives its memory back
    big = await chat(h, QWEN, content="long answer", max_tokens=5000)
    assert big.status_code == 200, big.text
    states = {p["id"]: (p["state"], p["ctx"]) for p in pipelines(h)}
    assert states[first["id"]] == ("stopped", 4096)
    assert big.json()["network"]["pipeline_id"] != first["id"]
    assert states[big.json()["network"]["pipeline_id"]] == ("active", 8192)
    # a smaller request reuses the bigger pipeline instead of evicting it
    again = await chat(h, QWEN, content="short again", max_tokens=16)
    assert again.json()["network"]["pipeline_id"] == big.json()["network"]["pipeline_id"]


async def test_requests_during_a_drain_wait_instead_of_failing():
    h = await start_harness(fast_settings(), time_scale=1.0)
    try:
        h.add_model(qwen_layout())
        await h.add_nodes([FakeNode("studio", 64, may_be_head=True, files=(QWEN,)),
                           FakeNode("pc", 32, gen_score=0.8)])
        assert (await chat(h, QWEN, max_tokens=2)).status_code == 200
        long = asyncio.create_task(chat(h, QWEN, max_tokens=30, content="in flight while draining"))
        await asyncio.sleep(0.3)
        await h.agents["studio"].update_commitment(Commitment(memory_gb=12, may_be_head=True))
        during = await chat(h, QWEN, max_tokens=4, content="arrives during the drain")
        assert (await long).status_code == 200
        assert during.status_code == 200, during.text
        assert {m["node"] for m in during.json()["network"]["members"]} == {"studio", "pc"}
    finally:
        await h.stop()


def _late_agent(port):
    from slashcompute.inference.node.agent import Agent
    from slashcompute.inference.node.config import NodeConfig
    from slashcompute.inference.node.fake_engine import FakeCluster, FakeEngine

    import tempfile

    from inf_harness import isolated_paths

    url = f"http://127.0.0.1:{port}/inference"
    cfg = NodeConfig(coordinator_url=url, name="late", **isolated_paths(tempfile.mkdtemp(prefix="late-node-")),
                     commitment=Commitment(memory_gb=16, may_be_head=True))
    return Agent(cfg, FakeEngine("late", FakeCluster()), info={"os": "fake", "chip": "fake"}, build="fake",
                 ip="127.0.0.1", gguf_files=[], latency_fn=None, busy_fn=lambda: False,
                 client=httpx.AsyncClient(base_url=url, timeout=5))


async def test_node_waits_for_a_coordinator_that_is_still_starting():
    """The launcher may start the node before the coordinator answers: keep retrying, then join."""
    import logging

    from inf_harness import Harness, free_port
    from slashcompute.inference.node.__main__ import connect
    from slashcompute.inference.node.fake_engine import FakeCluster

    port = free_port()
    agent = _late_agent(port)
    joining = asyncio.create_task(connect(agent, asyncio.Event(), logging.getLogger("t"), max_delay=0.2))
    await asyncio.sleep(0.5)
    assert not joining.done()                        # nothing listening yet: still retrying
    h = await Harness(fast_settings(), FakeCluster()).start(port)
    try:
        assert await asyncio.wait_for(joining, 10) is True
        assert agent.node_id in {r["id"] for r in h.conn.execute("SELECT id FROM nodes")}
    finally:
        await agent.client.aclose()
        await h.stop()


async def test_node_gives_up_retrying_when_stopped():
    import logging

    from inf_harness import free_port
    from slashcompute.inference.node.__main__ import connect

    agent = _late_agent(free_port())
    shutdown = asyncio.Event()
    joining = asyncio.create_task(connect(agent, shutdown, logging.getLogger("t"), max_delay=0.2))
    await asyncio.sleep(0.3)
    shutdown.set()
    assert await asyncio.wait_for(joining, 2) is False
    await agent.client.aclose()


async def test_node_reports_a_coordinator_without_inference_instead_of_stale_status():
    """An older coordinator answers /health but 404s /inference/*: say so in status.json, keep retrying."""
    import logging

    import uvicorn
    from fastapi import FastAPI

    from inf_harness import free_port
    from slashcompute.inference.node.__main__ import connect

    old = FastAPI()
    old.get("/health")(lambda: {"ok": True, "nodes": 0, "jobs": 0})
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(old, host="127.0.0.1", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    agent = _late_agent(port)
    status = agent.cfg.status_file
    with open(status, "w") as fh:   # an earlier run's leftovers
        json.dump({"available": True, "reason": "available", "last_error": ""}, fh)
    shutdown = asyncio.Event()
    joining = asyncio.create_task(connect(agent, shutdown, logging.getLogger("t"), max_delay=0.2))
    try:
        await asyncio.sleep(0.5)
        assert not joining.done()                    # still retrying: the coordinator may be updated
        data = json.load(open(status))
        assert data["available"] is False and data["reason"] == "unsupported"
        assert "no LLM inference" in data["last_error"]
    finally:
        shutdown.set()
        assert await asyncio.wait_for(joining, 2) is False
        await agent.client.aclose()
        server.should_exit = True
        await serving


async def test_malformed_messages_are_400_before_any_job(two_node):
    h = two_node
    async with httpx.AsyncClient(base_url=h.url, timeout=30) as c:
        for extra in ({"messages": []}, {}, {"messages": "hello"}, {"messages": [1]}):
            r = await c.post("/v1/chat/completions", json={"model": QWEN, **extra})
            assert r.status_code == 400, (extra, r.text)
    assert h.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert pipelines(h) == []


class RejectingEngine(FakeEngine):
    """Rejects the request like llama-server does a prompt larger than its context."""

    async def complete(self, pipeline_id, body):
        if body["messages"][0]["content"] == "too long":
            raise EngineError("invalid request: request (20010 tokens) exceeds the available context size "
                              "(8192 tokens)", pipeline_broken=False, status=400)
        async for ev in super().complete(pipeline_id, body):
            yield ev


@pytest.mark.parametrize("stream", [False, True])
async def test_rejected_request_is_a_client_error_and_keeps_the_pipeline(stream):
    h = await start_harness(fast_settings())
    try:
        h.add_model(qwen_layout())
        await h.add_nodes([FakeNode("head", 16, may_be_head=True, files=(QWEN,)), FakeNode("worker", 16)],
                          engine_cls=RejectingEngine)
        assert (await chat(h, QWEN)).status_code == 200
        (p,) = pipelines(h)

        r = await chat(h, QWEN, content="too long", stream=stream)
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["retryable"] is False and "exceeds the available context size" in err["message"]
        # not retried, and the pipeline (and everyone else's requests on it) is untouched
        failed = h.conn.execute("SELECT * FROM jobs WHERE state='failed'").fetchall()
        assert len(failed) == 1 and failed[0]["retryable"] == 0
        assert h.conn.execute("SELECT COUNT(*) FROM jobs WHERE retry_of IS NOT NULL").fetchone()[0] == 0
        assert [(q["id"], q["state"]) for q in pipelines(h)] == [(p["id"], "active")]
        r = await chat(h, QWEN, content="again")
        assert r.status_code == 200 and r.json()["network"]["pipeline_id"] == p["id"]
    finally:
        await h.stop()
