"""Public coordinator: login, one stage, no WAN pipeline."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from slashcompute.agent.daemon import Daemon, AgentOptions, peered_assignment, public_transport
from slashcompute.common import protocol as P
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.jobs import LoraFinetuneSpec


@pytest.fixture
def env(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0,
                       public_pool=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app.state.core, tiny_model, tiny_dataset


def _device(mem=8 << 30):
    return P.DeviceProfile(
        chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
        working_set_bytes=mem, memory_contrib_bytes=mem,
        matmul_tflops=1.0, mem_bandwidth_gbps=100.0,
    )


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=2,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=2, max_stages=4)
    return LoraFinetuneSpec(**(base | kw))


def _hdr(token):
    return {"Authorization": f"Bearer {token}"}


def _account(client, email="ada@lan.test"):
    token = client.post("/auth/register", json={
        "email": email, "password": "password1", "name": "Ada",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(token))
    return token


def test_public_anon_submit_rejected(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec)
    assert r.status_code == 401
    assert core.jobs == {}


def test_public_upload_clamps_to_one_stage(env):
    client, core, tiny_model, tiny_dataset = env
    token = _account(client)
    core.credits.contribute(core.auth.session_user(token).id, 1e12, 0)
    r = client.post(
        "/jobs/upload",
        headers=_hdr(token),
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": str(tiny_model), "steps": 2, "min_stages": 4,
              "batch_size": 2, "microbatches": 1, "max_flops": 1e9},
    )
    assert r.status_code == 200, r.text
    job = core.jobs[r.json()["id"]]
    assert job.spec.min_stages == 1
    assert job.spec.max_stages == 1


def test_public_register_requires_session(env):
    client, *_ = env
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_text(P.dump(P.Register(
                node_id="anon", name="mac", device=_device(),
                data_host="127.0.0.1", data_port=9700, gpu_percent=50,
            )))
            ws.receive_text()
    assert e.value.code == 4003


def test_public_register_requires_terms(env):
    client, *_ = env
    token = client.post("/auth/register", json={
        "email": "bob@lan.test", "password": "password1", "name": "Bob",
    }).json()["token"]
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_text(P.dump(P.Register(
                node_id="bob", name="mac", device=_device(),
                data_host="127.0.0.1", data_port=9700, gpu_percent=50,
                session_token=token,
            )))
            ws.receive_text()
    assert e.value.code == 4003


def test_public_register_welcome_after_terms(env):
    client, core, *_ = env
    token = _account(client)
    with client.websocket_connect("/ws/agent") as ws:
        ws.send_text(P.dump(P.Register(
            node_id="ok", name="mac", device=_device(),
            data_host="127.0.0.1", data_port=9700, gpu_percent=50,
            session_token=token,
        )))
        welcome = P.parse_coordinator_message(ws.receive_text())
        assert isinstance(welcome, P.Welcome)
        assert core.registry.get("ok").user_id is not None


def _register(ws, node_id, token):
    ws.send_text(P.dump(P.Register(
        node_id=node_id, name="mac", device=_device(),
        data_host="127.0.0.1", data_port=9700, gpu_percent=50,
        session_token=token,
    )))
    return P.parse_coordinator_message(ws.receive_text())


def test_public_register_refuses_another_users_node_id(env):
    client, core, *_ = env
    victim = _account(client, "victim@lan.test")
    attacker = _account(client, "mallory@lan.test")
    victim_id = core.auth.session_user(victim).id
    with client.websocket_connect("/ws/agent") as honest:
        assert isinstance(_register(honest, "honest-node", victim), P.Welcome)
        honest_state = core.registry.get("honest-node")
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect("/ws/agent") as ws:
                _register(ws, "honest-node", attacker)
        assert e.value.code == 4003
        assert core.registry.get("honest-node") is honest_state
        assert honest_state.user_id == victim_id
        assert core.credits.owner_of("honest-node") == victim_id
    # The owner reconnecting with its own node id (agent restart) still works.
    with client.websocket_connect("/ws/agent") as ws:
        assert isinstance(_register(ws, "honest-node", victim), P.Welcome)
        assert core.registry.get("honest-node").user_id == victim_id
    with pytest.raises(PermissionError):
        core.credits.bind_node("honest-node", core.auth.session_user(attacker).id)
    assert core.credits.owner_of("honest-node") == victim_id


def test_public_transport_and_peered_assignment():
    spec = LoraFinetuneSpec(dataset_path="/tmp/d.jsonl", steps=2)
    peered = P.StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=2, layer_start=0, layer_end=4,
        num_layers=8, spec=spec, next_peer=P.PeerAddr(node_id="b", host="h", port=1),
        checkpoint_every=25, verify_ring_size=8,
    )
    solo = P.StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=spec, checkpoint_every=25, verify_ring_size=8,
    )
    assert public_transport("https://pool.example.com") is True
    assert public_transport("http://10.0.0.1:8765") is False
    assert public_transport("http://10.0.0.1:8765", public_pool=True) is True
    assert peered_assignment(peered) is True
    assert peered_assignment(solo) is False


def test_daemon_refuses_peered_assignment_on_https(tmp_path):
    opt = AgentOptions(url="https://pool.example.com", home=tmp_path, localhost=True)
    daemon = Daemon(opt)
    asg = P.StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=2, layer_start=0, layer_end=4,
        num_layers=8, spec=LoraFinetuneSpec(dataset_path="/tmp/d.jsonl", steps=2),
        next_peer=P.PeerAddr(node_id="b", host="h", port=1),
        checkpoint_every=25, verify_ring_size=8,
    )
    asyncio.run(daemon._start_stage(asg))
    assert daemon._session is None
    assert daemon.status == "idle"
