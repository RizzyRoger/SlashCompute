"""Shell + launcher side of LLM inference: the LLMs view, streaming chat/upload passthrough, the inference node."""

import json
import os

import httpx
from fastapi.testclient import TestClient

from slashcompute.launcher.controller import Launcher, LauncherSettings
from slashcompute.web.server import create_shell


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        if self.health is None:
            raise ConnectionError("down")
        health = self.health

        class R:
            status_code = 200
            content = json.dumps(health).encode()
            headers = {"content-type": "application/json"}

            def json(self_inner):
                return health
        return R()


def _launcher(tmp_path, health=None):
    spawned = []
    n = {"p": 6000}

    def popen(argv, **_):
        n["p"] += 1
        spawned.append(FakeProc(n["p"], argv))
        return spawned[-1]

    launcher = Launcher(home=tmp_path, python="/opt/venv/bin/python", popen=popen, http=FakeHTTP(health),
                        discover_fn=lambda timeout=5.0: None, lan_ip_fn=lambda: "192.168.1.20")
    return launcher, spawned


CURRENT = {"ok": True, "inference_nodes": 0, "inference_transport": "direct"}
OUTDATED = {"ok": True, "nodes": 4, "jobs": 1}   # /health of a coordinator from before LLM inference


def _shell(tmp_path, handler, health=CURRENT):
    launcher, _ = _launcher(tmp_path, health)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return create_shell(launcher, stream_client=client)


# ------------------------------------------------------------ UI

def test_llm_view_is_served(tmp_path):
    with TestClient(_shell(tmp_path, lambda r: httpx.Response(404))) as c:
        html = c.get("/").content
        assert b'data-tab="llm"' in html and b'id="view-llm"' in html and b'id="chat-form"' in html
        js = c.get("/static/app.js").content
        assert b"/api/chat" in js and b"/api/models/upload" in js and b'"inference_memory_gb"' in js
        assert b"GOOGLE" not in js


# ------------------------------------------------------------ streaming passthrough

def test_chat_streams_sse_through_with_session(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["cookie"] = request.headers.get("cookie")
        sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    with TestClient(_shell(tmp_path, handler)) as c:
        c.cookies.set("slashcompute_session", "tok")
        r = c.post("/api/chat", json={"model": "m.gguf", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text.endswith("data: [DONE]\n\n")
    assert seen["url"] == "http://127.0.0.1:8765/v1/chat/completions"
    assert seen["body"]["stream"] is True
    assert "slashcompute_session=tok" in seen["cookie"]


def test_chat_errors_keep_their_status(tmp_path):
    def handler(request):
        return httpx.Response(503, json={"error": {"message": "no eligible head node has m.gguf on disk"}})

    with TestClient(_shell(tmp_path, handler)) as c:
        r = c.post("/api/chat", json={"model": "m.gguf", "messages": []})
    assert r.status_code == 503 and "no eligible head" in r.json()["error"]["message"]


def test_chat_without_a_coordinator_is_502(tmp_path):
    def handler(request):
        raise httpx.ConnectError("refused")

    with TestClient(_shell(tmp_path, handler)) as c:
        assert c.post("/api/chat", json={"model": "m", "messages": []}).status_code == 502


def test_chat_rejects_bodies_that_are_not_a_json_object(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200)

    with TestClient(_shell(tmp_path, handler)) as c:
        for raw in (b"notjson", b"[1,2]", b'"hi"', b"", b"\xff"):
            r = c.post("/api/chat", content=raw, headers={"content-type": "application/json"})
            assert r.status_code == 400, (raw, r.text)
            assert "JSON" in r.json()["detail"]
    assert sent == []


def test_upload_streams_the_file_to_the_coordinator(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, json={"name": "m.gguf", "size": len(seen["body"])})

    data = os.urandom(300_000)
    with TestClient(_shell(tmp_path, handler)) as c:
        r = c.post("/api/models/upload", content=data, headers={"x-filename": "m.gguf"})
    assert r.status_code == 200 and r.json()["size"] == len(data)
    assert seen["url"] == "http://127.0.0.1:8765/inference/models/upload?name=m.gguf"
    assert seen["body"] == data


def test_outdated_coordinator_is_explained_before_anything_is_sent(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404, json={"detail": "Not Found"})

    with TestClient(_shell(tmp_path, handler, OUTDATED)) as c:
        up = c.post("/api/models/upload", content=b"GGUF" + bytes(1000), headers={"x-filename": "m.gguf"})
        chat = c.post("/api/chat", json={"model": "m", "messages": []})
        status = c.get("/api/status").json()
    assert up.status_code == chat.status_code == 409
    assert "older /compute" in up.json()["detail"] and "older /compute" in chat.json()["detail"]
    assert seen == []                                   # the GGUF never left this Mac
    assert status["inference_supported"] is False


def test_inference_support_follows_coordinator_health(tmp_path):
    for health, expected in ((CURRENT, True), (OUTDATED, False), (None, None)):
        launcher, _ = _launcher(tmp_path, health)
        launcher.save_settings(LauncherSettings(mode="join", url="10.0.0.5"))
        assert launcher.snapshot().inference_supported is expected


# ------------------------------------------------------------ launcher

def test_new_settings_round_trip_and_clamp(tmp_path):
    launcher, _ = _launcher(tmp_path)
    launcher.save_settings(LauncherSettings(inference=True, training=False, inference_memory_gb=24,
                                            inference_head=False, models_dir="/Volumes/m", transport="relay"))
    s = launcher.load_settings()
    assert (s.inference, s.training, s.inference_memory_gb, s.inference_head, s.models_dir, s.transport) == \
        (True, False, 24, False, "/Volumes/m", "relay")
    bad = LauncherSettings(transport="carrier-pigeon", inference_memory_gb="lots").clamp()
    assert bad.transport == "direct" and bad.inference_memory_gb == 0


def test_inference_argv(tmp_path):
    launcher, _ = _launcher(tmp_path)
    argv = launcher.inference_argv("http://127.0.0.1:8765", LauncherSettings(
        inference_memory_gb=12, inference_head=False, session_token="tok"))
    assert argv[:4] == ["/opt/venv/bin/python", "-m", "slashcompute.inference.node", "start"]
    assert argv[argv.index("--url") + 1] == "http://127.0.0.1:8765"
    assert argv[argv.index("--memory-gb") + 1] == "12"
    assert "--no-head" in argv and argv[-2:] == ["--session-token", "tok"]
    assert launcher.coordinator_argv("relay")[-2:] == ["--inference-transport", "relay"]
    assert "--inference-transport" not in launcher.coordinator_argv()


def test_host_with_inference_spawns_the_node_and_relay_coordinator(tmp_path):
    launcher, spawned = _launcher(tmp_path)
    launcher.poll_health = lambda url: {"ok": True, "inference_transport": "relay"} if spawned else None
    launcher.start(LauncherSettings(mode="host", contribute=True, inference=True, transport="relay"))
    mods = [p.argv[2] for p in spawned]
    assert mods == ["slashcompute.coordinator.main", "slashcompute.agent.main", "slashcompute.inference.node"]
    assert spawned[0].argv[-2:] == ["--inference-transport", "relay"]
    assert "127.0.0.1" in spawned[2].argv[spawned[2].argv.index("--url") + 1]


def test_training_off_lends_only_to_inference(tmp_path):
    launcher, spawned = _launcher(tmp_path)
    launcher.poll_health = lambda url: {"ok": True} if spawned else None
    launcher.start(LauncherSettings(mode="host", contribute=True, training=False, inference=True))
    assert [p.argv[2] for p in spawned] == ["slashcompute.coordinator.main", "slashcompute.inference.node"]


def test_changed_inference_settings_restart_the_node(tmp_path, monkeypatch):
    launcher, spawned = _launcher(tmp_path, {"ok": True})
    running = {"pid": None}
    monkeypatch.setattr(launcher, "read_inference_pid", lambda: running["pid"])
    stopped = []
    monkeypatch.setattr(launcher, "_stop_inference", lambda wait=0.0: (stopped.append(1), running.update(pid=None)))
    launcher.start(LauncherSettings(mode="join", url="10.0.0.9", training=False, inference=True))
    running["pid"] = spawned[-1].pid
    launcher.start(LauncherSettings(mode="join", url="10.0.0.9", training=False, inference=True))
    assert len(spawned) == 1                                    # same settings: keep running
    launcher.start(LauncherSettings(mode="join", url="10.0.0.9", training=False, inference=True,
                                    inference_memory_gb=8))
    assert len(spawned) == 2 and stopped                        # new memory: drained and rejoined
    assert spawned[-1].argv[spawned[-1].argv.index("--memory-gb") + 1] == "8"


def test_transport_change_restarts_our_coordinator(tmp_path, monkeypatch):
    launcher, spawned = _launcher(tmp_path, {"ok": True, "inference_transport": "direct"})
    monkeypatch.setattr(launcher, "read_coordinator_pid", lambda: 4242)   # we started it
    killed = []
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: False)
    monkeypatch.setattr(launcher, "wait_health", lambda url, timeout=8.0: True)
    launcher.start(LauncherSettings(mode="host", contribute=False, transport="relay"))
    assert killed and killed[0][0] == 4242
    assert spawned[0].argv[-2:] == ["--inference-transport", "relay"]
