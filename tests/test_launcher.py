import json
import signal
import stat
from pathlib import Path

import pytest

from slashcompute.launcher.controller import (
    Launcher, LauncherError, LauncherSettings, health_timeout, normalize_url,
)


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health
        self.urls: list[str] = []

    def get(self, url: str, timeout: float = 1.0):
        self.urls.append(url)
        if self.health is None:
            raise ConnectionError("down")
        return FakeResponse(self.health)


def _launcher(tmp_path: Path, **kw) -> Launcher:
    spawned: list[FakeProc] = []
    next_pid = {"n": 4000}

    def popen(argv, **_kw):
        next_pid["n"] += 1
        proc = FakeProc(next_pid["n"], argv)
        spawned.append(proc)
        return proc

    launcher = Launcher(
        home=tmp_path,
        python="/opt/venv/bin/python",
        popen=popen,
        http=kw.pop("http", FakeHTTP()),
        discover_fn=kw.pop("discover_fn", lambda timeout=5.0: None),
        lan_ip_fn=kw.pop("lan_ip_fn", lambda: "192.168.1.20"),
    )
    launcher._spawned = spawned  # type: ignore[attr-defined]
    return launcher


def test_normalize_url():
    assert normalize_url("") == ""
    assert normalize_url("192.168.1.10") == "http://192.168.1.10:8765"
    assert normalize_url("192.168.1.10:9000") == "http://192.168.1.10:9000"
    assert normalize_url("http://10.0.0.2:8765/") == "http://10.0.0.2:8765"
    assert normalize_url("pool.example.com", scheme="https") == "https://pool.example.com:8765"
    assert normalize_url("https://pool.example.com/") == "https://pool.example.com"
    assert health_timeout("https://pool.example.com") == 5.0
    assert health_timeout("http://10.0.0.1:8765") == 1.0


def test_settings_persist(tmp_path):
    launcher = _launcher(tmp_path)
    s = LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=75,
                         contribute=False)
    launcher.save_settings(s)
    raw = json.loads((tmp_path / "launcher.json").read_text())
    assert raw["mode"] == "join"
    assert raw["gpu_percent"] == 75
    loaded = launcher.load_settings()
    assert loaded == s.clamp()


def test_settings_corrupt_and_clamp(tmp_path):
    launcher = _launcher(tmp_path)
    (tmp_path / "launcher.json").write_text("not-json")
    assert launcher.load_settings() == LauncherSettings()
    launcher.save_settings(LauncherSettings(mode="nope", gpu_percent=999, contribute=1))
    s = launcher.load_settings()
    assert s.mode == "host"
    assert s.gpu_percent == 100
    assert s.contribute is True
    assert LauncherSettings(mode="public").clamp().mode == "public"


def test_coordinator_and_agent_argv(tmp_path):
    launcher = _launcher(tmp_path)
    assert launcher.coordinator_argv() == [
        "/opt/venv/bin/python", "-m", "slashcompute.coordinator.main", "serve",
        "--home", str(tmp_path),
    ]
    assert launcher.agent_argv("http://192.168.1.20:8765", 40) == [
        "/opt/venv/bin/python", "-m", "slashcompute.agent.main", "start",
        "--url", "http://192.168.1.20:8765", "--gpu-percent", "40",
        "--home", str(tmp_path),
    ]
    assert "--session-token" in launcher.agent_argv(
        "http://192.168.1.20:8765", 40, session_token="tok",
    )


def test_host_url_uses_lan_ip(tmp_path):
    launcher = _launcher(tmp_path, lan_ip_fn=lambda: "10.1.2.3")
    assert launcher.coordinator_url(LauncherSettings(mode="host")) == "http://10.1.2.3:8765"
    assert launcher.coordinator_url(LauncherSettings(mode="join", url="10.1.2.9")) == (
        "http://10.1.2.9:8765"
    )
    assert launcher.coordinator_url(LauncherSettings(mode="public", url="pool.example.com")) == (
        "https://pool.example.com:8765"
    )
    launcher.cfg.public_url = "https://pool.example.com"
    assert launcher.coordinator_url(LauncherSettings(mode="public", url="")) == (
        "https://pool.example.com"
    )


def test_start_host_spawns_coordinator_and_agent(tmp_path, monkeypatch):
    http = FakeHTTP()
    launcher = _launcher(tmp_path, http=http)

    def health_after_spawn(url: str):
        if launcher._spawned:  # type: ignore[attr-defined]
            http.health = {"ok": True, "nodes": 0, "jobs": 0}
            return {"ok": True, "nodes": 0, "jobs": 0}
        return None

    def alive(pid: int) -> bool:
        return any(p.pid == pid for p in launcher._spawned)  # type: ignore[attr-defined]

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", alive)
    launcher.poll_health = health_after_spawn  # type: ignore[method-assign]
    snap = launcher.start(LauncherSettings(mode="host", gpu_percent=50, contribute=True))
    argv_lists = [p.argv for p in launcher._spawned]  # type: ignore[attr-defined]
    assert argv_lists[0][:4] == ["/opt/venv/bin/python", "-m", "slashcompute.coordinator.main", "serve"]
    assert argv_lists[1][2:4] == ["slashcompute.agent.main", "start"]
    assert "--url" in argv_lists[1] and "127.0.0.1" in argv_lists[1][argv_lists[1].index("--url") + 1]
    assert "--no-sandbox" not in argv_lists[1]
    assert (tmp_path / "coordinator.pid").read_text().strip() == str(launcher._spawned[0].pid)
    assert json.loads((tmp_path / "agent.args").read_text()) == argv_lists[1]
    assert stat.S_IMODE((tmp_path / "agent.args").stat().st_mode) == 0o600
    assert snap.last_error == ""


def test_start_host_without_contribute_skips_agent(tmp_path):
    http = FakeHTTP()
    launcher = _launcher(tmp_path, http=http)
    launcher.poll_health = lambda url: {"ok": True} if launcher._spawned else None  # type: ignore
    launcher.start(LauncherSettings(mode="host", contribute=False))
    kinds = [p.argv[2] for p in launcher._spawned]  # type: ignore[attr-defined]
    assert kinds == ["slashcompute.coordinator.main"]


@pytest.mark.parametrize(("old_token", "new_url", "new_gpu", "new_token"), [
    ("tok", "http://10.0.0.2:8765", 50, "tok"),
    ("tok", "http://10.0.0.1:8765", 75, "tok"),
    ("tok", "http://10.0.0.1:8765", 50, "newtok"),
    ("", "http://10.0.0.1:8765", 50, "tok"),
    ("tok", "http://10.0.0.1:8765", 50, ""),
])
def test_start_restarts_agent_when_effective_arguments_change(
    tmp_path, monkeypatch, old_token, new_url, new_gpu, new_token,
):
    http = FakeHTTP({"ok": True, "nodes": 0, "jobs": 0})
    launcher = _launcher(tmp_path, http=http)
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    old_args = launcher.agent_argv("http://10.0.0.1:8765", 50, old_token)
    (tmp_path / "agent.args").write_text(json.dumps(old_args))
    running = {77: True}
    stopped = []

    def alive(pid: int) -> bool:
        return running.get(pid, False)

    def stop(paths):
        stopped.append(paths.read_pid())
        running[77] = False
        paths.clear_pid()
        return True

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", alive)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    snap = launcher.start(LauncherSettings(
        mode="join", url=new_url, gpu_percent=new_gpu, session_token=new_token,
    ))
    spawned = launcher._spawned  # type: ignore[attr-defined]
    assert stopped == [77]
    assert len(spawned) == 1
    assert spawned[0].argv == launcher.agent_argv(new_url, new_gpu, new_token)
    assert json.loads((tmp_path / "agent.args").read_text()) == spawned[0].argv
    assert snap.last_error == ""


@pytest.mark.parametrize(("explicit_token", "expected_token", "should_restart"), [
    ("", "new-environment-token", True),
    ("explicit-token", "explicit-token", False),
])
def test_agent_restart_compares_effective_environment_session(
    tmp_path, monkeypatch, explicit_token, expected_token, should_restart,
):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    settings = LauncherSettings(
        mode="join", url="http://10.0.0.1:8765", session_token=explicit_token,
    )
    monkeypatch.setenv("SLASHCOMPUTE_SESSION", "old-environment-token")
    launcher.start(settings)
    old_args = launcher._spawned[0].argv
    assert old_args[-2:] == ["--session-token", explicit_token or "old-environment-token"]
    launcher.paths.pid_file.write_text("77\n")
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    stopped = []

    def stop(paths):
        stopped.append(paths.read_pid())
        paths.clear_pid()

    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    monkeypatch.setenv("SLASHCOMPUTE_SESSION", "new-environment-token")
    # Ensure rewriting an existing args file also repairs overly broad permissions.
    (tmp_path / "agent.args").chmod(0o644 if should_restart else 0o600)

    launcher.start(settings)

    assert stopped == ([77] if should_restart else [])
    assert len(launcher._spawned) == (2 if should_restart else 1)
    assert launcher._spawned[-1].argv[-2:] == ["--session-token", expected_token]
    assert json.loads((tmp_path / "agent.args").read_text()) == launcher._spawned[-1].argv
    assert stat.S_IMODE((tmp_path / "agent.args").stat().st_mode) == 0o600


def test_start_when_already_up_is_noop(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 1, "jobs": 0}))
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    argv = launcher.agent_argv("http://10.0.0.1:8765", 100, "tok")
    (tmp_path / "agent.args").write_text(json.dumps(argv))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: pytest.fail("unchanged agent should not stop"))
    snap = launcher.start(LauncherSettings(
        mode="join", url="http://10.0.0.1:8765/", gpu_percent=999, session_token="tok",
    ))
    assert launcher._spawned == []  # type: ignore[attr-defined]
    assert snap.agent_running and snap.last_error == ""


@pytest.mark.parametrize("args_text", [None, "not-json"])
def test_start_restarts_agent_with_unknown_previous_arguments(tmp_path, monkeypatch, args_text):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("77\n")
    # Launchers before agent.args recorded only the bound session.
    (tmp_path / "agent.session").write_text("tok")
    if args_text is not None:
        (tmp_path / "agent.args").write_text(args_text)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: paths.clear_pid())
    launcher.start(LauncherSettings(mode="host", session_token="tok"))
    assert len(launcher._spawned) == 1
    assert json.loads((tmp_path / "agent.args").read_text()) == launcher._spawned[0].argv


def test_agent_restart_waits_for_graceful_stop_and_reports_timeout(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("77\n")
    old_args = launcher.agent_argv("http://10.0.0.1:8765", 50, "tok")
    (tmp_path / "agent.args").write_text(json.dumps(old_args))
    stopped = []
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: stopped.append(paths.read_pid()))
    times = iter([0.0, 0.0, 4.0])
    monkeypatch.setattr("slashcompute.launcher.controller.time.monotonic", lambda: next(times))
    monkeypatch.setattr("slashcompute.launcher.controller.time.sleep", lambda seconds: None)
    settings = LauncherSettings(mode="join", url="http://10.0.0.2:8765", session_token="tok")

    snap = launcher.start(settings)

    assert stopped == [77]
    assert snap.agent_running and snap.agent_pid == 77
    assert "Settings have not been applied" in snap.last_error
    assert launcher._spawned == []
    assert json.loads((tmp_path / "agent.args").read_text()) == old_args

    # Retrying after the old agent drains applies the saved configuration.
    launcher.paths.clear_pid()
    snap = launcher.start(settings)
    assert len(launcher._spawned) == 1
    assert launcher._spawned[0].argv == launcher.agent_argv(settings.url, 50, "tok")
    assert json.loads((tmp_path / "agent.args").read_text()) == launcher._spawned[0].argv
    assert snap.last_error == ""


def test_start_join_requires_url(tmp_path):
    launcher = _launcher(tmp_path)
    with pytest.raises(LauncherError, match="coordinator URL"):
        launcher.start(LauncherSettings(mode="join", url=""))
    assert launcher._spawned == []  # type: ignore[attr-defined]


def test_start_join_without_health_fails(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP(None))
    with pytest.raises(LauncherError, match="No coordinator"):
        launcher.start(LauncherSettings(mode="join", url="http://10.0.0.8:8765"))


def test_unreachable_error_clears_once_the_coordinator_answers(tmp_path):
    http = FakeHTTP(None)
    launcher = _launcher(tmp_path, http=http)
    with pytest.raises(LauncherError, match="No coordinator"):
        launcher.start(LauncherSettings(mode="join", url="http://127.0.0.1:9399"))
    # Still down: the error stays.
    assert launcher.snapshot().last_error == "No coordinator at http://127.0.0.1:9399."

    # Connect saves a fixed address (no start); the next status poll must drop the stale banner.
    launcher.save_settings(LauncherSettings(mode="join", url="http://10.0.0.8:8765"))
    http.health = {"ok": True}
    snap = launcher.snapshot()
    assert snap.coordinator_up and snap.last_error == ""
    assert launcher.last_error == ""


def test_reachable_coordinator_keeps_other_errors(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.last_error = "Training agent is still stopping. Settings have not been applied."
    assert launcher.snapshot().last_error == launcher.last_error != ""


def test_start_public_requires_url_and_token(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    with pytest.raises(LauncherError, match="public coordinator URL"):
        launcher.start(LauncherSettings(mode="public", url="", session_token="tok"))
    with pytest.raises(LauncherError, match="Sign in first"):
        launcher.start(LauncherSettings(mode="public", url="https://pool.example.com"))
    assert launcher._spawned == []  # type: ignore[attr-defined]


def test_start_public_does_not_spawn_coordinator(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 0, "jobs": 0}))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in launcher._spawned))
    snap = launcher.start(LauncherSettings(
        mode="public", url="https://pool.example.com", session_token="tok",
        contribute=True,
    ))
    kinds = [p.argv[2] for p in launcher._spawned]  # type: ignore[attr-defined]
    assert kinds == ["slashcompute.agent.main"]
    assert snap.coordinator_pid is None
    agent = launcher._spawned[0].argv  # type: ignore[attr-defined]
    assert agent[agent.index("--url") + 1] == "https://pool.example.com"
    assert agent[agent.index("--session-token") + 1] == "tok"


def test_find_on_lan(tmp_path):
    launcher = _launcher(
        tmp_path,
        discover_fn=lambda timeout=5.0: "http://192.168.0.4:8765/",
    )
    assert launcher.find_on_lan() == "http://192.168.0.4:8765"


def test_stop_signals_agent_and_coordinator(tmp_path, monkeypatch):
    kills: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))

    monkeypatch.setattr("os.kill", fake_kill)
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    (tmp_path / "agent" / "agent.pid").write_text("222\n")
    launcher.stop()
    assert (111, signal.SIGTERM) in kills
    assert (222, signal.SIGTERM) in kills


def test_snapshot_reads_health_and_agent_status(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: pid == 9)
    launcher = _launcher(
        tmp_path,
        http=FakeHTTP({"ok": True, "nodes": 2, "jobs": 1}),
        lan_ip_fn=lambda: "192.168.9.9",
    )
    launcher.paths.write_status(status="idle", job_id=None)
    launcher.paths.pid_file.write_text("9\n")
    snap = launcher.snapshot(LauncherSettings(mode="host"))
    assert snap.coordinator_up is True
    assert snap.nodes == 2
    assert snap.jobs == 1
    assert snap.agent_running is True
    assert snap.agent_status == "idle"
    assert snap.lan_ip == "192.168.9.9"


def test_stop_agent_leaves_coordinator_running(tmp_path, monkeypatch):
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    (tmp_path / "agent" / "agent.pid").write_text("222\n")
    launcher.stop_agent()
    assert (222, signal.SIGTERM) in kills
    assert (111, signal.SIGTERM) not in kills  # only liveness probes (signal 0)


def test_my_node_id_never_creates_one(tmp_path):
    launcher = _launcher(tmp_path)
    assert launcher.my_node_id() is None
    assert not launcher.paths.node_id_file.exists()
    launcher.paths.node_id_file.write_text("abc123\n")
    assert launcher.my_node_id() == "abc123"


def test_fetch_pool_down_and_partial(tmp_path):
    assert _launcher(tmp_path, http=FakeHTTP(None)).fetch_pool("http://x:8765").online is False
    # Health answers but list endpoints return a dict: tolerated as empty lists.
    pool = _launcher(tmp_path, http=FakeHTTP({"ok": True})).fetch_pool("http://x:8765")
    assert pool.online is True and (pool.nodes, pool.jobs, pool.ledger) == ([], [], [])


# ------------------------------------------------------------ memory lent

def test_memory_setting_round_trips_clamps_and_reaches_the_agent(tmp_path):
    launcher = _launcher(tmp_path)
    launcher.save_settings(LauncherSettings(memory_gb=6, inference_memory_gb=10))
    s = launcher.load_settings()
    assert (s.memory_gb, s.inference_memory_gb) == (6, 10)
    assert LauncherSettings(memory_gb="lots").clamp().memory_gb == 0
    assert LauncherSettings(memory_gb=-3).clamp().memory_gb == 0
    assert "--max-memory-gb" not in launcher.agent_argv("http://10.0.0.1:8765", 50)       # 0 = automatic
    assert launcher.agent_argv("http://10.0.0.1:8765", 50, "", 6)[-2:] == ["--max-memory-gb", "6"]


def test_changing_memory_restarts_the_running_agent(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 0, "jobs": 0}))
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    (tmp_path / "agent.args").write_text(json.dumps(launcher.agent_argv("http://10.0.0.1:8765", 50)))
    running = {77: True}
    stopped = []

    def stop(paths):
        stopped.append(paths.read_pid())
        running[77] = False
        paths.clear_pid()
        return True

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: running.get(pid, False))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    launcher.start(LauncherSettings(mode="join", url="http://10.0.0.1:8765", memory_gb=6))
    assert stopped == [77]
    assert launcher._spawned[0].argv == launcher.agent_argv("http://10.0.0.1:8765", 50, "", 6)  # type: ignore


def test_status_reports_this_macs_memory(tmp_path):
    launcher = _launcher(tmp_path)
    launcher._memory = lambda: (16 << 30, 5 << 30)
    snap = launcher.snapshot()
    assert (snap.memory_total_bytes, snap.memory_available_bytes) == (16 << 30, 5 << 30)
