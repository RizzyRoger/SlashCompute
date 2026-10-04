"""Start/stop the coordinator, the MLX agent and the inference node; persist launcher settings."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import psutil

from slashcompute.agent.daemon import request_stop
from slashcompute.agent.paths import AgentPaths
from slashcompute.common.config import EngineConfig
from slashcompute.common.discovery import discover, lan_ip
from slashcompute.launcher.dashboard import PoolData


class LauncherError(Exception):
    """User-facing start/stop failure."""


FINISHES = ("carbon", "poster", "signal", "thermal", "void")
TRANSPORTS = ("direct", "relay")
# Errors that only say the coordinator could not be reached: stale once it answers.
UNREACHABLE_ERRORS = ("No coordinator at ", "Coordinator started but is not answering ")
OUTDATED_COORDINATOR = ("This pool's coordinator has no LLM inference: it runs an older /compute. "
                        "Ask whoever hosts it to update and restart it, or host a pool on this Mac.")


def stateless_http(**kw: Any) -> httpx.Client:
    """Client for the shared shell: it proxies many browsers, so it must never keep a cookie (a
    stored Set-Cookie would sign every cookie-less request in as the last user). Each request
    carries only the caller's own Cookie/Authorization headers."""
    jar = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))
    return httpx.Client(follow_redirects=False, cookies=jar, **kw)


def supports_inference(health: Optional[dict]) -> Optional[bool]:
    """Whether the coordinator serves LLMs (every build since inference reports its transport in
    /health). None while it is unreachable."""
    return None if not health else "inference_transport" in health


@dataclass
class LauncherSettings:
    mode: str = "host"
    url: str = ""
    gpu_percent: int = 50
    contribute: bool = True
    finish: str = "carbon"
    session_token: str = ""
    grant_split: int = 0
    training: bool = True              # lend this Mac to MLX fine-tunes
    memory_gb: int = 0                 # GiB lent to fine-tunes; 0 = automatic (what is free at start)
    inference: bool = False            # also host llama.cpp layers for the pool's LLMs
    inference_memory_gb: int = 0       # 0 = automatic (75% of RAM minus 4 GiB)
    inference_head: bool = True        # may run llama-server (needs the model file; uploads are pushed)
    models_dir: str = "~/models"
    transport: str = "direct"          # host only: direct (LAN) or relay (internet, RPC via the coordinator)

    def clamp(self) -> "LauncherSettings":
        mode = self.mode if self.mode in ("host", "join", "public") else "host"
        finish = self.finish if self.finish in FINISHES else "carbon"
        transport = self.transport if self.transport in TRANSPORTS else "direct"
        try:
            gpu = max(1, min(100, int(self.gpu_percent)))
        except (TypeError, ValueError):
            gpu = 50
        try:
            split = max(0, min(100, int(self.grant_split)))
        except (TypeError, ValueError):
            split = 0
        try:
            mem = max(0, min(1024, int(self.inference_memory_gb)))
        except (TypeError, ValueError):
            mem = 0
        try:
            train_mem = max(0, min(1024, int(self.memory_gb)))
        except (TypeError, ValueError):
            train_mem = 0
        return LauncherSettings(
            mode=mode, url=str(self.url or ""), gpu_percent=gpu,
            contribute=bool(self.contribute), finish=finish,
            session_token=str(self.session_token or ""), grant_split=split,
            training=bool(self.training), memory_gb=train_mem, inference=bool(self.inference),
            inference_memory_gb=mem,
            inference_head=bool(self.inference_head), models_dir=str(self.models_dir or "~/models"),
            transport=transport,
        )


@dataclass
class StatusSnapshot:
    coordinator_up: bool = False
    nodes: int = 0
    jobs: int = 0
    agent_running: bool = False
    agent_status: str = ""
    agent_job_id: Optional[str] = None
    coordinator_pid: Optional[int] = None
    agent_pid: Optional[int] = None
    lan_ip: str = ""
    last_error: str = ""
    inference_running: bool = False
    inference_pid: Optional[int] = None
    inference_status: dict = field(default_factory=dict)   # the node's status.json
    inference_nodes: int = 0
    inference_transport: str = ""
    inference_supported: Optional[bool] = None
    memory_total_bytes: int = 0        # this Mac's unified memory, for the memory sliders
    memory_available_bytes: int = 0


PopenFn = Callable[..., Any]


def normalize_url(url: str, port: int = 8765, scheme: str = "http") -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        if ":" not in u.split("/")[0]:
            u = f"{u}:{port}"
        u = f"{scheme}://{u}"
    return u.rstrip("/")


def health_timeout(url: str) -> float:
    return 5.0 if (url or "").lower().startswith("https://") else 1.0


def system_memory() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Launcher:
    def __init__(
        self,
        home: Optional[Path] = None,
        python: Optional[str] = None,
        popen: PopenFn = subprocess.Popen,
        http: Optional[httpx.Client] = None,
        discover_fn: Callable[[float], Optional[str]] = discover,
        lan_ip_fn: Callable[[], str] = lan_ip,
        memory_fn: Callable[[], tuple[int, int]] = system_memory,
    ) -> None:
        self.cfg = EngineConfig.from_env(home=home)
        if home is not None:
            self.cfg.home = Path(home)
        self.home = Path(self.cfg.home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.python = python or sys.executable
        self._popen = popen
        self._http = http or stateless_http()
        self._discover = discover_fn
        self._lan_ip = lan_ip_fn
        self._memory = memory_fn
        self.last_error = ""
        self.paths = AgentPaths(self.home)

    @property
    def settings_path(self) -> Path:
        return self.home / "launcher.json"

    @property
    def coordinator_pid_path(self) -> Path:
        return self.home / "coordinator.pid"

    @property
    def log_dir(self) -> Path:
        d = self.home / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_settings(self) -> LauncherSettings:
        if not self.settings_path.exists():
            return LauncherSettings()
        try:
            raw = json.loads(self.settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            return LauncherSettings()
        if not isinstance(raw, dict):
            return LauncherSettings()
        return LauncherSettings(
            mode=raw.get("mode", "host"),
            url=raw.get("url", ""),
            gpu_percent=raw.get("gpu_percent", 50),
            contribute=raw.get("contribute", True),
            finish=raw.get("finish", "carbon"),
            session_token=raw.get("session_token", ""),
            grant_split=raw.get("grant_split", 0),
            training=raw.get("training", True),
            memory_gb=raw.get("memory_gb", 0),
            inference=raw.get("inference", False),
            inference_memory_gb=raw.get("inference_memory_gb", 0),
            inference_head=raw.get("inference_head", True),
            models_dir=raw.get("models_dir", "~/models"),
            transport=raw.get("transport", "direct"),
        ).clamp()

    def save_settings(self, settings: LauncherSettings) -> None:
        s = settings.clamp()
        self.settings_path.write_text(json.dumps(asdict(s), indent=2) + "\n")

    def coordinator_url(self, settings: LauncherSettings) -> str:
        if settings.mode == "host":
            return f"http://{self._lan_ip()}:{self.cfg.coordinator_port}"
        raw = settings.url or (self.cfg.public_url if settings.mode == "public" else "")
        scheme = "https" if settings.mode == "public" else "http"
        return normalize_url(raw, self.cfg.coordinator_port, scheme=scheme)

    def proxy_url(self, settings: Optional[LauncherSettings] = None) -> str:
        """Coordinator URL the local shell should dial (loopback when hosting)."""
        s = settings.clamp() if settings is not None else self.load_settings()
        if s.mode == "host":
            return f"http://127.0.0.1:{self.cfg.coordinator_port}"
        return self.coordinator_url(s)

    def coordinator_argv(self, transport: str = "direct") -> list[str]:
        argv = [self.python, "-m", "slashcompute.coordinator.main", "serve",
                "--home", str(self.home)]
        if transport != "direct":
            argv.extend(["--inference-transport", transport])
        return argv

    def agent_argv(self, url: str, gpu_percent: int, session_token: str = "", memory_gb: int = 0) -> list[str]:
        argv = [
            self.python, "-m", "slashcompute.agent.main", "start",
            "--url", url, "--gpu-percent", str(int(gpu_percent)),
            "--home", str(self.home),
        ]
        if memory_gb:
            argv.extend(["--max-memory-gb", str(int(memory_gb))])
        if session_token:
            argv.extend(["--session-token", session_token])
        return argv

    def inference_argv(self, url: str, settings: LauncherSettings) -> list[str]:
        s = settings.clamp()
        argv = [
            self.python, "-m", "slashcompute.inference.node", "start",
            "--url", url, "--home", str(self.home), "--models-dir", s.models_dir,
            "--memory-gb", str(s.inference_memory_gb), "--head" if s.inference_head else "--no-head",
        ]
        if s.session_token:
            argv.extend(["--session-token", s.session_token])
        return argv

    @property
    def inference_pid_path(self) -> Path:
        return self.home / "inference" / "node.pid"

    def read_inference_pid(self) -> Optional[int]:
        try:
            pid = int(self.inference_pid_path.read_text().strip())
        except (OSError, ValueError):
            return None
        return pid if process_alive(pid) else None

    def inference_status(self) -> dict:
        try:
            data = json.loads((self.home / "inference" / "status.json").read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _stop_inference(self, wait: float = 0.0) -> None:
        pid = self.read_inference_pid()
        if pid is None:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and process_alive(pid):
            time.sleep(0.05)

    def _inference_args_path(self) -> Path:
        return self.home / "inference.args"

    def _restart_coordinator(self, settings: LauncherSettings) -> None:
        """The inference transport is fixed when the coordinator starts: restart ours to change it."""
        pid = self.read_coordinator_pid()
        if pid is None:
            self.last_error = ("Inference transport can only change when this app started the coordinator; "
                               "restart it with --inference-transport " + settings.transport + ".")
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and process_alive(pid):
            time.sleep(0.05)
        self.coordinator_pid_path.unlink(missing_ok=True)
        self._spawn(self.coordinator_argv(settings.transport), self.log_dir / "coordinator.log",
                    pid_writer=self.write_coordinator_pid)
        self.wait_health(self.proxy_url(settings))

    def read_coordinator_pid(self) -> Optional[int]:
        if not self.coordinator_pid_path.exists():
            return None
        try:
            pid = int(self.coordinator_pid_path.read_text().strip())
        except ValueError:
            return None
        if process_alive(pid):
            return pid
        self.coordinator_pid_path.unlink(missing_ok=True)
        return None

    def write_coordinator_pid(self, pid: int) -> None:
        self.coordinator_pid_path.write_text(str(pid) + "\n")

    def poll_health(self, url: str) -> Optional[dict]:
        if not url:
            return None
        try:
            r = self._http.get(f"{url.rstrip('/')}/health", timeout=health_timeout(url))
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, dict) else {"ok": True}
        except Exception:
            return None
        return None

    def wait_health(self, url: str, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.poll_health(url):
                return True
            time.sleep(0.2)
        return False

    def find_on_lan(self, timeout: float = 5.0) -> Optional[str]:
        found = self._discover(timeout)
        return normalize_url(found) if found else None

    def start(self, settings: LauncherSettings) -> StatusSnapshot:
        s = settings.clamp()
        self.save_settings(s)
        self.last_error = ""
        url = self.coordinator_url(s)
        if s.mode == "join" and not url:
            self.last_error = "Enter a coordinator URL, or find one on the LAN."
            raise LauncherError(self.last_error)
        if s.mode == "public" and not url:
            self.last_error = "Enter the public coordinator URL."
            raise LauncherError(self.last_error)
        if s.mode == "public" and not s.session_token:
            self.last_error = "Sign in first."
            raise LauncherError(self.last_error)

        want_coord = s.mode == "host"
        lend = s.mode == "join" or s.contribute
        want_agent = lend and s.training
        want_inference = lend and s.inference

        health = self.poll_health(url) if want_coord else None
        if want_coord and not health:
            self._spawn(self.coordinator_argv(s.transport), self.log_dir / "coordinator.log",
                        pid_writer=self.write_coordinator_pid)
            check = self.proxy_url(s) if s.mode == "host" else url
            if not (self.wait_health(check) or self.poll_health(url)):
                self.last_error = (
                    f"Coordinator started but is not answering {url}/health yet. "
                    f"Watch {self.log_dir / 'coordinator.log'}."
                )
        elif want_coord and health.get("inference_transport", "direct") != s.transport:
            self._restart_coordinator(s)

        agent_url = self.proxy_url(s) if s.mode == "host" else url
        if s.mode in ("join", "public") and not self.poll_health(url):
            self.last_error = f"No coordinator at {url}."
            raise LauncherError(self.last_error)
        if not want_agent and self._agent_running():
            request_stop(self.paths)
        if want_agent:
            token = s.session_token or os.environ.get("SLASHCOMPUTE_SESSION", "")
            argv = self.agent_argv(agent_url, s.gpu_percent, token, s.memory_gb)
            if self._agent_running() and self._read_agent_args() != argv:
                request_stop(self.paths)
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and self._agent_running():
                    time.sleep(0.05)
                if self._agent_running():
                    self.last_error = (
                        "Training agent is still stopping. Settings have not been applied; "
                        "start again after its current work finishes."
                    )
            if not self._agent_running():
                self._spawn(argv, self.log_dir / "agent.log")
                args_path = self._agent_args_path()
                args_path.touch(mode=0o600)
                args_path.chmod(0o600)
                args_path.write_text(json.dumps(argv))

        if want_inference:
            if s.mode == "join" and not self.poll_health(url):
                self.last_error = f"No coordinator at {url}."
                raise LauncherError(self.last_error)
            argv = self.inference_argv(agent_url, s)
            if self.read_inference_pid() is not None and self._read_inference_args() != argv:
                self._stop_inference(wait=10.0)   # settings changed: drain, then rejoin with the new ones
            if self.read_inference_pid() is None:
                self._spawn(argv, self.log_dir / "inference.log")
                self._inference_args_path().write_text(json.dumps(argv))
        else:
            self._stop_inference()

        return self.snapshot(s)

    def stop(self) -> StatusSnapshot:
        self.last_error = ""
        request_stop(self.paths)
        self._stop_inference()
        pid = self.read_coordinator_pid()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                self.coordinator_pid_path.unlink(missing_ok=True)
        return self.snapshot()

    def stop_agent(self) -> StatusSnapshot:
        """Stop contributing; a coordinator hosted here keeps running."""
        self.last_error = ""
        request_stop(self.paths)
        return self.snapshot()

    def my_node_id(self) -> Optional[str]:
        """This Mac's agent id, once it has registered. Never creates one."""
        try:
            return self.paths.node_id_file.read_text().strip() or None
        except OSError:
            return None

    def fetch_pool(self, url: str) -> PoolData:
        """Nodes, jobs and ledger from the coordinator, or empty when it is down."""
        if not url or not self.poll_health(url):
            return PoolData()
        return PoolData(online=True, nodes=self._get_list(url, "/nodes"),
                        jobs=self._get_list(url, "/jobs"), ledger=self._get_list(url, "/ledger"))

    def _get_list(self, url: str, path: str) -> list:
        try:
            r = self._http.get(f"{url.rstrip('/')}{path}", timeout=1.5)
            data = r.json() if r.status_code == 200 else []
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def snapshot(self, settings: Optional[LauncherSettings] = None) -> StatusSnapshot:
        s = settings.clamp() if settings is not None else self.load_settings()
        url = self.coordinator_url(s)
        health = self.poll_health(url) or {}
        if not health and s.mode == "host":
            health = self.poll_health(self.proxy_url(s)) or {}
        if health and self.last_error.startswith(UNREACHABLE_ERRORS):
            self.last_error = ""   # the coordinator answers now (e.g. after Connect fixed the address)
        agent = self.paths.read_status()
        agent_pid = self.paths.read_pid()
        agent_running = bool(agent_pid and process_alive(agent_pid))
        if agent_pid and not agent_running:
            self.paths.clear_pid()
            agent_pid = None
        inference_pid = self.read_inference_pid()
        mem_total, mem_free = self._memory()
        return StatusSnapshot(
            coordinator_up=bool(health),
            nodes=int(health.get("nodes", 0) or 0),
            jobs=int(health.get("jobs", 0) or 0),
            agent_running=agent_running,
            agent_status=str(agent.get("status", "") or ""),
            agent_job_id=agent.get("job_id"),
            coordinator_pid=self.read_coordinator_pid(),
            agent_pid=agent_pid if agent_running else None,
            lan_ip=self._lan_ip(),
            last_error=self.last_error,
            inference_running=inference_pid is not None,
            inference_pid=inference_pid,
            inference_status=self.inference_status() if inference_pid is not None else {},
            inference_nodes=int(health.get("inference_nodes", 0) or 0),
            inference_transport=str(health.get("inference_transport", "") or ""),
            inference_supported=supports_inference(health),
            memory_total_bytes=mem_total,
            memory_available_bytes=mem_free,
        )

    def _read_inference_args(self) -> list[str]:
        try:
            return json.loads(self._inference_args_path().read_text())
        except (OSError, ValueError):
            return []

    def _agent_args_path(self) -> Path:
        return self.home / "agent.args"

    def _read_agent_args(self) -> list[str]:
        try:
            return json.loads(self._agent_args_path().read_text())
        except (OSError, ValueError):
            return []

    def _agent_running(self) -> bool:
        pid = self.paths.read_pid()
        return bool(pid and process_alive(pid))

    def _spawn(self, argv: list[str], log_path: Path,
               pid_writer: Optional[Callable[[int], None]] = None) -> Any:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "ab")
        try:
            proc = self._popen(
                argv, stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True, env=os.environ.copy(),
            )
        except OSError as e:
            fh.close()
            self.last_error = f"Could not start process: {e}"
            raise LauncherError(self.last_error) from e
        if pid_writer is not None:
            pid_writer(int(proc.pid))
        return proc
