"""On-disk agent state: node id, pid, status, per-job working directories."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional


def resolve_data_host(localhost: bool) -> str:
    """Address peers should dial. 127.0.0.1 is only for same-machine clusters."""
    if localhost:
        return "127.0.0.1"
    from slashcompute.common.discovery import lan_ip

    return lan_ip()


def claim_data_port(preferred: int, host: str = "0.0.0.0") -> int:
    """Use ``preferred`` if it is free; otherwise bind an ephemeral port.

    A leftover agent on 9700 would otherwise keep the new worker from listening,
    and peers that dial 9700 get that stale listener's hello_reject.
    """
    import socket

    for port in (int(preferred), 0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return int(sock.getsockname()[1])
        except OSError:
            continue
        finally:
            sock.close()
    raise OSError(f"no free data port (preferred {preferred})")


class AgentPaths:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.root = self.home / "agent"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = self.root / "jobs"
        self.jobs.mkdir(exist_ok=True)
        self.node_id_file = self.root / "node_id"
        self.pid_file = self.root / "agent.pid"
        self.status_file = self.root / "status.json"

    def node_id(self) -> str:
        if self.node_id_file.exists():
            return self.node_id_file.read_text().strip()
        nid = uuid.uuid4().hex
        self.node_id_file.write_text(nid + "\n")
        return nid

    def job_dir(self, job_id: str, epoch: int) -> Path:
        d = self.jobs / job_id / f"epoch{epoch}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_pid(self) -> None:
        self.pid_file.write_text(str(os.getpid()) + "\n")

    def clear_pid(self) -> None:
        try:
            self.pid_file.unlink()
        except FileNotFoundError:
            pass

    def read_pid(self) -> Optional[int]:
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except ValueError:
            return None

    def write_status(self, **fields) -> None:
        cur = self.read_status()
        cur.update(fields)
        self.status_file.write_text(json.dumps(cur, indent=2) + "\n")

    def read_status(self) -> dict:
        if not self.status_file.exists():
            return {}
        try:
            return json.loads(self.status_file.read_text())
        except json.JSONDecodeError:
            return {}
