"""Live view of connected nodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel

from slashcompute.common.protocol import DeviceProfile, PeerAddr, Register
from slashcompute.common.reliable import Inbox, Outbox

SendFn = Callable[[BaseModel], Awaitable[None]]
CloseFn = Callable[[], Awaitable[None]]


@dataclass
class Assignment:
    job_id: str
    epoch: int
    stage_idx: int


@dataclass
class NodeState:
    node_id: str
    name: str
    device: DeviceProfile
    data_host: str
    data_port: int
    gpu_percent: int
    send: SendFn
    last_heartbeat: float = field(default_factory=time.monotonic)
    status: str = "idle"
    draining: bool = False
    drain_deadline: Optional[float] = None
    assignment: Optional[Assignment] = None
    verifying: Optional[str] = None  # verification id
    canary_passed: Optional[bool] = None
    last_canary: float = 0.0
    user_id: Optional[str] = None
    close: Optional[CloseFn] = None  # closes the current WebSocket
    # Reliable session (agents that send Register.session_id): messages are numbered and
    # acknowledged, and a reconnect within reconnect_grace_s keeps the node's place.
    session_id: Optional[str] = None
    outbox: Outbox = field(default_factory=Outbox)
    inbox: Inbox = field(default_factory=Inbox)
    connected: bool = True
    disconnected_at: Optional[float] = None

    @property
    def reliable(self) -> bool:
        return self.session_id is not None

    @property
    def peer(self) -> PeerAddr:
        return PeerAddr(node_id=self.node_id, host=self.data_host, port=self.data_port)

    @property
    def schedulable(self) -> bool:
        return (self.connected and not self.draining and self.assignment is None
                and self.verifying is None and self.canary_passed is not False)


class Registry:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeState] = {}

    def register(self, msg: Register, send: SendFn, close: Optional[CloseFn] = None) -> NodeState:
        state = NodeState(node_id=msg.node_id, name=msg.name, device=msg.device,
                          data_host=msg.data_host, data_port=msg.data_port,
                          gpu_percent=msg.gpu_percent, send=send, close=close,
                          session_id=msg.session_id)
        self.nodes[msg.node_id] = state
        return state

    def get(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.get(node_id)

    def remove(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.pop(node_id, None)

    def heartbeat(self, node_id: str, status: str) -> None:
        n = self.nodes.get(node_id)
        if n is not None:
            n.last_heartbeat = time.monotonic()
            n.status = status

    def expired(self, timeout_s: float) -> list[NodeState]:
        """Connected nodes whose heartbeats stopped (disconnected ones wait out their grace)."""
        cutoff = time.monotonic() - timeout_s
        return [n for n in self.nodes.values() if n.connected and n.last_heartbeat < cutoff]

    def away(self, grace_s: float) -> list[NodeState]:
        """Disconnected nodes that did not come back within ``grace_s``."""
        cutoff = time.monotonic() - grace_s
        return [n for n in self.nodes.values()
                if not n.connected and n.disconnected_at is not None and n.disconnected_at < cutoff]

    def schedulable(self) -> list[NodeState]:
        return [n for n in self.nodes.values() if n.schedulable]
