"""Reliable delivery of control messages across WebSocket reconnects.

When both ends support it (the agent sends ``Register.session_id`` and the
coordinator answers ``Welcome.session``), every message except handshakes,
heartbeats and acks carries a sequence number. The sender keeps each one in its
``Outbox`` until the peer acknowledges it, and after a reconnect replays what is
still unacknowledged. The receiver's ``Inbox`` handles each number once, so a
replayed message that already arrived is acknowledged again but not acted on.
"""

from __future__ import annotations

from collections import OrderedDict

from slashcompute.common.protocol import Ack, Heartbeat, Msg, Register, Welcome

_UNSEQUENCED = (Register, Welcome, Heartbeat, Ack)


def is_sequenced(msg: Msg) -> bool:
    return not isinstance(msg, _UNSEQUENCED)


class Outbox:
    def __init__(self) -> None:
        self._next = 1
        self._pending: OrderedDict[int, Msg] = OrderedDict()

    def stamp(self, msg: Msg) -> Msg:
        """Number a copy of ``msg`` (the original may be sent to other nodes) and keep it."""
        msg = msg.model_copy(update={"seq": self._next})
        self._pending[self._next] = msg
        self._next += 1
        return msg

    def ack(self, upto: int) -> None:
        while self._pending and next(iter(self._pending)) <= upto:
            self._pending.popitem(last=False)

    def pending(self) -> list[Msg]:
        return list(self._pending.values())

    def __len__(self) -> int:
        return len(self._pending)


class Inbox:
    def __init__(self) -> None:
        self.last = 0  # highest sequence number handled

    def accept(self, seq: int) -> bool:
        """True the first time ``seq`` arrives; False for a replay already handled."""
        if seq <= self.last:
            return False
        self.last = seq
        return True
