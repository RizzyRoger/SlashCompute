"""Coordinator <-> agent messaging. Agents only make outbound connections:
- commands wait in a per-node queue until the agent long-polls `GET /agent/commands`
- the agent posts each command's result back; `send()` awaits it
- token streams come back on `POST /agent/jobs/{id}/stream` and land in a per-job queue
"""
from __future__ import annotations

import asyncio
import uuid


class CommandFailed(RuntimeError):
    pass


class CommandBus:
    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}
        self.pending: dict[str, tuple[str, asyncio.Future]] = {}
        self.closing = asyncio.Event()

    def queue(self, node_id: str) -> asyncio.Queue:
        return self.queues.setdefault(node_id, asyncio.Queue())

    def post(self, node_id: str, kind: str, payload: dict) -> str:
        """Fire-and-forget command (its result is ignored)."""
        cid = uuid.uuid4().hex[:16]
        self.queue(node_id).put_nowait({'id': cid, 'kind': kind, 'payload': payload})
        return cid

    async def send(self, node_id: str, kind: str, payload: dict, timeout: float) -> dict:
        cid = uuid.uuid4().hex[:16]
        fut = asyncio.get_running_loop().create_future()
        self.pending[cid] = (node_id, fut)
        self.queue(node_id).put_nowait({'id': cid, 'kind': kind, 'payload': payload})
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as e:
            raise CommandFailed(f'{kind} on node {node_id} timed out after {timeout:.0f}s') from e
        finally:
            self.pending.pop(cid, None)

    async def poll(self, node_id: str, wait: float) -> list[dict]:
        """Wait up to `wait`s for commands; returns empty at once when the coordinator shuts down."""
        q = self.queue(node_id)
        get, closing = asyncio.ensure_future(q.get()), asyncio.ensure_future(self.closing.wait())
        try:
            await asyncio.wait((get, closing), timeout=wait, return_when=asyncio.FIRST_COMPLETED)
        finally:
            closing.cancel()
            got = get.done()
            if not got:
                get.cancel()  # a dangling get() would swallow the node's next command
        if not got:
            return []
        out = [get.result()]
        while not q.empty():
            out.append(q.get_nowait())
        return out

    def resolve(self, node_id: str, cid: str, ok: bool, result: dict | None, error: str | None) -> None:
        entry = self.pending.get(cid)
        if entry is None or entry[0] != node_id or entry[1].done():
            return
        if ok:
            entry[1].set_result(result or {})
        else:
            entry[1].set_exception(CommandFailed(error or 'command failed'))

    def fail_node(self, node_id: str, reason: str) -> None:
        """Node went offline: fail everything waiting on it and drop its queued commands."""
        for cid, (nid, fut) in list(self.pending.items()):
            if nid == node_id and not fut.done():
                fut.set_exception(CommandFailed(reason))
        self.queues.pop(node_id, None)

    def close(self) -> None:
        """Coordinator shutting down: release every agent's long-poll so the server can exit."""
        self.closing.set()


class JobStreams:
    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}

    def open(self, job_id: str) -> asyncio.Queue:
        return self.queues.setdefault(job_id, asyncio.Queue())

    def push(self, job_id: str, event: dict) -> bool:
        q = self.queues.get(job_id)
        if q is None:
            return False
        q.put_nowait(event)
        return True

    def fail(self, job_id: str, error: str, retryable: bool = True) -> None:
        self.push(job_id, {'type': 'error', 'error': error, 'retryable': retryable})

    def close(self, job_id: str) -> None:
        self.queues.pop(job_id, None)
