"""Engine interface: the node agent drives llama.cpp (or a simulation of it) through this.

Event stream from `complete()` (one dict per event):
  {'type': 'chunk', 'data': <OpenAI chat.completion.chunk>}
  {'type': 'final', 'timings': {...llama-server timings...}, 'usage': {...}, 'finish_reason': str}
An engine raises `EngineError` when the pipeline breaks (e.g. an RPC worker disappears), or with a `status`
when llama-server rejects the request itself (bad messages, prompt larger than the context).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator, Protocol

from slashcompute.inference.devices import Device, model_devices, parse_list_devices, tensor_split_for


class EngineError(RuntimeError):
    """`status` is set when the request itself was rejected (the HTTP status for the client): the pipeline
    is fine, and retrying the same request elsewhere would fail the same way."""

    def __init__(self, message: str, pipeline_broken: bool = True, status: int | None = None):
        super().__init__(message)
        self.pipeline_broken = pipeline_broken
        self.status = status


class Engine(Protocol):
    async def start_worker(self, pipeline_id: str, spec: dict) -> dict: ...
    async def stop_worker(self, pipeline_id: str) -> None: ...
    async def start_head(self, pipeline_id: str, spec: dict) -> dict: ...
    async def stop_head(self, pipeline_id: str) -> None: ...
    def complete(self, pipeline_id: str, body: dict) -> AsyncIterator[dict]: ...
    async def benchmark(self) -> dict: ...
    async def stop_all(self) -> None: ...


def reap_orphans(pid_file: Path) -> list[int]:
    """Kill llama.cpp processes a previous (crashed) agent left behind, as recorded in pid_file."""
    killed = []
    if not pid_file.exists():
        return killed
    with contextlib.suppress(ValueError, OSError):
        for pid in json.loads(pid_file.read_text()):
            comm = subprocess.run(['ps', '-p', str(pid), '-o', 'comm='], capture_output=True, text=True).stdout
            if 'llama-server' in comm or 'rpc-server' in comm:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGTERM)
                    killed.append(pid)
    pid_file.unlink(missing_ok=True)
    return killed


def head_layout(devices: list[Device], spec: dict, head_node_id: str) -> dict:
    """Map the planner's layer counts onto llama.cpp's reported device order."""
    counts = {w['endpoint']: w['n_layers'] for w in spec['workers']}
    vec = tensor_split_for(devices, counts, spec['head_layers'])
    ep_to_node = {w['endpoint']: w['node_id'] for w in spec['workers']}
    order = list(dict.fromkeys(ep_to_node[d.endpoint] for d in devices if d.is_rpc and d.endpoint in ep_to_node))
    order.append(head_node_id)
    return {
        'devices': [{'name': d.name, 'description': d.description, 'total_mib': d.total_mib}
                    for d in model_devices(devices)],
        'tensor_split': vec,
        'order': order,
    }


class LlamaCppRpcEngine:
    """The real engine: `rpc-server` on workers, `llama-server --rpc ... --tensor-split ...` on heads."""

    def __init__(self, node_id: str, llama_server: str, rpc_server: str, models_dir: str | list, bind_ip: str,
                 device: str | None = None, basket_model: str | None = None,
                 basket_ref: tuple[float, float] = (2000.0, 110.0), form_timeout: float = 600.0):
        from slashcompute.inference.node import head, rpc_worker
        self._head, self._rpc = head, rpc_worker
        self.node_id = node_id
        self.llama_server, self.rpc_server = llama_server, rpc_server
        dirs = [models_dir] if isinstance(models_dir, (str, Path)) else list(models_dir)
        self.models_dirs = [Path(d).expanduser() for d in dirs]
        self.models_dir = self.models_dirs[0]
        self.bind_ip = bind_ip
        self.device = device
        self.basket_model = basket_model
        self.basket_ref = basket_ref
        self.form_timeout = form_timeout
        self.workers: dict[str, asyncio.subprocess.Process] = {}
        self.heads: dict[str, tuple[asyncio.subprocess.Process, int]] = {}
        self._rpc_flags: set[str] | None = None
        self.pid_file: Path | None = None

    def _record_pids(self) -> None:
        if self.pid_file is None:
            return
        pids = [p.pid for p in self.workers.values()] + [p.pid for p, _ in self.heads.values()]
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(json.dumps(pids))

    def model_path(self, name: str) -> Path:
        """First models directory that has this file (downloads land in a separate directory)."""
        return next((d / name for d in self.models_dirs if (d / name).exists()), self.models_dir / name)

    async def start_worker(self, pipeline_id: str, spec: dict) -> dict:
        if self._rpc_flags is None:
            self._rpc_flags = await self._rpc.detect_flags(self.rpc_server)
        port = self._rpc.free_port(self.bind_ip)
        proc = await self._rpc.start(self.rpc_server, self.bind_ip, port, flags=self._rpc_flags,
                                     device=spec.get('device') or self.device,
                                     mem_mb=spec.get('mem_bytes', 0) // (1024 * 1024) or None)
        self.workers[pipeline_id] = proc
        self._record_pids()
        await self._rpc.wait_listening(self.bind_ip, port, proc, timeout=30)
        return {'endpoint': f'{self.bind_ip}:{port}'}

    async def stop_worker(self, pipeline_id: str) -> None:
        proc = self.workers.pop(pipeline_id, None)
        if proc:
            await self._rpc.terminate(proc)
            self._record_pids()

    async def start_head(self, pipeline_id: str, spec: dict) -> dict:
        model_path = self.model_path(spec['model'])
        if not model_path.exists():
            raise EngineError(f'model file not found on head: {model_path}', pipeline_broken=False)
        endpoints = [w['endpoint'] for w in spec['workers']]
        if endpoints:
            text = await self._head.list_devices(self.llama_server, endpoints)
            devices = parse_list_devices(text)
            layout = head_layout(devices, spec, self.node_id)
        else:
            layout = {'devices': [], 'tensor_split': [], 'order': [self.node_id]}
        port = self._head.free_port()
        t0 = time.monotonic()
        proc = await self._head.start_server(self.llama_server, str(model_path), endpoints, layout['tensor_split'],
                                             spec['ctx'], port)
        self.heads[pipeline_id] = (proc, port)
        self._record_pids()
        try:
            await self._head.wait_health(port, proc, timeout=self.form_timeout)
        except Exception:
            await self.stop_head(pipeline_id)
            raise
        return {**layout, 'load_seconds': time.monotonic() - t0}

    async def stop_head(self, pipeline_id: str) -> None:
        entry = self.heads.pop(pipeline_id, None)
        if entry:
            await self._rpc.terminate(entry[0])
            self._record_pids()

    async def complete(self, pipeline_id: str, body: dict) -> AsyncIterator[dict]:
        entry = self.heads.get(pipeline_id)
        if entry is None:
            raise EngineError(f'no head process for pipeline {pipeline_id}')
        proc, port = entry
        try:
            async for ev in self._head.stream_chat(port, body):
                yield ev
        except EngineError:
            raise
        except Exception as e:
            # An RPC worker that disappears makes llama-server abort or hang up: treat as broken.
            raise EngineError(f'llama-server failed: {e}', pipeline_broken=True) from e

    async def benchmark(self) -> dict:
        from slashcompute.inference.node.benchmark import run_benchmark
        return await run_benchmark(self.llama_server, self.model_path(self.basket_model or ''), *self.basket_ref)

    async def stop_all(self) -> None:
        for pid in list(self.heads):
            await self.stop_head(pid)
        for pid in list(self.workers):
            await self.stop_worker(pid)
