"""The pipeline head: `llama-server` with `--rpc` workers, bound to localhost only.

Requests reach it through the agent's outbound connection to the coordinator; nothing on the head
listens on a public interface.
"""
from __future__ import annotations

import asyncio
import json
import re
import socket
from typing import AsyncIterator

import httpx

from slashcompute.inference.node.engine import EngineError

_VERSION = re.compile(r'build[ :]*(\d+)[^)]*?commit[ :]*([0-9a-f]+)', re.I)
_OLD_VERSION = re.compile(r'version:\s*(\d+)\s*\(([0-9a-f]+)\)')


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def parse_build(version_text: str) -> str:
    """'version: 0.5.0-dev (build 11160, commit a3c12db9d)' -> 'b11160-a3c12db9d'."""
    m = _VERSION.search(version_text) or _OLD_VERSION.search(version_text)
    return f'b{m.group(1)}-{m.group(2)}' if m else 'unknown'


async def build_string(llama_server: str) -> str:
    proc = await asyncio.create_subprocess_exec(llama_server, '--version', stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return parse_build(out.decode(errors='replace'))


async def list_devices(llama_server: str, rpc_endpoints: list[str]) -> str:
    args = [llama_server]
    if rpc_endpoints:
        args += ['--rpc', ','.join(rpc_endpoints)]
    proc = await asyncio.create_subprocess_exec(*args, '--list-devices', stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    out, _ = await asyncio.wait_for(proc.communicate(), 60)
    return out.decode(errors='replace')


def server_command(llama_server: str, model_path: str, rpc_endpoints: list[str], tensor_split: list[int],
                   ctx: int, port: int, extra_args: tuple[str, ...] = ()) -> list[str]:
    cmd = [llama_server, '-m', model_path, '-ngl', '999', '-c', str(ctx), '-np', '1',
           '--host', '127.0.0.1', '--port', str(port), *extra_args]
    if rpc_endpoints:
        cmd += ['--rpc', ','.join(rpc_endpoints), '--tensor-split', ','.join(str(x) for x in tensor_split)]
    return cmd


async def start_server(llama_server: str, model_path: str, rpc_endpoints: list[str], tensor_split: list[int],
                       ctx: int, port: int, log_path: str | None = None,
                       extra_args: tuple[str, ...] = ()) -> asyncio.subprocess.Process:
    out = open(log_path, 'ab') if log_path else asyncio.subprocess.DEVNULL
    return await asyncio.create_subprocess_exec(
        *server_command(llama_server, model_path, rpc_endpoints, tensor_split, ctx, port, extra_args),
        stdout=out, stderr=asyncio.subprocess.STDOUT)


async def wait_health(port: int, proc: asyncio.subprocess.Process, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with httpx.AsyncClient(timeout=2) as client:
        while loop.time() < deadline:
            if proc.returncode is not None:
                raise RuntimeError(f'llama-server exited with code {proc.returncode} while loading')
            try:
                r = await client.get(f'http://127.0.0.1:{port}/health')
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError('llama-server did not become healthy in time')


def request_error_status(status: int, text: str) -> int | None:
    """The status for the client when llama-server rejected the request itself, else None (server trouble).

    llama-server answers 4xx for malformed requests and prompts larger than the context, and 500 when the
    model's chat template raises (e.g. no user message)."""
    if 400 <= status < 500:
        return status
    if status == 500 and ('Jinja' in text or 'template' in text):
        return 400
    return None


def error_message(text: str) -> str:
    """llama-server's error message; for a template error, just its last line ('Error: Jinja Exception: ...')."""
    try:
        msg = str(json.loads(text)['error']['message']).strip()
    except (ValueError, KeyError, TypeError):
        return text[:300]
    return (msg.splitlines() or [''])[-1][:300]


async def stream_chat(port: int, body: dict) -> AsyncIterator[dict]:
    """Relay an OpenAI chat completion from the local llama-server as engine events."""
    req = {**body, 'stream': True, 'stream_options': {'include_usage': True}}
    timings, usage, finish = None, None, None
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5)) as client:
        async with client.stream('POST', f'http://127.0.0.1:{port}/v1/chat/completions', json=req) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode(errors='replace')
                status = request_error_status(r.status_code, text)
                if status is not None:
                    raise EngineError(f'invalid request: {error_message(text)}', pipeline_broken=False, status=status)
                raise RuntimeError(f'llama-server HTTP {r.status_code}: {text[:300]}')
            async for line in r.aiter_lines():
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                chunk = json.loads(payload)
                timings = chunk.pop('timings', None) or timings
                usage = chunk.get('usage') or usage
                for c in chunk.get('choices') or []:
                    finish = c.get('finish_reason') or finish
                if chunk.get('choices'):
                    yield {'type': 'chunk', 'data': chunk}
    if timings is None:
        raise RuntimeError('llama-server response had no timings')
    yield {'type': 'final', 'timings': timings, 'usage': usage or {}, 'finish_reason': finish or 'stop'}
