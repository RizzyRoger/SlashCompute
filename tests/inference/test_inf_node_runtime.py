import asyncio
import json
import subprocess

import pytest

from slashcompute.inference.node import head
from slashcompute.inference.node.engine import EngineError, reap_orphans


def test_reaper_only_kills_llama_processes(tmp_path):
    other = subprocess.Popen(['sleep', '30'])
    try:
        pid_file = tmp_path / 'node.pids'
        pid_file.write_text(json.dumps([other.pid, 999999]))
        assert reap_orphans(pid_file) == []      # not llama-server / rpc-server: left alone
        assert other.poll() is None
        assert not pid_file.exists()
    finally:
        other.kill()


# Error bodies as llama-server (b11160) returns them.
JINJA_500 = {'error': {'code': 500, 'type': 'server_error', 'message': (
    "\n------------\nWhile executing CallExpression at line 43, column 24 in source:\n... "
    "raise_exception('No messages provided.') ...\nError: Jinja Exception: No messages provided.")}}
CTX_400 = {'error': {'code': 400, 'type': 'exceed_context_size_error', 'n_prompt_tokens': 20010, 'n_ctx': 4096,
                     'message': 'request (20010 tokens) exceeds the available context size (4096 tokens), '
                                'try increasing it'}}
LOADING_503 = {'error': {'code': 503, 'type': 'unavailable_error', 'message': 'Loading model'}}
PORT = 9982


async def stub_llama_server(status: int, body: dict):
    payload = json.dumps(body).encode()

    async def handle(reader, writer):
        headers = (await reader.readuntil(b'\r\n\r\n')).decode().split('\r\n')
        length = next((int(h.split(':')[1]) for h in headers if h.lower().startswith('content-length')), 0)
        await reader.readexactly(length)
        writer.write(f'HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n'
                     'Connection: close\r\n\r\n'.encode() + payload)
        await writer.drain()
        writer.close()

    return await asyncio.start_server(handle, '127.0.0.1', PORT)


async def run_chat(status: int, body: dict) -> BaseException:
    server = await stub_llama_server(status, body)
    try:
        with pytest.raises(RuntimeError) as e:
            async for _ in head.stream_chat(PORT, {'messages': []}):
                pass
        return e.value
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize('status,body,message', [(500, JINJA_500, 'No messages provided'),
                                                 (400, CTX_400, 'exceeds the available context size')])
async def test_llama_server_request_errors_are_client_errors_not_broken_pipelines(status, body, message):
    err = await run_chat(status, body)
    assert isinstance(err, EngineError) and err.status == 400 and err.pipeline_broken is False
    assert message in str(err)


async def test_llama_server_failures_still_break_the_pipeline():
    err = await run_chat(503, LOADING_503)
    assert getattr(err, 'status', None) is None  # engine wraps it as EngineError(pipeline_broken=True)
