import asyncio
import socket

import mlx.core as mx
import pytest

from slashcompute.transport import (
    Frame, LinkClosed, LinkServer, LinkTimeout, MemoryLink, ResilientLink, TcpLink, connect, digest,
)
from slashcompute.transport.peer import KEEPALIVE_IDLE_S
from slashcompute.transport.serialization import decode_bytes, encode_bytes


def _tensors():
    return {
        "h": mx.random.normal((2, 5, 16)).astype(mx.bfloat16),
        "f16": mx.arange(12, dtype=mx.float16).reshape(3, 4),
        "ids": mx.array([[1, 2, 3]], dtype=mx.int32),
        "mask": mx.array([True, False]),
    }


def _assert_same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        assert a[k].shape == b[k].shape, k
        assert mx.array_equal(a[k], b[k]).item(), k


def test_bytes_roundtrip_all_dtypes():
    t = _tensors()
    out = decode_bytes(encode_bytes(Frame("fwd", {"mb": 3, "step": 7}, t)))
    assert out.kind == "fwd" and out.meta == {"mb": 3, "step": 7}
    _assert_same(t, out.tensors)


def test_digest_stable_and_sensitive():
    a = mx.ones((4, 4), dtype=mx.bfloat16)
    assert digest(a) == digest(mx.ones((4, 4), dtype=mx.bfloat16))
    assert digest(a) != digest(a.astype(mx.float16))
    assert digest(a) != digest(a * 2)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.int32, mx.bool_])
@pytest.mark.parametrize("shape", [(0,), (0, 4), (2, 0, 3)])
def test_empty_tensor_roundtrip(shape, dtype):
    t = {"empty": mx.zeros(shape, dtype=dtype), "ids": mx.array([[1, 2]], dtype=mx.int32)}
    out = decode_bytes(encode_bytes(Frame("x", {}, t)))
    _assert_same(t, out.tensors)


def test_digest_empty_tensor():
    a = mx.zeros((0, 4), dtype=mx.bfloat16)
    assert digest(a) == digest(mx.zeros((0, 4), dtype=mx.bfloat16))
    assert digest(a) != digest(mx.zeros((4, 0), dtype=mx.bfloat16))
    assert digest(a) != digest(a.astype(mx.float16))


async def test_tcp_link_roundtrip_and_hello():
    server = await LinkServer("127.0.0.1", 0, {"job": "j1", "epoch": 2}).start()
    up_task = asyncio.create_task(connect("127.0.0.1", server.port, {"job": "j1", "epoch": 2}))
    down = await server.accept(5)
    up = await up_task

    t = _tensors()
    await up.send(Frame("fwd", {"mb": 0}, t))
    big = mx.random.normal((8, 256, 512))  # ~4 MB, exceeds socket buffers
    await down.send(Frame("bwd", {"mb": 0}, {"g": big}))

    got = await down.recv(5)
    _assert_same(t, got.tensors)
    back = await up.recv(5)
    assert mx.allclose(back.tensors["g"], big).item()

    await up.close()
    with pytest.raises(LinkClosed):
        await down.recv(5)
    await down.close()
    await server.close()


async def test_tcp_hello_mismatch_rejected():
    server = await LinkServer("127.0.0.1", 0, {"job": "j1", "epoch": 2}).start()
    with pytest.raises(LinkClosed, match="listening for epoch 2, got 1"):
        await connect("127.0.0.1", server.port, {"job": "j1", "epoch": 1}, timeout=0.5)
    await server.close()


async def test_connect_waits_out_previous_epoch_listener():
    # The downstream still has epoch 1's listener on its data port when the
    # epoch-2 upstream dials; the dialer must keep trying until epoch 2 binds.
    old = await LinkServer("127.0.0.1", 0, {"job_id": "j1", "epoch": 1}).start()
    port = old.port
    up_task = asyncio.create_task(
        connect("127.0.0.1", port, {"job_id": "j1", "epoch": 2}, timeout=10, retry_interval=0.05))
    await asyncio.sleep(0.5)
    assert not up_task.done()
    await old.close()
    new = await LinkServer("127.0.0.1", port, {"job_id": "j1", "epoch": 2}).start()
    up = await asyncio.wait_for(up_task, 10)
    down = await new.accept(5)
    await up.send(Frame("ping", {"n": 1}))
    assert (await down.recv(5)).meta == {"n": 1}
    await up.close()
    await down.close()
    await new.close()


async def test_connect_retries_legacy_reject_without_reason():
    # An older peer rejects with an empty meta; the dialer must not depend on a reason.
    async def legacy(reader, writer):
        link = TcpLink(reader, writer)
        await link.recv(5)
        await link.send(Frame("hello_reject", {}))
        await link.close()

    server = await asyncio.start_server(legacy, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    t0 = asyncio.get_running_loop().time()
    with pytest.raises(LinkClosed, match=r"peer rejected hello: \{\}"):
        await connect("127.0.0.1", port, {"job_id": "j1", "epoch": 2}, timeout=0.5,
                      retry_interval=0.05)
    assert asyncio.get_running_loop().time() - t0 >= 0.5
    server.close()
    await server.wait_closed()


async def test_link_server_close_with_unclaimed_link():
    # A stage cancelled after its upstream connected but before it called
    # accept() must still free its listener rather than hang in close().
    server = await LinkServer("127.0.0.1", 0, {"job_id": "j1", "epoch": 1}).start()
    up = await connect("127.0.0.1", server.port, {"job_id": "j1", "epoch": 1}, timeout=5)
    await asyncio.wait_for(server.close(), 5)
    with pytest.raises(LinkClosed):
        await up.recv(5)
    await up.close()


async def test_memory_link():
    a, b = MemoryLink.pair()
    await a.send(Frame("x", {"v": 1}))
    assert (await b.recv(1)).meta == {"v": 1}
    await a.close()
    with pytest.raises(LinkClosed):
        await b.recv(1)


async def test_recv_timeout_is_a_link_timeout_and_keeps_queued_frames():
    a, b = MemoryLink.pair()
    with pytest.raises(LinkTimeout, match="nothing from the peer in 0.05s"):
        await b.recv(0.05)
    await a.send(Frame("x", {"v": 1}))
    assert (await b.recv(1)).meta == {"v": 1}


async def test_tcp_links_use_keepalive():
    # A peer that loses power or Wi-Fi never sends a FIN; keepalive is what notices.
    server = await LinkServer("127.0.0.1", 0, {"job": "j1", "epoch": 1}).start()
    up = await connect("127.0.0.1", server.port, {"job": "j1", "epoch": 1}, timeout=5)
    down = await server.accept(5)
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    for link in (up, down):
        sock = link._writer.get_extra_info("socket")
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        if idle is not None:
            assert sock.getsockopt(socket.IPPROTO_TCP, idle) == KEEPALIVE_IDLE_S
    await up.close()
    await down.close()
    await server.close()


async def test_send_times_out_when_the_peer_stops_reading():
    release = asyncio.Event()

    async def never_reads(reader, writer):
        await release.wait()
        writer.close()

    server = await asyncio.start_server(never_reads, "127.0.0.1", 0)
    reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
    link = TcpLink(reader, writer, send_timeout=0.3)
    big = mx.zeros((8, 1024, 1024), dtype=mx.float32)  # 32 MB, far beyond the socket buffers
    with pytest.raises(LinkTimeout, match="took no data for 0.3s"):
        for _ in range(4):
            await link.send(Frame("fwd", {}, {"h": big}))
    with pytest.raises(LinkClosed):  # half a frame may be on the wire: the stream is done
        await link.send(Frame("x", {}))
    await asyncio.wait_for(link.close(), 10)  # must not wait for the stuck peer to read
    release.set()
    server.close()
    await server.wait_closed()


async def _resilient_pair(window=5.0):
    hello = {"job_id": "j", "epoch": 1}
    server = await LinkServer("127.0.0.1", 0, hello, resume_window=window).start()
    up = await connect("127.0.0.1", server.port, hello, timeout=5, retry_interval=0.05,
                       resume_window=window)
    down = await server.accept(5)
    assert isinstance(up, ResilientLink) and isinstance(down, ResilientLink)
    return server, up, down


async def test_resilient_link_retransmits_across_dropped_connections():
    server, up, down = await _resilient_pair()
    h = mx.random.normal((4, 64, 256))
    got_down, got_up = [], []

    async def send_all(link, kind, cut_at):
        for i in range(40):
            if i in cut_at:
                while link._ready is not link._tcp or link._tcp._writer.is_closing():
                    await asyncio.sleep(0.01)  # let the previous drop recover first
                link._tcp.abort()  # the network drops mid-stream, frames in flight are lost
            await link.send(Frame(kind, {"i": i}, {"h": h + i}))

    async def recv_all(link, out):
        for _ in range(40):
            f = await link.recv(10)
            assert mx.array_equal(f.tensors["h"], h + f.meta["i"]).item()
            out.append(f.meta["i"])

    await asyncio.wait_for(asyncio.gather(
        send_all(up, "fwd", {12, 30}), send_all(down, "bwd", {21}),
        recv_all(down, got_down), recv_all(up, got_up)), 30)
    assert got_down == list(range(40)) and got_up == list(range(40))  # each once, in order
    assert up.reconnects >= 3
    await up.close()
    await down.close()
    await server.close()


async def test_resilient_link_gives_up_after_its_window():
    server, up, down = await _resilient_pair(window=0.5)
    server._server.close()  # the downstream stops listening: redials are refused
    down._tcp.abort()
    t0 = asyncio.get_running_loop().time()
    for link in (up, down):
        with pytest.raises(LinkClosed) as e:
            await link.recv(10)
        assert not isinstance(e.value, LinkTimeout)
    assert asyncio.get_running_loop().time() - t0 < 5
    await up.close()
    await down.close()
    await server.close()


async def test_resilient_close_is_final_for_the_peer():
    server, up, down = await _resilient_pair(window=30)
    await up.send(Frame("done", {"step": 3}))
    await up.close()
    assert (await down.recv(5)).kind == "done"
    t0 = asyncio.get_running_loop().time()
    with pytest.raises(LinkClosed):
        await down.recv(10)
    assert asyncio.get_running_loop().time() - t0 < 2  # not after the 30 s resume window
    assert down.reconnects == 0
    await down.close()
    await server.close()


async def test_resume_hello_must_match_the_stage():
    server, up, down = await _resilient_pair()
    with pytest.raises(LinkClosed, match=r"does not match this stage \(job_id\)"):
        await connect("127.0.0.1", server.port, {"job_id": "other", "epoch": 1, "resume": 0},
                      timeout=0.3, retry_interval=0.05)
    await up.send(Frame("x", {"v": 1}))
    assert (await down.recv(5)).meta == {"v": 1}
    await up.close()
    await down.close()
    await server.close()


@pytest.mark.parametrize("dial_window,listen_window", [(5.0, None), (None, 5.0)])
async def test_resilience_needs_both_ends(dial_window, listen_window):
    # Agents from before resilient links answer (or offer) a plain hello.
    hello = {"job_id": "j", "epoch": 1}
    server = await LinkServer("127.0.0.1", 0, hello, resume_window=listen_window).start()
    up = await connect("127.0.0.1", server.port, hello, timeout=5, resume_window=dial_window)
    down = await server.accept(5)
    assert type(up) is TcpLink and type(down) is TcpLink
    await up.send(Frame("x", {"v": 1}))
    assert (await down.recv(5)).meta == {"v": 1}
    await up.close()
    await down.close()
    await server.close()
