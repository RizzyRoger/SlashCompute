import asyncio

import mlx.core as mx
import pytest

from slashcompute.transport import Frame, LinkClosed, LinkServer, MemoryLink, connect, digest
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
    with pytest.raises(LinkClosed):
        await connect("127.0.0.1", server.port, {"job": "j1", "epoch": 1}, timeout=5)
    await server.close()


async def test_memory_link():
    a, b = MemoryLink.pair()
    await a.send(Frame("x", {"v": 1}))
    assert (await b.recv(1)).meta == {"v": 1}
    await a.close()
    with pytest.raises(LinkClosed):
        await b.recv(1)
