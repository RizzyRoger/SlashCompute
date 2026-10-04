"""Wire format for tensor frames.

    [u32 header_len][msgpack header][tensor 0 bytes][tensor 1 bytes]...

The header carries a ``kind`` string, a free-form ``meta`` dict and, per
tensor, its name, dtype, shape and byte length. Tensor payloads are written
straight from the array buffers.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import msgpack
import numpy as np

_HDR = struct.Struct("!I")

# MLX dtypes numpy can't represent travel as same-width unsigned ints.
_VIEW_AS = {"bfloat16": ("uint16", mx.uint16)}

_MX_DTYPES = {
    "float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16,
    "int32": mx.int32, "int64": mx.int64, "uint32": mx.uint32, "uint16": mx.uint16,
    "uint8": mx.uint8, "int8": mx.int8, "bool": mx.bool_,
}


@dataclass
class Frame:
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, mx.array] = field(default_factory=dict)


def dtype_name(a: mx.array) -> str:
    return str(a.dtype).removeprefix("mlx.core.")


def to_numpy(a: mx.array) -> np.ndarray:
    name = dtype_name(a)
    if name in _VIEW_AS:
        a = a.view(_VIEW_AS[name][1])
    return np.asarray(a)


def from_numpy(arr: np.ndarray, name: str) -> mx.array:
    out = mx.array(arr)
    if name in _VIEW_AS:
        out = out.view(_MX_DTYPES[name])
    return out


def _raw(arr: np.ndarray) -> memoryview | bytes:
    """Flat byte view of a C-contiguous array (memoryview can't cast empty N-d)."""
    return memoryview(arr).cast("B") if arr.size else b""


def encode(frame: Frame) -> list[bytes | memoryview]:
    """Return the chunks to write, in order. Evaluates tensors first."""
    if frame.tensors:
        mx.eval(*frame.tensors.values())
    specs, buffers = [], []
    for name, a in frame.tensors.items():
        arr = np.ascontiguousarray(to_numpy(a))
        specs.append({"name": name, "dtype": dtype_name(a), "shape": list(a.shape),
                      "nbytes": arr.nbytes})
        buffers.append(_raw(arr))
    header = msgpack.packb({"kind": frame.kind, "meta": frame.meta, "tensors": specs},
                           use_bin_type=True)
    return [_HDR.pack(len(header)), header, *buffers]


def decode_header(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=False)


def tensor_from_bytes(spec: dict, buf: bytes) -> mx.array:
    name = spec["dtype"]
    np_dtype = _VIEW_AS[name][0] if name in _VIEW_AS else name
    arr = np.frombuffer(buf, dtype=np_dtype).reshape(spec["shape"])
    return from_numpy(arr, name)


def encode_bytes(frame: Frame) -> bytes:
    return b"".join(bytes(c) for c in encode(frame))


def decode_bytes(data: bytes) -> Frame:
    (hlen,) = _HDR.unpack_from(data, 0)
    off = _HDR.size
    header = decode_header(data[off:off + hlen])
    off += hlen
    tensors = {}
    for spec in header["tensors"]:
        tensors[spec["name"]] = tensor_from_bytes(spec, data[off:off + spec["nbytes"]])
        off += spec["nbytes"]
    return Frame(header["kind"], header["meta"], tensors)


def digest(*arrays: mx.array) -> str:
    h = hashlib.sha256()
    for a in arrays:
        mx.eval(a)
        arr = np.ascontiguousarray(to_numpy(a))
        h.update(dtype_name(a).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(_raw(arr))
    return h.hexdigest()
