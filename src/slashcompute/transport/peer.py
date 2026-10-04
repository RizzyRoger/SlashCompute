"""Direct TCP links between neighbouring pipeline stages.

Each link is one TCP connection. A background task reads frames into a queue
so a stage can keep receiving while its own compute or sends are in flight.
The downstream stage listens on its data port and the upstream stage dials
it. A ``hello`` frame binds the connection to a (job, epoch) so stale stages
can't cross-talk.

Real LANs drop packets, sleep laptops and lose Wi-Fi, so nothing here waits
forever: sockets use TCP keepalive (a vanished peer is noticed in about
``KEEPALIVE_IDLE_S + KEEPALIVE_INTERVAL_S * KEEPALIVE_COUNT`` seconds even when
no data is moving), and receives, sends and dials take timeouts.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Optional

from slashcompute.transport.serialization import (
    Frame, _HDR, decode_header, encode, tensor_from_bytes,
)

log = logging.getLogger(__name__)

_CLOSED = object()

# The peer's kernel answers keepalive probes, so a stage busy in MLX compute
# never trips them; only a peer that is gone (or a path that is cut) does.
KEEPALIVE_IDLE_S = 15
KEEPALIVE_INTERVAL_S = 5
KEEPALIVE_COUNT = 4
# One dial attempt; a silently dropped SYN would otherwise block for the OS
# connect timeout (about 75 s on macOS) before the overall deadline is checked.
CONNECT_ATTEMPT_S = 10.0
CLOSE_TIMEOUT_S = 5.0


class LinkClosed(ConnectionError):
    pass


class LinkTimeout(LinkClosed):
    """The peer sent nothing, or took nothing, for longer than the link allows."""


def _enable_keepalive(sock) -> None:
    if sock is None:
        return
    opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    # TCP_KEEPIDLE on Linux; macOS calls the same option TCP_KEEPALIVE.
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    for name, opt, val in (("idle", idle, KEEPALIVE_IDLE_S),
                           ("interval", getattr(socket, "TCP_KEEPINTVL", None), KEEPALIVE_INTERVAL_S),
                           ("count", getattr(socket, "TCP_KEEPCNT", None), KEEPALIVE_COUNT)):
        if opt is not None:
            opts.append((socket.IPPROTO_TCP, opt, val))
    try:
        for level, opt, val in opts:
            sock.setsockopt(level, opt, val)
    except OSError as e:
        log.debug("could not enable TCP keepalive: %s", e)


class Link:
    """Bidirectional frame channel. Subclasses provide the byte transport."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self.bytes_sent = 0
        self.bytes_received = 0

    async def send(self, frame: Frame) -> None:
        raise NotImplementedError

    async def recv(self, timeout: Optional[float] = None) -> Frame:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            raise LinkTimeout(f"nothing from the peer in {timeout:g}s") from None
        if item is _CLOSED:
            self._queue.put_nowait(_CLOSED)
            raise LinkClosed("peer link closed")
        if isinstance(item, BaseException):
            self._queue.put_nowait(item)
            raise LinkClosed(f"peer link failed: {item!r}") from item
        return item

    async def close(self) -> None:
        raise NotImplementedError


class TcpLink(Link):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 send_timeout: Optional[float] = None) -> None:
        super().__init__()
        self._reader = reader
        self._writer = writer
        self.send_timeout = send_timeout
        self._send_lock = asyncio.Lock()
        _enable_keepalive(writer.get_extra_info("socket"))
        self._task = asyncio.create_task(self._read_loop())

    @property
    def peername(self):
        return self._writer.get_extra_info("peername")

    async def _read_frame(self) -> Frame:
        (hlen,) = _HDR.unpack(await self._reader.readexactly(_HDR.size))
        header = decode_header(await self._reader.readexactly(hlen))
        tensors = {}
        for spec in header["tensors"]:
            buf = await self._reader.readexactly(spec["nbytes"])
            tensors[spec["name"]] = tensor_from_bytes(spec, buf)
            self.bytes_received += spec["nbytes"]
        self.bytes_received += _HDR.size + hlen
        return Frame(header["kind"], header["meta"], tensors)

    async def _read_loop(self) -> None:
        try:
            while True:
                self._queue.put_nowait(await self._read_frame())
        except (asyncio.IncompleteReadError, OSError):  # EOF, reset, keepalive ETIMEDOUT
            self._queue.put_nowait(_CLOSED)
        except asyncio.CancelledError:
            self._queue.put_nowait(_CLOSED)
        except Exception as e:  # malformed frame
            log.exception("peer link read failed")
            self._queue.put_nowait(e)

    async def send(self, frame: Frame) -> None:
        chunks = encode(frame)
        async with self._send_lock:
            if self._writer.is_closing():
                raise LinkClosed("peer link closed")
            try:
                for c in chunks:
                    self._writer.write(c)
                    self.bytes_sent += len(c)
                await asyncio.wait_for(self._writer.drain(), self.send_timeout)
            except TimeoutError:
                # Part of a frame may be in flight, so this stream can't carry another;
                # abort rather than close, which would wait for the stuck peer to read.
                self._writer.transport.abort()
                raise LinkTimeout(f"the peer took no data for {self.send_timeout:g}s") from None
            except OSError as e:
                raise LinkClosed(str(e)) from e

    async def close(self) -> None:
        self._task.cancel()
        self._writer.close()
        try:
            # A graceful close flushes what is buffered first; a peer that stopped
            # reading would make that wait forever.
            await asyncio.wait_for(self._writer.wait_closed(), CLOSE_TIMEOUT_S)
        except TimeoutError:
            self._writer.transport.abort()
        except Exception:
            pass


class MemoryLink(Link):
    """In-process link, used by the single-process pipeline and tests."""

    def __init__(self) -> None:
        super().__init__()
        self.other: Optional[MemoryLink] = None

    @classmethod
    def pair(cls) -> tuple["MemoryLink", "MemoryLink"]:
        a, b = cls(), cls()
        a.other, b.other = b, a
        return a, b

    async def send(self, frame: Frame) -> None:
        if self.other is None:
            raise LinkClosed("peer link closed")
        self.other._queue.put_nowait(frame)

    async def close(self) -> None:
        if self.other is not None:
            self.other._queue.put_nowait(_CLOSED)
            self.other.other = None
        self.other = None


async def connect(host: str, port: int, hello: dict, timeout: float = 60.0,
                  retry_interval: float = 0.25, send_timeout: Optional[float] = None) -> TcpLink:
    """Dial a downstream stage, retrying until it accepts this hello.

    A reject is retried like a refused connection: the port may still be held
    by the downstream's previous-epoch listener, which goes away once that
    stage is cancelled and the new epoch's listener binds.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    rejects = 0
    while True:
        attempt = min(CONNECT_ATTEMPT_S, max(deadline - loop.time(), retry_interval))
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), attempt)
        except OSError:  # includes TimeoutError: the attempt itself timed out
            if loop.time() > deadline:
                raise
            await asyncio.sleep(retry_interval)
            continue
        link = TcpLink(reader, writer, send_timeout)
        try:
            await link.send(Frame("hello", hello))
            ack = await link.recv(max(deadline - loop.time(), retry_interval))
            if ack.kind == "hello_ack":
                return link
            # Older peers send an empty meta, so the reason is only for the message.
            why = f"peer rejected hello: {ack.meta.get('reason') or ack.meta}"
        except (LinkClosed, asyncio.TimeoutError) as e:
            why = f"hello failed: {str(e) or 'no reply'}"
        await link.close()
        if loop.time() > deadline:
            raise LinkClosed(why)
        rejects += 1
        log.log(logging.INFO if rejects == 1 else logging.DEBUG,
                "%s:%s not ready for this stage (%s); retrying", host, port, why)
        await asyncio.sleep(retry_interval)


class LinkServer:
    """Accepts exactly one upstream link whose hello matches ``expect``."""

    def __init__(self, host: str, port: int, expect: dict,
                 send_timeout: Optional[float] = None) -> None:
        self.host, self.port, self.expect = host, port, expect
        self.send_timeout = send_timeout
        self._accepted: asyncio.Future = asyncio.get_running_loop().create_future()
        self._claimed = False                 # accept() handed the link to the caller
        self._pending: set[TcpLink] = set()   # connections still in the hello exchange
        self._server: Optional[asyncio.base_events.Server] = None

    async def start(self) -> "LinkServer":
        self._server = await asyncio.start_server(self._on_conn, self.host, self.port,
                                                  reuse_address=True)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    def _reject_reason(self, hello: Frame) -> Optional[str]:
        if hello.kind != "hello":
            return "expected a hello frame"
        wrong = [k for k, v in self.expect.items() if hello.meta.get(k) != v]
        if wrong == ["epoch"]:
            # Same job, so the dialer may learn which epoch this stage is on.
            return f"stage is listening for epoch {self.expect['epoch']}, got {hello.meta.get('epoch')}"
        if wrong:
            return f"hello does not match this stage ({', '.join(wrong)})"
        if self._accepted.done():
            return "stage already has an upstream link"
        return None

    async def _on_conn(self, reader, writer) -> None:
        link = TcpLink(reader, writer, self.send_timeout)
        self._pending.add(link)
        try:
            hello = await link.recv(10.0)
        except Exception:
            await link.close()
            return
        finally:
            self._pending.discard(link)
        reason = self._reject_reason(hello)
        if reason is not None:
            await link.send(Frame("hello_reject", {"reason": reason}))
            await link.close()
            return
        await link.send(Frame("hello_ack", {}))
        self._accepted.set_result(link)

    async def accept(self, timeout: Optional[float] = None) -> TcpLink:
        link = await asyncio.wait_for(asyncio.shield(self._accepted), timeout)
        self._claimed = True
        return link

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()                  # stops listening, so the port frees now
        # wait_closed() waits for every connection, so close the ones nobody else
        # will: half-done hellos and a link accepted after the caller gave up on it.
        links = list(self._pending)
        if self._accepted.done() and not self._accepted.cancelled() and not self._claimed:
            links.append(self._accepted.result())
        for link in links:
            await link.close()
        await self._server.wait_closed()
