"""GPipe-style pipeline training loop for one stage.

Per step, stage 0 pushes all microbatches forward, then gradients flow back.
The last stage computes the loss and sends grads back as each microbatch
arrives. Every stage applies its optimizer once per step after summing the
microbatch grads.

Stage 0 drives the pipeline. Between steps it may emit ``stop`` (drain) or
``done`` instead of the next step's activations; each stage checkpoints at
that step boundary, forwards the frame, and exits. Because stage 0 finishes a
step's backward last, every downstream stage has already applied that step
when the frame arrives, so all checkpoints land on the same step.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import mlx.core as mx

from slashcompute.pipeline.data import Batch, Example, make_batch
from slashcompute.pipeline.stage import RingEntry, StageCompute
from slashcompute.transport import Frame, Link, digest


@dataclass
class StepStats:
    step: int
    loss: Optional[float]
    tokens_processed: int  # positions run through this stage (incl. padding)
    loss_tokens: int
    seq_len: int
    wall_s: float
    busy_s: float
    peak_mem_bytes: int
    resident_mem_bytes: int
    in_digest: str
    out_digest: str


@dataclass
class StageResult:
    reason: str  # done | drained
    last_step: int


@dataclass
class StageRunner:
    compute: StageCompute
    total_steps: int
    microbatches: int
    microbatch_size: int
    checkpoint_every: int
    checkpoint_dir: Path
    prev: Optional[Link] = None  # upstream (None on stage 0)
    next: Optional[Link] = None  # downstream (None on last stage)
    examples: Optional[list[Example]] = None  # stage 0 only
    batch_size: int = 0
    seed: int = 0
    start_step: int = 0
    on_step: Optional[Callable[[StepStats], Awaitable[None]]] = None
    on_checkpoint: Optional[Callable[[int, Path], Awaitable[None]]] = None
    pace: Optional[Callable[[float], Awaitable[None]]] = None
    executor: Optional[ThreadPoolExecutor] = None
    # Longest a neighbour may stay silent before this stage gives up on it (None: no limit).
    peer_timeout: Optional[float] = None
    drain: asyncio.Event = field(default_factory=asyncio.Event)

    _busy: float = 0.0
    _checkpointed: set = field(default_factory=set)

    # ------------------------------------------------------------ helpers

    async def _run(self, fn, *args):
        """Run MLX work on the stage's compute thread, then apply throttling."""
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        if self.executor is None:
            out = fn(*args)
        else:
            out = await loop.run_in_executor(self.executor, fn, *args)
        dt = time.perf_counter() - t0
        self._busy += dt
        if self.pace is not None:
            await self.pace(dt)
        return out

    async def _checkpoint(self, step: int) -> None:
        if step in self._checkpointed or step <= 0:
            return
        self._checkpointed.add(step)
        path = self.checkpoint_dir / f"step_{step:06d}.safetensors"
        await self._run(self.compute.save_checkpoint, path, step)
        if self.on_checkpoint is not None:
            await self.on_checkpoint(step, path)

    async def _end_of_step(self, step: int, loss, t0: float, tokens: int, loss_tokens: int,
                           seq_len: int, in_d: str, out_d: str) -> None:
        await self._run(self.compute.apply_update)
        if self.on_step is not None:
            await self.on_step(StepStats(
                step=step, loss=loss, tokens_processed=tokens, loss_tokens=loss_tokens,
                seq_len=seq_len, wall_s=time.perf_counter() - t0, busy_s=self._busy,
                peak_mem_bytes=mx.get_peak_memory(), resident_mem_bytes=mx.get_active_memory(),
                in_digest=in_d, out_digest=out_d,
            ))
        if step % self.checkpoint_every == 0 or step == self.total_steps:
            await self._checkpoint(step)

    def _begin_step(self) -> float:
        self._busy = 0.0
        mx.reset_peak_memory()
        return time.perf_counter()

    # ------------------------------------------------------------ main loop

    async def run(self) -> StageResult:
        if self.compute.is_first:
            return await self._run_first()
        return await self._run_downstream()

    async def _run_first(self) -> StageResult:
        step = self.start_step
        c = self.compute
        while True:
            if step >= self.total_steps or self.drain.is_set():
                kind = "done" if step >= self.total_steps else "stop"
                if kind == "stop":
                    await self._checkpoint(step)
                if self.next is not None:
                    await self.next.send(Frame(kind, {"step": step}))
                return StageResult("done" if kind == "done" else "drained", step)

            step += 1
            t0 = self._begin_step()
            batch: Batch = make_batch(self.examples, step, self.batch_size, self.seed)
            mbs = [batch.microbatch(j, self.microbatch_size) for j in range(self.microbatches)]
            seq_len = batch.inputs.shape[1]
            tokens = batch.inputs.size
            loss_total = None

            if c.is_last:  # single-stage pipeline
                loss_total = 0.0
                for j, mb in enumerate(mbs):
                    loss, tl, _ = await self._run(c.forward_backward_loss, mb.inputs, mb.targets,
                                                  mb.mask, batch.ntoks)
                    loss_total += loss
                    if j == 0:
                        entry = RingEntry(mb.inputs, tl, c.current_adapters(), mb.targets, mb.mask)
                in_d, out_d = digest(entry.x_in), digest(entry.out)
                c.remember(step, entry)
            else:
                adapters = c.current_adapters()
                for j, mb in enumerate(mbs):
                    h = await self._run(c.forward, mb.inputs)
                    if j == 0:
                        entry = RingEntry(mb.inputs, h, adapters)
                    await self.next.send(Frame("fwd", {"step": step, "mb": j, "ntoks": batch.ntoks},
                                               {"h": h, "targets": mb.targets, "mask": mb.mask}))
                for _ in range(self.microbatches):
                    g = await self.next.recv(self.peer_timeout)
                    self._expect(g, "bwd", step)
                    j = g.meta["mb"]
                    await self._run(c.backward, mbs[j].inputs, g.tensors["g"])
                in_d, out_d = digest(entry.x_in), digest(entry.out)
                c.remember(step, entry)

            await self._end_of_step(step, loss_total, t0, tokens, batch.ntoks, seq_len, in_d, out_d)

    async def _run_downstream(self) -> StageResult:
        c = self.compute
        last_step = self.start_step
        while True:
            first = await self.prev.recv(self.peer_timeout)
            if first.kind in ("stop", "done"):
                step = first.meta["step"]
                if first.kind == "stop":
                    await self._checkpoint(step)
                if self.next is not None:
                    await self.next.send(Frame(first.kind, {"step": step}))
                return StageResult("done" if first.kind == "done" else "drained", step)

            self._expect(first, "fwd", None)
            step = first.meta["step"]
            t0 = self._begin_step()
            adapters = c.current_adapters()
            inputs: dict[int, mx.array] = {}
            loss_total = 0.0 if c.is_last else None
            entry = None
            frame = first
            tokens = 0
            for k in range(self.microbatches):
                if k > 0:
                    frame = await self.prev.recv(self.peer_timeout)
                    self._expect(frame, "fwd", step)
                j = frame.meta["mb"]
                x = frame.tensors["h"]
                tokens += x.shape[0] * x.shape[1]
                inputs[j] = x
                if c.is_last:
                    loss, tl, gx = await self._run(c.forward_backward_loss, x, frame.tensors["targets"],
                                                   frame.tensors["mask"], frame.meta["ntoks"])
                    loss_total += loss
                    await self.prev.send(Frame("bwd", {"step": step, "mb": j}, {"g": gx}))
                    if j == 0:
                        entry = RingEntry(x, tl, adapters, frame.tensors["targets"], frame.tensors["mask"])
                else:
                    h = await self._run(c.forward, x)
                    await self.next.send(Frame("fwd", {**frame.meta}, {**frame.tensors, "h": h}))
                    if j == 0:
                        entry = RingEntry(x, h, adapters)

            if not c.is_last:
                for _ in range(self.microbatches):
                    g = await self.next.recv(self.peer_timeout)
                    self._expect(g, "bwd", step)
                    j = g.meta["mb"]
                    gx = await self._run(c.backward, inputs[j], g.tensors["g"])
                    await self.prev.send(Frame("bwd", {"step": step, "mb": j}, {"g": gx}))

            in_d, out_d = digest(entry.x_in), digest(entry.out)
            c.remember(step, entry)
            seq_len = first.tensors["h"].shape[1]
            await self._end_of_step(step, loss_total, t0, tokens, first.meta["ntoks"], seq_len,
                                    in_d, out_d)
            last_step = step

    @staticmethod
    def _expect(frame: Frame, kind: str, step: Optional[int]) -> None:
        if frame.kind != kind or (step is not None and frame.meta.get("step") != step):
            raise RuntimeError(f"pipeline desync: expected {kind}@{step}, got {frame.kind}@{frame.meta}")
