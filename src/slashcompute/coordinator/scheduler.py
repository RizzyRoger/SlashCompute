"""FIFO job scheduling and per-job runtime state.

Strict FIFO: only the oldest waiting job is considered. If the pool can't fit
it yet, later jobs wait too (no starvation of large jobs). A job no pool could
ever fit fails instead of waiting, so it can't hold up the queue.

Each (re)start of a job is an *epoch*. Messages carry the epoch so anything
from a torn-down epoch is ignored.

Starting an epoch is all-or-nothing: its rows are written in one transaction
before any node is marked or messaged, and an assignment that can't be sent
rolls the whole epoch back (the stages already told are cancelled) so the job
simply waits for the next tick.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from slashcompute.common.protocol import CancelStage, StageAssignment
from slashcompute.coordinator.db import Job, StageRun, now
from slashcompute.coordinator.partitioner import NodeCapacity, Overhead, PartitionError, StagePlan, partition
from slashcompute.coordinator.registry import Assignment
from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.model_profile import ModelProfile, profile_model

if TYPE_CHECKING:
    from slashcompute.coordinator.core import Coordinator

log = logging.getLogger(__name__)

WAITING = ("queued", "recovering")
ACTIVE = ("starting", "running")
TERMINAL = ("completed", "failed", "cancelled")


@dataclass
class EpochState:
    epoch: int
    plans: list[StagePlan]
    started: float = field(default_factory=time.monotonic)
    last_progress: float = field(default_factory=time.monotonic)  # last StageReady / StepMetrics
    ready: set[int] = field(default_factory=set)
    finished: dict[int, str] = field(default_factory=dict)
    steps_seen: set[tuple[int, int]] = field(default_factory=set)  # (stage_idx, step) billed
    drain_requested: bool = False
    closed: bool = False

    def node_for(self, stage_idx: int) -> str:
        return self.plans[stage_idx].node_id

    def stage_of(self, node_id: str) -> Optional[StagePlan]:
        return next((p for p in self.plans if p.node_id == node_id), None)


@dataclass
class JobRuntime:
    row: Job
    spec: LoraFinetuneSpec
    profile: Optional[ModelProfile] = None
    current: Optional[EpochState] = None
    # (epoch, step) -> {stage_idx: (in_digest, out_digest, node_id)}
    digests: dict[tuple[int, int], dict[int, tuple[str, str, str]]] = field(default_factory=dict)
    wait_reason: Optional[str] = None
    last_step_flops: Optional[float] = None

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def num_stages(self) -> int:
        return len(self.current.plans) if self.current else 0


class Scheduler:
    def __init__(self, core: "Coordinator") -> None:
        self.core = core

    def overhead(self) -> Overhead:
        cfg = self.core.cfg
        return Overhead(frac=cfg.stage_overhead_frac, fixed_bytes=cfg.stage_overhead_bytes)

    @staticmethod
    def _waiting(job: JobRuntime) -> bool:
        return job.row.status in WAITING and (job.current is None or job.current.closed)

    def next_waiting(self) -> Optional[JobRuntime]:
        waiting = [j for j in self.core.jobs.values() if self._waiting(j)]
        return min(waiting, key=lambda j: j.row.submitted_at, default=None)

    async def tick(self) -> None:
        job = self.next_waiting()
        if job is not None:
            await self.try_start(job)
        await self._check_start_timeouts()

    async def try_start(self, job: JobRuntime) -> bool:
        core = self.core
        if job.profile is None:
            try:
                # Off the loop: the first look at a model may download it from the Hub.
                job.profile = await asyncio.to_thread(profile_model, job.spec.model)
            except Exception as e:
                await core.fail_job(job, f"could not read model {job.spec.model!r}: {e}")
                return False
            if not self._waiting(job):
                return False  # cancelled while we looked
        if job.spec.min_stages > job.profile.num_layers:
            await core.fail_job(job, f"min_stages={job.spec.min_stages} exceeds the model's "
                                     f"{job.profile.num_layers} layers")
            return False
        nodes = core.registry.schedulable()
        caps = [NodeCapacity(n.node_id, n.device.memory_contrib_bytes) for n in nodes]
        try:
            plans = partition(job.profile, caps, min_stages=job.spec.min_stages,
                              max_stages=job.spec.max_stages, overhead=self.overhead())
        except PartitionError as e:
            if job.wait_reason != str(e):
                log.info("job %s waiting: %s", job.id, e)
            job.wait_reason = str(e)
            return False
        job.wait_reason = None

        # Build everything before changing anything.
        row = job.row
        epoch, resume = row.epoch + 1, row.last_checkpoint_step
        try:
            msgs = self._assignments(job, plans, epoch, resume)
        except LookupError as e:
            log.info("job %s waiting: %s", job.id, e)
            return False
        runs = [StageRun(job_id=job.id, epoch=epoch, stage_idx=p.stage_idx, node_id=p.node_id,
                         layer_start=p.layer_start, layer_end=p.layer_end) for p in plans]
        before = (row.epoch, row.status, row.started_at)
        row.epoch, row.status = epoch, "starting"
        row.started_at = row.started_at or now()
        try:
            core.db.save_all(row, *runs)  # the epoch is recorded in full or not at all
        except Exception as e:
            row.epoch, row.status, row.started_at = before
            job.wait_reason = f"could not record epoch {epoch}: {e}"
            log.exception("job %s: could not start epoch %d", job.id, epoch)
            return False

        cur = job.current = EpochState(epoch=epoch, plans=plans)
        for p in plans:
            core.registry.get(p.node_id).assignment = Assignment(job.id, epoch, p.stage_idx)
        log.info("job %s epoch %d: %d stage(s) %s resume_step=%d", job.id, epoch, len(plans),
                 [(p.node_id[:8], p.layer_start, p.layer_end) for p in plans], resume)
        sent: list[str] = []
        for p, msg in zip(plans, msgs):
            if job.current is not cur or cur.closed:
                return False  # torn down (node lost, job cancelled) while we were sending
            if not await core.send(p.node_id, msg):
                await self._roll_back(job, cur, before[1], sent,
                                      f"node {p.node_id[:8]} left before its assignment was sent")
                return False
            sent.append(p.node_id)
        return True

    def _assignments(self, job: JobRuntime, plans: list[StagePlan], epoch: int,
                     resume: int) -> list[StageAssignment]:
        """One assignment per stage. LookupError if a planned node is gone, which would
        otherwise leave its neighbour without a peer address."""
        core = self.core
        nodes = [core.registry.get(p.node_id) for p in plans]
        missing = [p.node_id[:8] for p, n in zip(plans, nodes) if n is None]
        if missing:
            raise LookupError(f"planned node(s) {', '.join(missing)} left")
        last = len(plans) - 1
        return [StageAssignment(
            job_id=job.id, epoch=epoch, stage_idx=p.stage_idx, num_stages=len(plans),
            layer_start=p.layer_start, layer_end=p.layer_end, num_layers=job.profile.num_layers,
            spec=job.spec, prev_peer=nodes[p.stage_idx - 1].peer if p.stage_idx > 0 else None,
            next_peer=nodes[p.stage_idx + 1].peer if p.stage_idx < last else None,
            resume_step=resume,
            checkpoint_url=f"/jobs/{job.id}/checkpoints/{resume}" if resume > 0 else None,
            dataset_url=f"/jobs/{job.id}/dataset" if p.stage_idx == 0 else None,
            checkpoint_every=job.spec.checkpoint_every or core.cfg.checkpoint_every,
            verify_ring_size=core.cfg.verify_ring_size,
            peer_timeout_s=core.cfg.peer_timeout_s,
        ) for p in plans]

    async def _roll_back(self, job: JobRuntime, cur: EpochState, status: str, sent: list[str],
                         reason: str) -> None:
        """Undo a partly sent epoch start. Nothing ran yet, so it isn't a recovery: the job
        goes back to waiting and is tried again on the next tick."""
        core = self.core
        log.warning("job %s epoch %d start rolled back: %s", job.id, cur.epoch, reason)
        cur.closed = True
        for p in cur.plans:
            node = core.registry.get(p.node_id)
            if node is not None and node.assignment == Assignment(job.id, cur.epoch, p.stage_idx):
                node.assignment = None
            core.recovery._close_stage_run(job.id, cur.epoch, p.stage_idx, "start rolled back")
        for node_id in sent:
            await core.send(node_id, CancelStage(job_id=job.id, epoch=cur.epoch))
        job.row.status, job.wait_reason = status, reason
        try:
            core.db.save(job.row)
        except Exception:  # memory stays authoritative; a restart re-queues the job anyway
            log.exception("job %s: could not save the rolled-back status", job.id)

    async def on_stage_ready(self, job: JobRuntime, epoch: int, stage_idx: int) -> None:
        cur = job.current
        if cur is None or cur.epoch != epoch or cur.closed:
            return
        cur.ready.add(stage_idx)
        cur.last_progress = time.monotonic()
        if len(cur.ready) == len(cur.plans) and job.row.status == "starting":
            # The epoch is healthy again: drop the reason the previous one aborted.
            job.row.status, job.row.error = "running", None
            self.core.db.save(job.row)
            log.info("job %s epoch %d running", job.id, epoch)

    async def _check_start_timeouts(self) -> None:
        limit = self.core.cfg.stage_start_timeout_s
        for job in list(self.core.jobs.values()):
            cur = job.current
            if (job.row.status == "starting" and cur and not cur.closed
                    and time.monotonic() - cur.started > limit):
                await self.core.recovery.abort_epoch(job, f"stages not ready within {limit:.0f}s")
