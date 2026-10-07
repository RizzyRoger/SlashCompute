"""Run a whole pipeline inside one process over in-memory links. Used as the
single-process reference and for fast tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.data import load_examples
from slashcompute.pipeline.lora import apply_lora
from slashcompute.pipeline.schedule import StageRunner, StepStats
from slashcompute.pipeline.shard import load_shard
from slashcompute.pipeline.stage import StageCompute
from slashcompute.transport import Link, MemoryLink


def build_compute(spec: LoraFinetuneSpec, start: int, end: int, num_layers: int,
                  ring_size: int = 8) -> StageCompute:
    shard, _ = load_shard(spec.model, start, end, num_layers)
    apply_lora(shard, spec.lora_targets, spec.lora_rank, spec.lora_scale, spec.seed)
    return StageCompute(shard, spec.learning_rate, ring_size)


async def run_local_pipeline(spec: LoraFinetuneSpec, boundaries: list[int], workdir: Path,
                             start_step: int = 0, resume_from: Optional[Path] = None,
                             checkpoint_every: int = 25,
                             links: Optional[list[tuple[Link, Link]]] = None,
                             ) -> dict[int, list[StepStats]]:
    """``boundaries`` like [0, 3, 6] makes stages [0,3) and [3,6).
    Returns per-stage step stats. ``links`` replaces the in-memory (upstream end,
    downstream end) pair between each pair of neighbouring stages."""
    num_layers = boundaries[-1]
    n = len(boundaries) - 1
    examples = load_examples(spec.dataset_path, spec.model, spec.max_seq_len)
    links = links if links is not None else [MemoryLink.pair() for _ in range(n - 1)]
    stats: dict[int, list[StepStats]] = {i: [] for i in range(n)}
    runners = []
    for i in range(n):
        compute = build_compute(spec, boundaries[i], boundaries[i + 1], num_layers)
        if resume_from is not None:
            compute.load_checkpoint(resume_from)
        ckdir = workdir / f"stage{i}"
        ckdir.mkdir(parents=True, exist_ok=True)

        async def on_step(s: StepStats, i=i):
            stats[i].append(s)

        runners.append(StageRunner(
            compute=compute, total_steps=spec.steps, microbatches=spec.microbatches,
            microbatch_size=spec.microbatch_size, checkpoint_every=checkpoint_every,
            checkpoint_dir=ckdir, prev=links[i - 1][1] if i > 0 else None,
            next=links[i][0] if i < n - 1 else None,
            examples=examples if i == 0 else None, batch_size=spec.batch_size, seed=spec.seed,
            start_step=start_step, on_step=on_step,
        ))
    await asyncio.gather(*(r.run() for r in runners))
    return stats
