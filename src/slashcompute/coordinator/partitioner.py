"""Split a model's layers into contiguous stages across nodes, sized by each
node's contributable memory.

Prefers the fewest stages that fit (fewer network hops), then a split
proportional to memory, falling back to a greedy fill when the proportional
split leaves some stage over budget. Nodes go largest-first; if that fails, the
two largest are tried on the end stages, which carry the embedding / LM head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from slashcompute.pipeline.model_profile import ModelProfile


class PartitionError(Exception):
    pass


@dataclass(frozen=True)
class NodeCapacity:
    node_id: str
    memory_bytes: int


@dataclass(frozen=True)
class StagePlan:
    stage_idx: int
    node_id: str
    layer_start: int
    layer_end: int
    est_bytes: int


@dataclass(frozen=True)
class Overhead:
    frac: float = 0.25
    fixed_bytes: int = 512 * 1024**2

    def need(self, weight_bytes: int) -> int:
        return int(weight_bytes * (1 + self.frac)) + self.fixed_bytes


def _need(profile: ModelProfile, oh: Overhead, start: int, end: int) -> int:
    return oh.need(profile.stage_weight_bytes(start, end))


def _proportional(profile: ModelProfile, caps: Sequence[NodeCapacity]) -> list[int]:
    n, k = profile.num_layers, len(caps)
    total = sum(c.memory_bytes for c in caps)
    bounds, acc = [0], 0.0
    for c in caps[:-1]:
        acc += c.memory_bytes / total * n
        prev = bounds[-1]
        remaining_stages = k - len(bounds)
        b = min(max(round(acc), prev + 1), n - remaining_stages)
        bounds.append(b)
    bounds.append(n)
    return bounds


def _greedy(profile: ModelProfile, caps: Sequence[NodeCapacity], oh: Overhead) -> Optional[list[int]]:
    n, k = profile.num_layers, len(caps)
    bounds = [0]
    for i, c in enumerate(caps[:-1]):
        start = bounds[-1]
        max_end = n - (k - 1 - i)  # leave one layer per remaining stage
        end = start + 1
        if _need(profile, oh, start, end) > c.memory_bytes:
            return None
        while end < max_end and _need(profile, oh, start, end + 1) <= c.memory_bytes:
            end += 1
        bounds.append(end)
    bounds.append(n)
    return bounds


def _fits(profile: ModelProfile, caps: Sequence[NodeCapacity], oh: Overhead, bounds: list[int]) -> bool:
    return all(
        bounds[i] < bounds[i + 1] and _need(profile, oh, bounds[i], bounds[i + 1]) <= c.memory_bytes
        for i, c in enumerate(caps)
    )


def _orderings(caps: Sequence[NodeCapacity]) -> list[list[NodeCapacity]]:
    """Stage orders to try for nodes sorted largest-first: as-is, then the two
    largest on the first and last stages (either way round), the rest between."""
    if len(caps) < 2:
        return [list(caps)]
    first, second, middle = caps[0], caps[1], list(caps[2:])
    out: list[list[NodeCapacity]] = []
    for order in ([*caps], [first, *middle, second], [second, *middle, first]):
        if order not in out:
            out.append(order)
    return out


def _place(profile: ModelProfile, caps: Sequence[NodeCapacity], oh: Overhead) -> Optional[list[int]]:
    bounds = _proportional(profile, caps)
    if _fits(profile, caps, oh, bounds):
        return bounds
    bounds = _greedy(profile, caps, oh)
    return bounds if bounds is not None and _fits(profile, caps, oh, bounds) else None


def partition(profile: ModelProfile, nodes: Sequence[NodeCapacity], *, min_stages: int = 1,
              max_stages: Optional[int] = None, overhead: Overhead = Overhead()) -> list[StagePlan]:
    if not nodes:
        raise PartitionError("no nodes available")
    ranked = sorted(nodes, key=lambda c: (-c.memory_bytes, c.node_id))
    upper = min(len(ranked), profile.num_layers, max_stages or len(ranked))
    if min_stages > upper:
        raise PartitionError(f"need at least {min_stages} stages but only {upper} possible "
                             f"({len(ranked)} nodes, {profile.num_layers} layers)")
    for k in range(min_stages, upper + 1):
        for caps in _orderings(ranked[:k]):
            bounds = _place(profile, caps, overhead)
            if bounds is None:
                continue
            return [
                StagePlan(i, c.node_id, bounds[i], bounds[i + 1],
                          _need(profile, overhead, bounds[i], bounds[i + 1]))
                for i, c in enumerate(caps)
            ]
    pool = sum(c.memory_bytes for c in ranked[:upper])
    raise PartitionError(
        f"model needs ~{overhead.need(profile.total_weight_bytes) / 1e9:.1f} GB "
        f"(plus per-stage overhead); pool of {upper} node(s) offers {pool / 1e9:.1f} GB"
    )


def pick_demo_model(profiles: Sequence[ModelProfile], nodes: Sequence[NodeCapacity],
                    overhead: Overhead = Overhead()) -> Optional[ModelProfile]:
    """Largest model that fits the pool but not any single node."""
    best = None
    biggest_single = max((c.memory_bytes for c in nodes), default=0)
    for p in sorted(profiles, key=lambda p: p.total_weight_bytes):
        if overhead.need(p.total_weight_bytes) <= biggest_single:
            continue
        try:
            partition(p, nodes, overhead=overhead)
        except PartitionError:
            continue
        best = p
    return best
