import pytest

from slashcompute.coordinator.partitioner import (
    NodeCapacity, Overhead, PartitionError, partition, pick_demo_model,
)
from slashcompute.pipeline.model_profile import ModelProfile

GB = 1024**3
NO_OH = Overhead(frac=0.0, fixed_bytes=0)


def _profile(n=10, layer=1 * GB, embed=GB // 2, head=GB // 2, name="m"):
    return ModelProfile(model=name, num_layers=n, hidden_size=8, vocab_size=10,
                        tie_word_embeddings=False, layer_bytes=(layer,) * n, embed_bytes=embed,
                        head_bytes=head, layer_params=1, head_params=1)


def _check(plans, n):
    assert plans[0].layer_start == 0 and plans[-1].layer_end == n
    for a, b in zip(plans, plans[1:]):
        assert a.layer_end == b.layer_start and a.layer_start < a.layer_end
    assert [p.stage_idx for p in plans] == list(range(len(plans)))


def test_single_node_when_it_fits():
    plans = partition(_profile(), [NodeCapacity("a", 20 * GB), NodeCapacity("b", 20 * GB)],
                      overhead=NO_OH)
    assert len(plans) == 1 and plans[0].node_id in ("a", "b")


def test_min_stages_forces_split():
    plans = partition(_profile(), [NodeCapacity("a", 20 * GB), NodeCapacity("b", 20 * GB)],
                      min_stages=2, overhead=NO_OH)
    _check(plans, 10)
    assert len(plans) == 2


def test_proportional_to_memory():
    plans = partition(_profile(), [NodeCapacity("big", 9 * GB), NodeCapacity("small", 3 * GB)],
                      overhead=NO_OH)
    _check(plans, 10)
    by = {p.node_id: p for p in plans}
    assert (by["big"].layer_end - by["big"].layer_start) > (by["small"].layer_end - by["small"].layer_start)
    for p in plans:
        assert p.est_bytes <= {"big": 9, "small": 3}[p.node_id] * GB


def test_greedy_fallback_and_three_way():
    nodes = [NodeCapacity("a", 4 * GB), NodeCapacity("b", 4 * GB), NodeCapacity("c", 4 * GB)]
    plans = partition(_profile(n=10), nodes, overhead=NO_OH)
    _check(plans, 10)
    assert len(plans) == 3


def test_larger_nodes_take_end_stages_when_smallest_cannot_hold_head():
    # Tied embeddings: the first and last stage both hold the 2 GB embedding,
    # so the 1.2 GB node only fits in the middle.
    tied = ModelProfile(model="t", num_layers=3, hidden_size=8, vocab_size=10,
                        tie_word_embeddings=True, layer_bytes=(GB,) * 3, embed_bytes=2 * GB,
                        head_bytes=2 * GB, layer_params=1, head_params=1)
    nodes = [NodeCapacity("a", 3 * GB), NodeCapacity("b", 3 * GB), NodeCapacity("c", int(1.2 * GB))]
    plans = partition(tied, nodes, overhead=NO_OH)
    _check(plans, 3)
    assert [p.node_id for p in plans] == ["a", "c", "b"]
    # Untied with a large head: the big node must take the last stage.
    plans = partition(_profile(n=3, embed=0, head=2 * GB),
                      [NodeCapacity("big", int(4.25 * GB)), NodeCapacity("small", GB)], overhead=NO_OH)
    _check(plans, 3)
    assert [p.node_id for p in plans] == ["small", "big"]


def test_does_not_fit_raises():
    with pytest.raises(PartitionError, match="pool"):
        partition(_profile(), [NodeCapacity("a", 3 * GB), NodeCapacity("b", 3 * GB)])
    with pytest.raises(PartitionError):
        partition(_profile(), [])


def test_overhead_applied():
    p = _profile(n=2, layer=GB, embed=0, head=0)
    with pytest.raises(PartitionError):
        partition(p, [NodeCapacity("a", 2 * GB)], overhead=Overhead(frac=0.25, fixed_bytes=0))
    assert partition(p, [NodeCapacity("a", 3 * GB)], overhead=Overhead(frac=0.25, fixed_bytes=0))


def test_pick_demo_model():
    small, mid, huge = _profile(n=4, name="s"), _profile(n=14, name="m"), _profile(n=40, name="h")
    nodes = [NodeCapacity("a", 8 * GB), NodeCapacity("b", 8 * GB)]
    assert pick_demo_model([small, mid, huge], nodes, NO_OH).model == "m"
