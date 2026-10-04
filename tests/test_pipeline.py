import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.utils import load_model

from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.data import load_examples, make_batch
from slashcompute.pipeline.local import build_compute, run_local_pipeline
from slashcompute.pipeline.model_profile import profile_model
from slashcompute.pipeline.shard import load_shard
from slashcompute.pipeline.schedule import PipelineDesync, StageRunner
from slashcompute.pipeline.stage import RingEntry, merge_checkpoints, token_losses
from slashcompute.transport import Frame, Link, LinkServer, LinkTimeout, MemoryLink, connect, digest


def _spec(model, data, **kw):
    base = dict(model=str(model), dataset_path=str(data), steps=4, batch_size=4, microbatches=2,
                learning_rate=1e-2, lora_rank=4, max_seq_len=32, seed=7)
    return LoraFinetuneSpec(**(base | kw))


def _batch_loss(compute, batch) -> float:
    tl = token_losses(compute.forward(batch.inputs), batch.targets, batch.mask)
    return (tl.sum() / batch.ntoks).item()


def test_profile(tiny_model):
    p = profile_model(str(tiny_model))
    assert p.num_layers == 6 and p.tie_word_embeddings
    assert all(b > 0 for b in p.layer_bytes)
    assert p.total_weight_bytes == sum(p.layer_bytes) + p.embed_bytes + p.head_bytes


def test_shards_compose_to_full_model(tiny_model):
    full, _ = load_model(tiny_model)
    x = mx.array([[1, 2, 3, 4, 5, 6]])
    ref = full(x)
    a, _ = load_shard(tiny_model, 0, 2)
    b, _ = load_shard(tiny_model, 2, 5)
    c, _ = load_shard(tiny_model, 5, 6)
    assert a.embed_tokens is not None and c.embed_tokens is not None  # tied head
    assert b.embed_tokens is None and b.norm is None
    out = c(b(a(x)))
    assert mx.allclose(out, ref, atol=1e-5).item()


def test_batches_deterministic(tiny_model, tiny_dataset):
    ex = load_examples(tiny_dataset, tiny_model, 32)
    b1, b2 = make_batch(ex, 3, 4, 0), make_batch(ex, 3, 4, 0)
    assert mx.array_equal(b1.inputs, b2.inputs).item()
    other = [make_batch(ex, s, 4, 0).inputs for s in range(4, 8)]
    assert any(o.shape != b1.inputs.shape or not mx.array_equal(o, b1.inputs).item() for o in other)
    assert b1.ntoks == int(b1.mask.sum().item()) > 0


def test_examples_with_completion_truncated_away_are_dropped(tiny_model, tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"tokens": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], "loss_start": 15}\n'
                    '{"tokens": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], "loss_start": 5}\n')
    ex = load_examples(path, tiny_model, 8)
    assert [e.loss_start for e in ex] == [5]
    assert make_batch(ex, 0, 1, 0).mask.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]
    path.write_text('{"tokens": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], "loss_start": 15}\n')
    with pytest.raises(ValueError, match="no usable examples"):
        load_examples(path, tiny_model, 8)


@pytest.mark.parametrize("bad", [256, 5000, -1, 1.5, None])
def test_token_ids_outside_vocab_rejected(tiny_model, tmp_path, bad):
    data = tmp_path / "bad.jsonl"
    data.write_text('{"tokens": [1, 2, 3, 4]}\n' + json.dumps({"tokens": [1, 2, bad, 7, 8]}) + "\n")
    with pytest.raises(ValueError, match=r"line 2: token id .* outside the model vocabulary \[0, 256\)"):
        load_examples(data, tiny_model, 32)
    data.write_text('{"tokens": [0, 1, 254, 255]}\n')
    assert load_examples(data, tiny_model, 32)[0].tokens == [0, 1, 254, 255]


async def test_pipeline_matches_single_stage_reference(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset)
    ref = await run_local_pipeline(spec, [0, 6], tmp_path / "ref")
    two = await run_local_pipeline(spec, [0, 3, 6], tmp_path / "two")
    three = await run_local_pipeline(spec, [0, 1, 4, 6], tmp_path / "three")

    ref_losses = [s.loss for s in ref[0]]
    assert len(ref_losses) == 4
    # It learns: each step draws its own batch, so compare the loss on one
    # fixed batch (step 1's) before training and with the final adapters.
    batch = make_batch(load_examples(tiny_dataset, tiny_model, spec.max_seq_len), 1,
                       spec.batch_size, spec.seed)
    compute = build_compute(spec, 0, 6, 6)
    before = _batch_loss(compute, batch)
    compute.load_checkpoint(tmp_path / "ref/stage0/step_000004.safetensors")
    assert before == pytest.approx(ref_losses[0], rel=1e-4)
    assert _batch_loss(compute, batch) < before
    for got in (two, three):
        last = max(got)
        losses = [s.loss for s in got[last]]
        assert losses == pytest.approx(ref_losses, rel=1e-4, abs=1e-5)
        # chain check: each stage's output digest is the next stage's input digest
        for i in range(last):
            for a, b in zip(got[i], got[i + 1]):
                assert a.out_digest == b.in_digest


async def test_resume_from_merged_checkpoint_with_new_partition(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset, steps=4)
    full = await run_local_pipeline(spec, [0, 3, 6], tmp_path / "full", checkpoint_every=2)

    # Checkpoint at step 2 from a 2-stage run, merged, then resumed on 3 stages.
    merged = tmp_path / "merged.safetensors"
    merge_checkpoints([tmp_path / "full/stage0/step_000002.safetensors",
                       tmp_path / "full/stage1/step_000002.safetensors"], merged)
    resumed = await run_local_pipeline(spec, [0, 2, 4, 6], tmp_path / "resumed",
                                       start_step=2, resume_from=merged, checkpoint_every=2)
    assert [s.step for s in resumed[2]] == [3, 4]
    assert [s.loss for s in resumed[2]] == pytest.approx([s.loss for s in full[1][2:]], rel=1e-4)


def test_checkpoint_missing_layers_rejected(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset)
    a = build_compute(spec, 0, 3, 6)
    a.save_checkpoint(tmp_path / "a.safetensors", 1)
    b = build_compute(spec, 2, 6, 6)
    with pytest.raises(ValueError):
        b.load_checkpoint(tmp_path / "a.safetensors")


async def test_stage_gives_up_on_a_silent_upstream():
    # Before peer timeouts a stage whose neighbour went quiet waited forever.
    prev, _upstream = MemoryLink.pair()
    runner = StageRunner(
        compute=SimpleNamespace(is_first=False, is_last=True), total_steps=1, microbatches=1,
        microbatch_size=1, checkpoint_every=1, checkpoint_dir=Path("."), prev=prev,
        peer_timeout=0.05,
    )
    with pytest.raises(LinkTimeout):
        await asyncio.wait_for(runner.run(), 5)


class _Replaying(Link):
    """Delivers every frame twice and re-sends the previous step's frames, like a
    peer retransmitting after a reconnect."""

    def __init__(self, inner: MemoryLink) -> None:
        super().__init__()
        self.inner, self.sent = inner, []

    async def send(self, frame):
        stale = [f for f in self.sent if f.meta.get("step", 0) < frame.meta.get("step", 0)]
        await self.inner.send(frame)
        await self.inner.send(frame)
        if stale and frame.kind in ("fwd", "bwd"):
            await self.inner.send(stale[-1])
        self.sent.append(frame)

    async def recv(self, timeout=None):
        return await self.inner.recv(timeout)

    async def close(self):
        await self.inner.close()


def _losses(stats):
    return [s.loss for s in stats[max(stats)]]


async def test_repeated_and_stale_frames_are_dropped_not_fatal(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset)
    clean = await run_local_pipeline(spec, [0, 3, 6], tmp_path / "clean")
    links = []
    for _ in range(2):
        a, b = MemoryLink.pair()
        links.append((_Replaying(a), _Replaying(b)))
    noisy = await run_local_pipeline(spec, [0, 2, 4, 6], tmp_path / "noisy", links=links)
    ref = await run_local_pipeline(spec, [0, 2, 4, 6], tmp_path / "ref")
    # Same numbers as without the noise: no microbatch's gradients were applied twice.
    assert _losses(noisy) == pytest.approx(_losses(ref), rel=1e-6)
    assert len(_losses(clean)) == len(_losses(noisy)) == spec.steps


async def test_frame_from_an_unknown_step_is_a_desync():
    prev, upstream = MemoryLink.pair()
    runner = StageRunner(
        compute=SimpleNamespace(is_first=False, is_last=True), total_steps=4, microbatches=1,
        microbatch_size=1, checkpoint_every=1, checkpoint_dir=Path("."), prev=prev,
        peer_timeout=5,
    )
    await upstream.send(Frame("fwd", {"step": 3, "mb": 0}))  # expected step 1
    with pytest.raises(PipelineDesync, match="expected fwd@1"):
        await runner.run()


async def test_training_survives_cut_peer_connections(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset)
    ref = await run_local_pipeline(spec, [0, 3, 6], tmp_path / "ref")

    hello = {"job_id": "j", "epoch": 1}
    server = await LinkServer("127.0.0.1", 0, hello, resume_window=30).start()
    up = await connect("127.0.0.1", server.port, hello, timeout=5, resume_window=30)
    down = await server.accept(5)

    def cut_after(link, counts):
        send, n = link.send, 0

        async def cutting(frame):
            nonlocal n
            await send(frame)
            n += 1
            if n in counts:
                link._tcp.abort()  # drop the connection with this frame possibly still in flight

        link.send = cutting

    cut_after(up, {2, 5})     # forward activations
    cut_after(down, {3})      # backward gradients
    try:
        got = await asyncio.wait_for(
            run_local_pipeline(spec, [0, 3, 6], tmp_path / "tcp", links=[(up, down)]), 120)
    finally:
        await up.close()
        await down.close()
        await server.close()
    assert up.reconnects >= 1  # cuts that land mid-reconnect are noticed as one drop
    assert _losses(got) == pytest.approx(_losses(ref), rel=1e-6)  # no step lost or repeated


def test_verification_ring_does_not_pin_gpu_memory(tiny_model, tiny_dataset):
    c = build_compute(_spec(tiny_model, tiny_dataset), 0, 6, 6, ring_size=4)
    base = mx.get_active_memory()
    for step in range(1, 7):
        x = mx.random.normal((4, 1024, 1024)).astype(mx.bfloat16)  # 8 MB
        c.remember(step, RingEntry(x, x * 2, {"layers.0.lora_a": x[:1]}))
    del x
    assert list(c.ring) == [3, 4, 5, 6]
    assert all(isinstance(a, np.ndarray) for held in c.ring.values() for a, _ in held.values())
    # Holding the arrays kept ring_size x 16 MB of activations in Metal memory.
    assert mx.get_active_memory() - base < 8 * 2**20


def test_bundle_from_the_ring_matches_step_time_digests(tiny_model, tiny_dataset, tmp_path):
    c = build_compute(_spec(tiny_model, tiny_dataset), 0, 6, 6)
    x = mx.random.normal((2, 8, 64)).astype(mx.bfloat16)
    out = mx.random.normal((2, 8, 64)).astype(mx.bfloat16)
    targets, mask = mx.array([[1, 2]], dtype=mx.int32), mx.array([[1.0, 0.0]])
    entry = RingEntry(x, out, c.current_adapters(), targets, mask)
    in_d, out_d = digest(entry.x_in), digest(entry.out)
    c.remember(5, entry)

    path = tmp_path / "bundle.safetensors"
    assert c.save_bundle(5, path)
    t = mx.load(str(path))
    assert digest(t["x_in"]) == in_d and digest(t["out"]) == out_d  # byte-identical bf16
    assert mx.array_equal(t["targets"], targets).item() and mx.array_equal(t["mask"], mask).item()
    assert {k for k in t if k.startswith("adapter/")} == {f"adapter/{k}" for k in c.current_adapters()}

    c.release_ring()
    assert not c.ring and not c.save_bundle(5, path)
