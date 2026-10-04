"""Coordinator reliability: stalled epochs, transactional starts, checkpoint races,
agent reconnects and loop isolation."""

from __future__ import annotations

import asyncio

import pytest

from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.scheduler import EpochState
from slashcompute.jobs import LoraFinetuneSpec


@pytest.fixture
def core(tmp_path, tiny_model, tiny_dataset):
    c = Coordinator(EngineConfig(home=tmp_path / "home", verify_rate=0.0))
    c._tiny = (tiny_model, tiny_dataset)
    return c


def _spec(core, **kw):
    tiny_model, tiny_dataset = core._tiny
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=10,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=1)
    return LoraFinetuneSpec(**(base | kw))


def test_stalled_running_epoch_is_aborted(core):
    job = core.submit(_spec(core))
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"
    asyncio.run(core.recovery.tick())
    assert job.row.status == "running"  # still within stall_timeout_s

    job.current.last_progress -= core.cfg.stall_timeout_s + 1
    asyncio.run(core.recovery.tick())
    assert job.current.closed
    assert job.row.status == "recovering"
    assert "no progress" in job.row.error
