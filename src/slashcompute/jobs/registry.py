"""Predefined job types. Users pick one of these; they never submit code."""

from __future__ import annotations

from slashcompute.common.config import allowed_model
from slashcompute.jobs.lora_finetune import LoraFinetuneSpec

JOB_TYPES = {
    "lora_finetune": LoraFinetuneSpec,
}

# Becomes a discriminated union on ``kind`` once there is more than one type.
JobSpec = LoraFinetuneSpec


def parse_spec(data: dict) -> JobSpec:
    kind = data.get("kind", "lora_finetune")
    if not isinstance(kind, str) or kind not in JOB_TYPES:
        raise ValueError(f"unknown job kind {kind!r}; known: {sorted(JOB_TYPES)}")
    spec = JOB_TYPES[kind].model_validate({**data, "kind": kind})
    if not allowed_model(spec.model):
        raise ValueError(f"model {spec.model!r} is not allowed.")
    return spec
