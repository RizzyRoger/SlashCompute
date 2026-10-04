"""Inference tunables. Override any field with ``SLASHCOMPUTE_INF_<FIELD>`` (e.g. ``SLASHCOMPUTE_INF_TRANSPORT``)."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

GIB = 1024 ** 3
ENV_PREFIX = "SLASHCOMPUTE_INF_"
TRANSPORTS = ("direct", "relay")


@dataclass(frozen=True)
class InferenceSettings:
    # ------------------------------------------------------------ network
    # direct: heads reach workers' rpc-server on the LAN (fast; trusted network)
    # relay:  every node dials out to the coordinator, which relays llama.cpp RPC over WebSockets
    TRANSPORT: str = "direct"
    REQUIRE_HEAD_APPROVAL: bool = False   # a LAN pool trusts its own heads
    TOKEN: str = ""                       # optional shared secret for /v1/* and node registration

    # ------------------------------------------------------------ planner
    MAX_HOP_MS: float = 30.0              # direct: one-way delay head -> node
    MAX_HOP_MS_RELAY: float = 100.0       # relay: head -> coordinator -> node
    UNKNOWN_LATENCY_MS: float = 10.0
    MAX_PIPELINE_NODES: int = 5
    PLANNER_TOP_CANDIDATES: int = 10
    DEFAULT_CTX: int = 4096
    MAX_CTX: int = 131072                 # context ceiling for models whose GGUF doesn't state one
    KV_BYTES_PER_ELEMENT: int = 2
    COMPUTE_BUFFER_FRACTION: float = 0.10
    COMPUTE_BUFFER_FIXED_BYTES: int = 1 * GIB
    MIN_RELIABILITY: float = 0.2
    RELIABILITY_PENALTY: float = 0.2
    # Speed estimate without measured CU rows: seconds per output token on the reference
    # machine ~= bytes read per token / its memory bandwidth (M-series Ultra ~800 GB/s).
    REF_BANDWIDTH_BYTES_PER_S: float = 800e9

    # ------------------------------------------------------------ pipelines + nodes
    PIPELINE_IDLE_SECONDS: float = 300.0
    HEARTBEAT_SECONDS: float = 5.0
    OFFLINE_AFTER_SECONDS: float = 15.0
    LATENCY_INTERVAL_SECONDS: float = 300.0
    PIPELINE_FORM_TIMEOUT_SECONDS: float = 600.0
    JOB_TIMEOUT_SECONDS: float = 600.0
    COMMAND_LONG_POLL_SECONDS: float = 25.0
    TICK_SECONDS: float = 5.0

    # ------------------------------------------------------------ llama.cpp
    # Empty: any build, but every member of a pipeline must match its head (RPC breaks across builds).
    PINNED_LLAMA_BUILD: str = ""

    # ------------------------------------------------------------ credits (time-weighted FLOPs)
    # Generated tokens are memory-bound: each costs ~2*params FLOPs but takes far longer than a
    # prompt token. Weight them by prompt tok/s / generation tok/s so an hour of hosting earns
    # about what an hour of training does.
    GEN_WEIGHT_DEFAULT: float = 10.0
    GEN_WEIGHT_MAX: float = 50.0
    GEN_WEIGHT_MIN_PROMPT: int = 64       # prompts shorter than this give a noisy prompt tok/s
    GEN_WEIGHT_MIN_PREDICTED: int = 16    # replies shorter than this give a noisy generation tok/s
    GEN_WEIGHT_EMA: float = 0.3

    # ------------------------------------------------------------ service
    DB_PATH: str = ":memory:"
    MODELS_DIR: str = ""                  # where uploaded GGUFs are stored (empty: uploads disabled)
    BACKGROUND_TASKS: bool = True

    EXTRA: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def max_hop_ms(self) -> float:
        return self.MAX_HOP_MS_RELAY if self.TRANSPORT == "relay" else self.MAX_HOP_MS

    def replace(self, **changes) -> "InferenceSettings":
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_env(cls, **overrides) -> "InferenceSettings":
        """Defaults, then ``overrides`` (from the caller), then the environment (wins)."""
        base = cls(**overrides)
        env = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(ENV_PREFIX + f.name)
            if raw is None or f.name == "EXTRA":
                continue
            current = getattr(base, f.name)
            if isinstance(current, bool):
                env[f.name] = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                env[f.name] = int(raw)
            elif isinstance(current, float):
                env[f.name] = float(raw)
            else:
                env[f.name] = raw
        return dataclasses.replace(base, **env)
