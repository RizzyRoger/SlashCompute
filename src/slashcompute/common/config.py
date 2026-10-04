"""Engine-wide defaults. Every value can be overridden with an environment
variable named ``SLASHCOMPUTE_<FIELD>`` (upper-case)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

DEV_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Ordered smallest to largest. The planner picks the largest one that fits the
# pool but not any single node.
DEMO_MODEL_CANDIDATES = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-32B-Instruct-4bit",
]

ALLOWED_MODELS = (DEV_MODEL, *DEMO_MODEL_CANDIDATES)

MDNS_SERVICE_TYPE = "_slashcompute._tcp.local."

log = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
# Numeric fields default to a floor of 0; these need more.
_MIN = {"checkpoint_every": 1}


def allowed_model(model: str) -> bool:
    """Catalog HF ids, or a local directory the coordinator already has."""
    name = (model or "").strip()
    if name in ALLOWED_MODELS:
        return True
    return Path(name).expanduser().is_dir()


@dataclass
class EngineConfig:
    home: Path = field(default_factory=lambda: Path.home() / ".slashcompute")
    coordinator_host: str = "0.0.0.0"
    coordinator_port: int = 8765
    data_port: int = 9700

    checkpoint_every: int = 25
    grace_period_s: float = 60.0
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 20.0
    scheduler_tick_s: float = 1.0
    stage_start_timeout_s: float = 900.0  # includes model download on first use
    max_recoveries: int = 20
    canary_size: int = 512
    canary_interval_s: float = 1800.0

    verify_rate: float = 0.05
    verify_rel_tolerance: float = 2e-2
    # GPU fp32 vs NumPy reference; zeros/random still fail (~1.0).
    canary_rel_tolerance: float = 0.5
    verify_ring_size: int = 8

    # Fraction of each stage's weight bytes reserved for activations, grads and
    # optimizer state, plus a fixed headroom per stage.
    stage_overhead_frac: float = 0.25
    stage_overhead_bytes: int = 512 * 1024**2

    sandbox: bool = True

    # Set on a hosted coordinator. Local LAN host leaves these alone.
    public_pool: bool = False
    public_url: str = ""

    @classmethod
    def from_env(cls, **overrides) -> "EngineConfig":
        cfg = cls()
        for f in fields(cls):
            name = f"SLASHCOMPUTE_{f.name.upper()}"
            env = os.environ.get(name)
            if env is None:
                continue
            val = _parse_env(env.strip(), getattr(cfg, f.name), _MIN.get(f.name, 0))
            if val is None:
                log.warning("ignoring %s=%r; using default %r", name, env, getattr(cfg, f.name))
                continue
            setattr(cfg, f.name, val)
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    @property
    def coordinator_dir(self) -> Path:
        return self.home / "coordinator"

    @property
    def agent_dir(self) -> Path:
        return self.home / "agent"


def _parse_env(env: str, cur, minimum):
    """Parse ``env`` like ``cur``; None means unusable, keep the default."""
    if isinstance(cur, bool):
        low = env.lower()
        return True if low in _TRUE else False if low in _FALSE else None
    if isinstance(cur, Path):
        return Path(env).expanduser() if env else None
    if isinstance(cur, str):
        return env
    try:
        # int(float()) so "5e8" and "25.0" work for byte counts and step counts.
        val = int(float(env)) if isinstance(cur, int) else float(env)
    except (ValueError, OverflowError):
        return None
    return val if val >= minimum else None
