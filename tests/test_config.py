import logging
from pathlib import Path

import pytest

from slashcompute.common.config import EngineConfig
from slashcompute.common.logging import setup_logging


@pytest.mark.parametrize("raw,expected", [
    ("True ", True), (" off", False), ("NO", False), ("1", True),
    # Unrecognized spellings must not silently disable the sandbox.
    ("enabled", True), ("", True),
])
def test_sandbox_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("SLASHCOMPUTE_SANDBOX", raw)
    assert EngineConfig.from_env().sandbox is expected


def test_numeric_env_accepts_float_notation(monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_STAGE_OVERHEAD_BYTES", "5e8")
    monkeypatch.setenv("SLASHCOMPUTE_CHECKPOINT_EVERY", " 10 ")
    cfg = EngineConfig.from_env()
    assert cfg.stage_overhead_bytes == 500_000_000
    assert cfg.checkpoint_every == 10


@pytest.mark.parametrize("raw", ["0", "-3", "often"])
def test_bad_checkpoint_every_falls_back_to_default(monkeypatch, caplog, raw):
    monkeypatch.setenv("SLASHCOMPUTE_CHECKPOINT_EVERY", raw)
    with caplog.at_level(logging.WARNING):
        cfg = EngineConfig.from_env()
    assert cfg.checkpoint_every == EngineConfig().checkpoint_every
    assert "SLASHCOMPUTE_CHECKPOINT_EVERY" in caplog.text


def test_empty_home_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_HOME", "  ")
    assert EngineConfig.from_env().home == Path.home() / ".slashcompute"


def test_unknown_log_level_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_LOG", "verbose")
    root = logging.getLogger()
    prev = root.level
    try:
        setup_logging("test")
        assert root.level == logging.INFO
    finally:
        root.setLevel(prev)
