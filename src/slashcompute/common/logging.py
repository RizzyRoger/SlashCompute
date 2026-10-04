from __future__ import annotations

import logging
import os
import sys


def setup_logging(name: str) -> logging.Logger:
    level = os.environ.get("SLASHCOMPUTE_LOG", "INFO").strip().upper()
    bad_level = not isinstance(logging.getLevelName(level), int)
    if bad_level:
        level = "INFO"
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname).1s [%(name)s] %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "uvicorn.access", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if bad_level:
        logging.getLogger(__name__).warning(
            "unknown SLASHCOMPUTE_LOG=%r; using INFO", os.environ["SLASHCOMPUTE_LOG"])
    return logging.getLogger(name)
