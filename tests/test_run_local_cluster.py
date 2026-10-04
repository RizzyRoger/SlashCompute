"""scripts/run_local_cluster.py exit status."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_local_cluster.py"


def test_crashed_agent_makes_script_exit_nonzero(tmp_path):
    # --gpu-percent 500 is rejected by the agent CLI, so the agent exits right away.
    env = os.environ.copy()
    env["SLASHCOMPUTE_VERIFY_RATE"] = "0"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--agents", "1",
         "--port", "9700", "--data-port", "9701", "--gpu-percent", "500",
         "--home", str(tmp_path / "cluster")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0, proc.stderr
    assert "process exited with 2" in proc.stderr
