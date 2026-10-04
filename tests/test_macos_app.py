import os
import stat
import subprocess
from pathlib import Path

from slashcompute.launcher.main import SHELL_GENERATION, ensure_shell, ui_ready


ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "macos" / "install_app.sh"


def test_ui_ready_accepts_compute_page(monkeypatch):
    class R:
        status_code = 200
        text = "<title>/compute</title>\n<h1>COMPUTE</h1>"

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", lambda *a, **k: R())
    assert ui_ready("http://127.0.0.1:8766") is True


def test_ui_ready_rejects_foreign_port(monkeypatch):
    class R:
        status_code = 200
        text = "ok"

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", lambda *a, **k: R())
    assert ui_ready("http://127.0.0.1:8766") is False


def test_ensure_shell_attaches_when_already_up(monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.main.ui_ready", lambda url, timeout=0.6: True)
    monkeypatch.setattr("slashcompute.launcher.main._shell_generation", lambda url: SHELL_GENERATION)
    assert ensure_shell() == "http://127.0.0.1:8766"


def test_install_app_writes_plist_and_launcher(tmp_path):
    repo = tmp_path / "repo"
    py = repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    dest = tmp_path / "out" / "compute.app"
    dest.parent.mkdir()
    subprocess.run(
        ["bash", str(INSTALL), "--repo", str(repo), "--dest", str(dest)],
        check=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    plist = (dest / "Contents" / "Info.plist").read_text()
    assert "com.slashcompute.app" in plist
    assert "<string>/compute</string>" in plist
    assert "<string>compute</string>" in plist
    launch = (dest / "Contents" / "MacOS" / "compute").read_text()
    assert str(repo) in launch
    assert "-m slashcompute.launcher.main" in launch
    assert os.access(dest / "Contents" / "MacOS" / "compute", os.X_OK)
    icon = dest / "Contents" / "Resources" / "icon.png"
    assert icon.is_file()
    assert icon.read_bytes() == (ROOT / "scripts" / "macos" / "icon.png").read_bytes()


def _fake_repo(repo: Path, marker: Path) -> Path:
    py = repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f"#!/bin/sh\npwd > '{marker}'\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return repo


def test_install_app_refuses_non_app_dest(tmp_path):
    repo = _fake_repo(tmp_path / "repo", tmp_path / "marker")
    home = tmp_path / "home"
    home.mkdir()
    keep = home / "keep.txt"
    keep.write_text("precious")
    # Only temp paths here: a regression would rm -rf whatever is listed.
    for dest in (str(home), f"{home}/", str(tmp_path / "somedir")):
        r = subprocess.run(
            ["bash", str(INSTALL), "--repo", str(repo), "--dest", dest],
            env={**os.environ, "HOME": str(home)},
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0, dest
        assert "Refusing --dest" in r.stderr
    assert keep.read_text() == "precious"


def test_install_app_launcher_quotes_hostile_repo_path(tmp_path):
    marker = tmp_path / "marker"
    repo = _fake_repo(tmp_path / 're"po $(touch pwned) `touch pwned` $HOME', marker)
    dest = tmp_path / "compute.app"
    subprocess.run(
        ["bash", str(INSTALL), "--repo", str(repo), "--dest", str(dest)],
        check=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    launch = dest / "Contents" / "MacOS" / "compute"
    subprocess.run(["bash", "-n", str(launch)], check=True)
    subprocess.run([str(launch)], check=True, cwd=tmp_path)
    assert not (tmp_path / "pwned").exists()
    assert Path(marker.read_text().strip()).resolve() == repo.resolve()
