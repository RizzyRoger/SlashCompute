"""``slashcompute-coordinator`` CLI."""

from __future__ import annotations

import contextlib
import json
import socket
from pathlib import Path
from typing import Optional

import httpx
import typer

from slashcompute.common.config import EngineConfig
from slashcompute.common.logging import setup_logging

app = typer.Typer(no_args_is_help=True, help="/compute coordinator")

SHUTDOWN_GRACE_S = 2.0  # in-flight requests (chat streams, uploads) get this long after SIGTERM


def _url(url: Optional[str]) -> str:
    return url or f"http://127.0.0.1:{EngineConfig.from_env().coordinator_port}"


SessionToken = typer.Option(None, "--session-token", envvar="SLASHCOMPUTE_SESSION",
                            help="Account session (needed on a public pool)")


def _bind(host: str, port: int) -> socket.socket:
    """Take the listening socket the way uvicorn would, or exit 3 like uvicorn does when it can't."""
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        sock.close()
        typer.echo(f"cannot listen on {host}:{port}: {e.strerror or e}", err=True)
        raise typer.Exit(3)
    return sock


def _call(method: str, path: str, url: Optional[str], token: Optional[str] = None,
          timeout: float = 30, **kw) -> None:
    base = _url(url)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.request(method, f"{base}{path}", headers=headers, timeout=timeout, **kw)
    except httpx.HTTPError as e:
        typer.echo(f"cannot reach coordinator at {base}: {e}", err=True)
        raise typer.Exit(1)
    if r.status_code >= 400:
        typer.echo(r.text, err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(r.json(), indent=2))


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address (default 0.0.0.0)"),
    port: int = typer.Option(None, help="Port (default 8765)"),
    home: Optional[Path] = typer.Option(None, help="State directory (default ~/.slashcompute)"),
    mdns: bool = typer.Option(True, help="Advertise on the LAN via mDNS"),
    inference_transport: Optional[str] = typer.Option(
        None, help="LLM inference: direct (LAN, default) or relay (internet: RPC through the coordinator)"),
    public: bool = typer.Option(False, "--public", help="Internet pool: login, one Mac per job, no mDNS"),
):
    """Run the coordinator."""
    import uvicorn

    from slashcompute.coordinator.app import create_app
    from slashcompute.inference.config import TRANSPORTS, InferenceSettings

    setup_logging("coordinator")
    cfg = EngineConfig.from_env(
        coordinator_host=host, coordinator_port=port, home=home,
        public_pool=True if public else None,
    )
    inference = InferenceSettings.from_env(DB_PATH=str(cfg.home / "inference.sqlite3"),
                                           MODELS_DIR=str(cfg.home / "models"))
    if inference_transport:
        if inference_transport not in TRANSPORTS:
            raise typer.BadParameter(f"--inference-transport must be one of {', '.join(TRANSPORTS)}")
        inference = inference.replace(TRANSPORT=inference_transport)
    # Bind before building the app: the Coordinator rewrites live jobs to `recovering` as it loads,
    # and the lifespan starts scheduling and mDNS, none of which a run that can't listen may do.
    sock = _bind(cfg.coordinator_host, cfg.coordinator_port)
    api = create_app(cfg, advertise=bool(mdns) and not cfg.public_pool, inference=inference)

    class Server(uvicorn.Server):
        # On SIGTERM uvicorn waits for every in-flight request before the lifespan shutdown, and
        # inference nodes hold 25 s `/agent/commands` long-polls open (even after they've gone):
        # release those first, and cap the wait for anything else.
        async def shutdown(self, sockets=None):
            api.state.inference.bus.close()
            await super().shutdown(sockets)

    server = Server(uvicorn.Config(api, host=cfg.coordinator_host, port=cfg.coordinator_port,
                                   log_level="warning", ws_ping_interval=20, ws_max_size=64 * 1024 * 1024,
                                   timeout_graceful_shutdown=SHUTDOWN_GRACE_S))
    with contextlib.suppress(KeyboardInterrupt):
        server.run(sockets=[sock])
    if not server.started:
        raise typer.Exit(3)  # startup failed, like uvicorn.run


@app.command()
def submit(spec: Path = typer.Argument(..., help="JSON job spec"), url: Optional[str] = None,
           session_token: Optional[str] = SessionToken):
    """Submit a job spec (JSON file)."""
    body = json.loads(spec.read_text())
    if "dataset_path" in body:
        body["dataset_path"] = str((spec.parent / body["dataset_path"]).resolve()) \
            if not Path(body["dataset_path"]).is_absolute() else body["dataset_path"]
    _call("POST", "/jobs", url, session_token, timeout=60, json=body)


def _get(path: str, url: Optional[str]):
    _call("GET", path, url)


@app.command()
def jobs(job_id: Optional[str] = typer.Argument(None), url: Optional[str] = None):
    """List jobs, or show one."""
    _get(f"/jobs/{job_id}" if job_id else "/jobs", url)


@app.command()
def nodes(url: Optional[str] = None):
    """List connected nodes."""
    _get("/nodes", url)


@app.command()
def ledger(url: Optional[str] = None):
    """Per-node usage totals (raw FLOPs, memory, time)."""
    _get("/ledger", url)


@app.command()
def verifications(url: Optional[str] = None):
    """List verification results."""
    _get("/verifications", url)


@app.command()
def cancel(job_id: str, url: Optional[str] = None,
           session_token: Optional[str] = SessionToken):
    """Cancel a job."""
    _call("POST", f"/jobs/{job_id}/cancel", url, session_token)


if __name__ == "__main__":
    app()
