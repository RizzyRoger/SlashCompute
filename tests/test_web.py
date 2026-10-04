import json

from fastapi.testclient import TestClient

from slashcompute.launcher.controller import Launcher, LauncherSettings
from slashcompute.web.server import create_shell


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        if self.health is None:
            raise ConnectionError("down")
        class R:
            status_code = 200
            content = b'{"ok":true,"nodes":1,"jobs":0}'
            headers = {"content-type": "application/json"}
            def json(self_inner):
                return {"ok": True, "nodes": 1, "jobs": 0}
        return R()

    def post(self, url: str, content=None, headers=None, timeout: float = 1.0, **_):
        class R:
            status_code = 200
            content = b'{"user":{"id":"u1","email":"ada@lan.test"},"token":"sess"}'
            headers = {"content-type": "application/json",
                       "set-cookie": "slashcompute_session=sess; HttpOnly; SameSite=lax"}
            def json(self_inner):
                return {"user": {"id": "u1", "email": "ada@lan.test"}, "token": "sess"}
        return R()


def _shell(tmp_path, **kw):
    spawned = []
    n = {"p": 5000}

    def popen(argv, **_):
        n["p"] += 1
        proc = FakeProc(n["p"], argv)
        spawned.append(proc)
        return proc

    launcher = Launcher(
        home=tmp_path,
        python="/opt/venv/bin/python",
        popen=popen,
        http=kw.pop("http", FakeHTTP()),
        discover_fn=kw.pop("discover_fn", lambda timeout=5.0: "http://10.0.0.9:8765"),
        lan_ip_fn=lambda: "192.168.1.20",
    )
    app = create_shell(launcher)
    return app, launcher, spawned


def test_index_and_css(tmp_path):
    app, _, _ = _shell(tmp_path)
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert b"COMPUTE" in r.content
        for tab in (b"contributions", b"usage", b"grants", b"pool"):
            assert b'data-tab="' + tab + b'"' in r.content
        assert b'data-ink="signal"' in r.content
        assert b"/static/logo.png" in r.content
        assert c.get("/static/logo.png").status_code == 200
        assert b'data-file="community"' not in r.content
        assert b">COM<" not in r.content
        assert b'data-mode="public"' in r.content
        assert b"Public pool" in r.content
        js = c.get("/static/app.js")
        assert js.status_code == 200
        assert b"AUTH.sc" not in js.content
        assert b"/api/overview" in js.content
        assert b"GOOGLE" not in js.content
        assert b"connect-public" in js.content
        assert b'mode: "host", contribute: false' not in js.content   # hosting must not stop contributing
        css = c.get("/static/app.css")
        assert css.status_code == 200
        assert b"IBM Plex Sans" in css.content
        assert b"archivo-black" not in css.content
        assert c.get("/static/fonts/ibm-plex-sans-regular.woff2").status_code == 200
        assert c.get("/static/fonts/ibm-plex-mono-regular.woff2").status_code == 200


def test_settings_and_status(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/settings", json={"mode": "join", "url": "10.1.2.3",
                                          "gpu_percent": 40, "finish": "signal"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "join" and body["finish"] == "signal"
        assert body["url"] == "10.1.2.3"
        assert body["session_token"] == ""
        assert body["grant_split"] == 0
        got = c.get("/api/settings").json()
        assert got["gpu_percent"] == 40
        st = c.get("/api/status").json()
        assert st["lan_ip"] == "192.168.1.20"
        assert "carbon" in st["finishes"]
    assert launcher.load_settings().finish == "signal"


def test_start_stop_and_discover(tmp_path, monkeypatch):
    http = FakeHTTP()
    app, launcher, spawned = _shell(tmp_path, http=http)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in spawned))

    def health_after(url):
        if spawned:
            http.health = {"ok": True, "nodes": 0, "jobs": 0}
            return {"ok": True, "nodes": 0, "jobs": 0}
        return None

    launcher.poll_health = health_after
    with TestClient(app) as c:
        r = c.post("/api/start", json={"mode": "host", "gpu_percent": 50, "contribute": True})
        assert r.status_code == 200, r.text
        assert len(spawned) == 2
        found = c.post("/api/discover").json()
        assert found["url"] == "http://10.0.0.9:8765"
        c.post("/api/stop")


def test_join_start_requires_url(tmp_path):
    app, _, spawned = _shell(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/start", json={"mode": "join", "url": ""})
        assert r.status_code == 400
        assert spawned == []


def test_proxy_allows_health_and_blocks_other(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app) as c:
        assert c.get("/api/coord/health").status_code == 200
        assert c.get("/api/coord/auth/me").status_code == 200
        assert c.get("/api/coord/verify/secret").status_code == 404
        assert c.get("/api/coord/jobs/../verify/secret").status_code == 404
        assert c.get("/api/coord/VERIFY/secret").status_code == 404
        assert c.get("/api/shell").json()["generation"] >= 6


class RoutedHTTP:
    """Answers GETs by path. ``routes`` maps path -> JSON payload."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        if path not in self.routes:
            raise ConnectionError("down")
        payload = self.routes[path]

        class R:
            status_code = 200

            def json(self_inner):
                return payload
        return R()


POOL = {
    "/health": {"ok": True, "nodes": 2, "jobs": 2},
    "/nodes": [
        {"node_id": "me", "name": "Air", "matmul_tflops": 2.0, "memory_contrib_bytes": 8 << 30,
         "gpu_percent": 50},
        {"node_id": "b", "name": "Studio", "matmul_tflops": 20.0,
         "memory_contrib_bytes": 64 << 30, "gpu_percent": 80},
    ],
    "/jobs": [
        {"id": "old", "status": "completed", "steps": 10, "progress_step": 10, "submitted_at": 1},
        {"id": "new", "status": "running", "steps": 10, "progress_step": 4, "submitted_at": 2},
    ],
    "/ledger": [
        {"node_id": "me", "kind": "train", "flops": 4e12, "disputed_flops": 0.0},
        {"node_id": "b", "kind": "train", "flops": 9e12, "disputed_flops": 0.0},
    ],
}


def test_overview_offline(tmp_path):
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is False
    assert ov["pool"]["jobs"] == [] and ov["leaderboard"] == []
    assert ov["me"]["flops"] == 0 and ov["me"]["rank"] is None
    assert ov["status"]["lan_ip"] == "192.168.1.20"
    assert ov["status"]["models"][0].endswith("0.5B-Instruct-4bit")
    assert "public_url" in ov["status"]


def test_settings_accept_public_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_PUBLIC_URL", "https://pool.example.com")
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/settings", json={"mode": "public", "url": "", "gpu_percent": 40})
        assert r.status_code == 200
        assert r.json()["mode"] == "public"
        ov = c.get("/api/overview").json()
        assert ov["status"]["mode"] == "public"
        assert ov["status"]["public_url"] == "https://pool.example.com"
        assert ov["status"]["coordinator_url"] == "https://pool.example.com"


def test_overview_online_ranks_this_mac(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP(POOL))
    launcher.save_settings(LauncherSettings(mode="host", grant_split=25))
    launcher.paths.node_id_file.write_text("me\n")
    with TestClient(app) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is True
    assert [j["id"] for j in ov["pool"]["jobs"]] == ["new", "old"]
    assert ov["pool"]["jobs"][0]["progress"] == 0.4 and ov["pool"]["jobs"][0]["can_cancel"]
    assert ov["pool"]["capacity"]["macs"] == 2 and ov["pool"]["capacity"]["running"] == 1
    assert ov["me"]["flops"] == 4e12
    assert ov["me"]["credits"] == {"earned": 4e12, "kept": 3e12, "to_grants": 1e12}
    assert (ov["me"]["rank"], ov["me"]["of"]) == (2, 2)
    assert [n["is_me"] for n in ov["pool"]["nodes"]] == [True, False]


def test_stop_agent_endpoint_leaves_coordinator(tmp_path, monkeypatch):
    kills = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    app, launcher, _ = _shell(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    launcher.paths.pid_file.write_text("222\n")
    with TestClient(app) as c:
        assert c.post("/api/stop-agent").status_code == 200
    assert (222, 15) in kills and (111, 15) not in kills


class GrantHTTP:
    """In-memory coordinator stand-in for the live grants adapter."""

    def __init__(self) -> None:
        self.grants = [
            {"id": "g1", "title": "Parser", "author": "Ada",
             "body": "Need FLOPs for a parser thesis project.",
             "goal_flops": 50e12, "received_flops": 10e12, "status": "approved", "progress": 0.2},
            {"id": "g2", "title": "Waiting", "author": "Bea",
             "body": "Waiting for review of this grant request.",
             "goal_flops": 20e12, "received_flops": 0.0, "status": "pending", "progress": 0.0},
        ]
        self.donated = 0.0

    def _path(self, url: str) -> str:
        return "/" + url.split("://", 1)[-1].split("/", 1)[-1].split("?")[0]

    def _resp(self, payload, status=200):
        raw = json.dumps(payload).encode()

        class R:
            status_code = status
            content = raw
            headers = {"content-type": "application/json"}

            def json(self_inner):
                return payload
        return R()

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        path = self._path(url)
        if path == "/grants":
            return self._resp(self.grants)
        if path == "/auth/me":
            return self._resp({
                "user": {"id": "u1", "name": "Ada", "admin": True, "grant_split": 10,
                         "accepted_terms": True},
                "credits": {"balance": 80e12, "lifetime_earned": 100e12},
            })
        if path == "/credits/transactions":
            return self._resp({"items": [
                {"kind": "donate", "amount": -self.donated} if self.donated else {"kind": "earn", "amount": 1},
            ]})
        if path == "/community/leaderboard":
            return self._resp([{"user_id": "u1", "name": "Ada", "lifetime_earned": 100e12}])
        raise ConnectionError("down")

    def post(self, url: str, content=None, headers=None, timeout: float = 1.0, **_):
        path = self._path(url)
        body = json.loads(content or b"{}")
        if path == "/grants":
            title = str(body.get("title", "")).strip()
            text = str(body.get("body", "")).strip()
            if len(title) < 4 or len(text) < 20:
                return self._resp({"detail": "Describe the need: a title and at least a short paragraph."}, 400)
            row = {"id": "g3", "title": title, "author": "You", "body": text,
                   "goal_flops": float(body.get("goal_flops") or 0), "received_flops": 0.0,
                   "status": "pending", "progress": 0.0}
            self.grants.append(row)
            return self._resp(row)
        if path.endswith("/donate"):
            gid = path.split("/")[2]
            g = next((x for x in self.grants if x["id"] == gid), None)
            if g is None or g["status"] != "approved":
                return self._resp({"detail": "Only approved grants can receive FLOPs."}, 400)
            flops = float(body.get("flops") or 0)
            if flops <= 0:
                return self._resp({"detail": "Enter a number."}, 400)
            g["received_flops"] += flops
            g["progress"] = g["received_flops"] / g["goal_flops"]
            self.donated += flops
            return self._resp(g)
        if "/admin/grants/" in path and path.endswith("/review"):
            gid = path.split("/")[3]
            g = next((x for x in self.grants if x["id"] == gid), None)
            if g is None or g["status"] != "pending":
                return self._resp({"detail": "This grant was already reviewed."}, 400)
            g["status"] = "approved" if body.get("approve") else "declined"
            return self._resp(g)
        return self._resp({"detail": "not found"}, 404)


def test_proxy_forwards_set_cookie(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app) as c:
        r = c.post("/api/coord/auth/login", json={"email": "ada@lan.test", "password": "password1"})
        assert r.status_code == 200
        assert "slashcompute_session=sess" in r.headers.get("set-cookie", "")


def test_live_grants_empty_when_coordinator_down(tmp_path):
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app) as c:
        board = c.get("/api/grants?sort=least").json()
    assert board["sample"] is False
    assert board["grants"] == [] and board["pending"] == []
    assert board["online"] is False


def test_live_grants_flow(tmp_path):
    T = 1e12
    app, launcher, _ = _shell(tmp_path, http=GrantHTTP())
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app) as c:
        board = c.get("/api/grants?sort=top").json()
        assert board["sample"] is False and board["online"] is True
        assert board["grants"][0]["summary"].startswith("Need FLOPs")
        assert board["grants"][0]["goal"] == 50e12
        assert board["pending"][0]["title"] == "Waiting"
        assert board["leaders"][0]["name"] == "Ada"
        assert board["available"] == 80e12

        funded = c.post("/api/grants/g1/fund", json={"amount": 10 * T})
        assert funded.status_code == 200, funded.text
        assert funded.json()["pledged"] == 10 * T
        assert funded.json()["grants"][0]["raised"] == 20e12

        made = c.post("/api/grants", json={
            "title": "Lecture notes",
            "summary": "Fine-tune a helper on my course notes for first years.",
            "goal": 50 * T,
        })
        assert made.status_code == 200, made.text
        assert any(g["title"] == "Lecture notes" for g in made.json()["pending"])
        assert c.post("/api/grants", json={"title": "", "summary": "x", "goal": 1}).status_code == 400
        assert c.post("/api/grants", json={"title": "t", "summary": "x",
                                           "goal": "1e309"}).status_code == 400
        assert c.post("/api/grants/g1/fund", json={"amount": "nan"}).status_code == 400

        approved = c.post("/api/grants/g2/review", json={"approve": True}).json()
        assert any(g["id"] == "g2" for g in approved["grants"])
        again = c.post("/api/grants/g2/review", json={"approve": True})
        assert again.status_code == 400
