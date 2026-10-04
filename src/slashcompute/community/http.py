"""HTTP surface for accounts, credits, grants, and admin."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from sqlmodel import select

from slashcompute.community.auth import AuthError, google_client_id
from slashcompute.community.credits import CreditError
from slashcompute.community.grants import GrantError
from slashcompute.community.terms import TERMS
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import User


def _token(request: Request, authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.cookies.get("slashcompute_session")


def _raise(exc: Exception):
    status = getattr(exc, "status", 400)
    raise HTTPException(status, str(exc)) from exc


def mount_community(app, core: Coordinator) -> None:
    r = APIRouter(tags=["community"])

    def require(request: Request, authorization: Optional[str] = None) -> User:
        user = core.auth.session_user(_token(request, authorization))
        if user is None:
            raise HTTPException(401, "Sign in first.")
        if user.banned:
            raise HTTPException(403, "This account is banned.")
        return user

    def require_terms(request: Request, authorization: Optional[str] = None) -> User:
        user = require(request, authorization)
        if user.accepted_terms_at is None:
            raise HTTPException(403, "Accept the terms first.")
        return user

    def names() -> dict[str, str]:
        with core.db.session() as s:
            return {u.id: u.name for u in s.exec(select(User)).all()}

    @r.get("/auth/terms")
    def terms():
        return {"text": TERMS}

    @r.get("/auth/providers")
    def providers():
        cid = google_client_id()
        return {"google": bool(cid), "google_client_id": cid or None}

    @r.post("/auth/register")
    def register(body: dict, request: Request):
        try:
            user = core.auth.register(body.get("email", ""), body.get("password", ""),
                                      body.get("name", ""))
            user, token = core.auth.login(user.email, body.get("password", ""))
        except AuthError as e:
            _raise(e)
        resp = {"user": core.auth.public_view(user), "token": token}
        # cookie is set by the wrapper below via a side channel — return token for the app
        request.state.session_token = token
        return resp

    @r.post("/auth/login")
    def login(body: dict, request: Request):
        try:
            user, token = core.auth.login(body.get("email", ""), body.get("password", ""))
        except AuthError as e:
            _raise(e)
        request.state.session_token = token
        return {"user": core.auth.public_view(user), "token": token}

    @r.post("/auth/google")
    def google(body: dict, request: Request):
        try:
            user, token = core.auth.login_google(body.get("id_token", ""))
        except AuthError as e:
            _raise(e)
        request.state.session_token = token
        return {"user": core.auth.public_view(user), "token": token}

    @r.post("/auth/logout")
    def logout(request: Request, authorization: Optional[str] = Header(default=None)):
        tok = _token(request, authorization)
        if tok:
            core.auth.logout(tok)
        return {"ok": True}

    @r.get("/auth/me")
    def me(request: Request, authorization: Optional[str] = Header(default=None)):
        user = core.auth.session_user(_token(request, authorization))
        if user is None:
            return {"user": None}
        return {"user": core.auth.public_view(user), "credits": core.credits.summary(user.id)}

    @r.post("/auth/accept-terms")
    def accept_terms(request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return {"user": core.auth.public_view(core.auth.accept_terms(user))}

    @r.patch("/auth/me")
    def patch_me(body: dict, request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        try:
            user = core.auth.update_profile(
                user, name=body.get("name"), grant_split=body.get("grant_split"),
                bio=body.get("bio"),
            )
        except AuthError as e:
            _raise(e)
        return {"user": core.auth.public_view(user)}

    @r.get("/auth/me/nodes")
    def my_nodes(request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        live = {n.node_id: n for n in core.registry.nodes.values()}
        out = []
        for row in core.credits.nodes_for(user.id):
            n = live.get(row.node_id)
            out.append({
                "node_id": row.node_id,
                "user_id": n.user_id if n is not None and n.user_id else row.user_id,
                "online": n is not None,
                "status": n.status if n is not None else None,
                "gpu_percent": n.gpu_percent if n is not None else None,
            })
        return out

    @r.get("/credits/me")
    def credits_me(request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return core.credits.summary(user.id)

    @r.get("/credits/transactions")
    def transactions(request: Request, limit: int = 50, before_id: Optional[int] = None,
                     authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return core.credits.list_txns(user.id, limit=limit, before_id=before_id)

    @r.get("/credits/live")
    def live(request: Request, window_s: float = 60,
             authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return core.credits.live(user.id, window_s=window_s)

    @r.get("/community/leaderboard")
    def leaderboard():
        return core.credits.leaderboard()

    @r.get("/grants")
    def list_grants(request: Request, sort: str = "top",
                    authorization: Optional[str] = Header(default=None)):
        user = core.auth.session_user(_token(request, authorization))
        nm = names()
        return [core.grants.view(g, nm) for g in core.grants.list(sort=sort, viewer=user)]

    @r.get("/grants/{grant_id}")
    def get_grant(grant_id: str, request: Request,
                  authorization: Optional[str] = Header(default=None)):
        try:
            g = core.grants.get(grant_id)
        except GrantError as e:
            _raise(e)
        user = core.auth.session_user(_token(request, authorization))
        if not core.grants.visible(g, user):
            raise HTTPException(404, "Grant not found.")
        comments = []
        nm = names()
        for c in core.grants.comments(grant_id):
            comments.append({
                "user_id": c.user_id, "name": nm.get(c.user_id, c.user_id[:8]),
                "body": c.body, "created_at": c.created_at,
            })
        return {**core.grants.view(g, nm), "comments": comments}

    @r.post("/grants")
    def create_grant(body: dict, request: Request, authorization: Optional[str] = Header(default=None)):
        user = require_terms(request, authorization)
        try:
            g = core.grants.create(user, body.get("title", ""), body.get("body", ""),
                                   body.get("goal_flops", 0))
        except GrantError as e:
            _raise(e)
        return core.grants.view(g, names())

    @r.post("/grants/{grant_id}/donate")
    def donate(grant_id: str, body: dict, request: Request,
               authorization: Optional[str] = Header(default=None)):
        user = require_terms(request, authorization)
        try:
            g = core.grants.donate(user, grant_id, float(body.get("flops", 0)),
                                   from_pot=bool(body.get("from_pot")))
        except (GrantError, CreditError, TypeError, ValueError) as e:
            _raise(e if isinstance(e, (GrantError, CreditError)) else CreditError(str(e)))
        return core.grants.view(g, names())

    @r.post("/grants/{grant_id}/comments")
    def comment(grant_id: str, body: dict, request: Request,
                authorization: Optional[str] = Header(default=None)):
        user = require_terms(request, authorization)
        try:
            c = core.grants.comment(user, grant_id, body.get("body", ""))
        except GrantError as e:
            _raise(e)
        return {"ok": True, "id": c.id, "created_at": c.created_at}

    @r.post("/admin/grants/{grant_id}/review")
    def review(grant_id: str, body: dict, request: Request,
               authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        try:
            g = core.grants.review(admin, grant_id, bool(body.get("approve")),
                                  note=body.get("note"))
        except GrantError as e:
            _raise(e)
        return core.grants.view(g, names())

    @r.post("/admin/users/{user_id}/ban")
    def ban(user_id: str, body: dict, request: Request,
            authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        if not admin.admin:
            raise HTTPException(403, "Admin only.")
        user = core.auth.get(user_id)
        if user is None:
            raise HTTPException(404, "No such user.")
        return {"user": core.auth.public_view(core.auth.set_banned(user, bool(body.get("banned", True))))}

    @r.post("/admin/users/{user_id}/flag")
    def flag(user_id: str, body: dict, request: Request,
             authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        user = core.auth.get(user_id)
        if user is None:
            raise HTTPException(404, "No such user.")
        try:
            core.grants.flag_user(admin, user, body.get("reason", ""))
        except GrantError as e:
            _raise(e)
        return {"user": core.auth.public_view(user)}

    @r.get("/admin/users")
    def users(request: Request, authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        if not admin.admin:
            raise HTTPException(403, "Admin only.")
        with core.db.session() as s:
            rows = s.exec(select(User)).all()
        return [core.auth.public_view(u) for u in rows]

    @r.get("/admin/flags")
    def flags(request: Request, authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        if not admin.admin:
            raise HTTPException(403, "Admin only.")
        return core.grants.list_flags()

    app.include_router(r)
