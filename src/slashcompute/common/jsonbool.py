"""Strict boolean fields in JSON request bodies."""

from __future__ import annotations

from fastapi import HTTPException

_STRINGS = {"true": True, "false": False}


def body_bool(body: dict, key: str, default: bool = False) -> bool:
    """``body[key]`` as a bool: a JSON bool, or exactly "true"/"false"; absent or null is ``default``.

    Anything else is a 400 — ``bool("false")`` is True, so coercing would flip admin actions.
    """
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in _STRINGS:
        return _STRINGS[value]
    raise HTTPException(400, f"{key} must be true or false.")
