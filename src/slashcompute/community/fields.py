"""Type checks for JSON body fields, so a wrong type is a 4xx, not a crash."""

from __future__ import annotations


def text(value: object, field: str, error: type[Exception]) -> str:
    """A string field, or "" when missing/null. Any other JSON type raises `error`."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise error(f"{field} must be text.")
    return value
