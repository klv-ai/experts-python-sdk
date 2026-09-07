"""Experts — the personas a conversation can be attached to."""

from __future__ import annotations

from typing import Any

from .._transport import Request
from ..types import Expert


def list_request() -> Request:
    return Request(path="/api/v1/profiles/")


def to_experts(raw: Any) -> list[Expert]:
    # Expert rows come back snake_case, unlike conversation rows.
    return [Expert.from_wire(e) for e in (raw or [])]
