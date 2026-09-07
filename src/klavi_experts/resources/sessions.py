"""Browser session tokens.

The bridge between a Python server that holds the API key and a browser that
must never see one. Your backend calls ``create()``, hands the returned token
to a visitor, and that token can talk to exactly one expert in one conversation
for a few minutes from one origin.

The origin has to be registered against the key first, by an administrator.
That registration is also what supplies CORS for the surface, so a customer's
site is never at the mercy of the install's operator-level allow-list.
"""

from __future__ import annotations

from typing import Any

from .._transport import Request
from ..types import BrowserSession


def create_request(
    *,
    expert: str,
    origin: str,
    conversation: str | None,
    ttl: int | None,
    metadata: dict[str, Any] | None,
) -> Request:
    return Request(
        method="POST",
        path="/api/v1/public/sessions",
        json={
            "expert": expert,
            "origin": origin,
            "conversation": conversation,
            "ttl": ttl,
            "metadata": metadata or {},
        },
        idempotent=False,
    )


def revoke_request(session_id: str) -> Request:
    return Request(method="DELETE", path=f"/api/v1/public/sessions/{session_id}")


def to_session(raw: Any) -> BrowserSession:
    raw = raw or {}
    expert = raw.get("expert") or {}
    return BrowserSession(
        token=raw.get("token") or "",
        session_id=raw.get("session_id") or "",
        expires_in=int(raw.get("expires_in") or 0),
        conversation=str(raw.get("conversation") or ""),
        expert_uid=str(expert.get("uid") or ""),
        expert_name=expert.get("name") or "",
    )
