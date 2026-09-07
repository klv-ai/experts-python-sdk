"""Knowledge — the corpus experts retrieve from.

Served by the embedding module rather than the conversations service, which
shows in the wire format: every response is wrapped in
``{timestamp, status, message, count, data}`` and the rows inside are
snake_case with epoch-millisecond timestamps. Unwrapped and normalised here so
callers see the same shape as everywhere else.

Uploading is asynchronous. A document is accepted, then split, embedded and
indexed by a worker — so a file that has just uploaded is NOT yet retrievable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .._transport import Request
from ..types import Collection, KnowledgeDocument


def unwrap(response: Any) -> Any:
    """Unwrap ``{data: ...}``, tolerating a bare array from an older install.

    An envelope is recognised by its own marker fields, not by ``data`` alone:
    a zero-count response omits ``data`` entirely, and keying off its presence
    then returns the envelope object itself — so the caller iterates the
    envelope's keys instead of its rows.
    """
    if isinstance(response, dict) and (
        "data" in response or ("status" in response and "timestamp" in response)
    ):
        return response.get("data") or []
    return response


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    # Epoch millis from this service; ISO strings from the others. A caller
    # doing datetime parsing on both gets one right and one in 1970.
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
    return str(value)


def collections_request() -> Request:
    return Request(path="/api/v1/collections")


def documents_request(*, collection: str | None, limit: int | None) -> Request:
    return Request(
        path="/api/v1/documents",
        params={"collection_uid": collection, "limit": limit},
    )


def document_request(uid: str) -> Request:
    return Request(path=f"/api/v1/documents/{uid}")


def upload_request(
    *, file: Any, filename: str, collection: str
) -> Request:
    return Request(
        method="POST",
        path="/api/v1/documents/upload",
        files={"file": (filename, file)},
        data={"collection_uid": collection},
        # An upload is never retried — a repeat ingests the file twice, which
        # then shows up as duplicate retrieval hits.
        idempotent=False,
        stream=True,
    )


def search_request(query: str, *, collections: list[str] | None, limit: int) -> Request:
    return Request(
        method="POST",
        path="/api/v1/conversations/similarity-search",
        json={"query": query, "collections": collections or [], "limit": limit},
    )


def to_collections(raw: Any) -> list[Collection]:
    return [
        Collection(
            uid=str(c.get("uid", "")),
            name=c.get("name") or "",
            description=c.get("description"),
            file_count=int(c.get("file_count") or 0),
            created_at=_iso(c.get("created_at")),
            updated_at=_iso(c.get("updated_at")),
        )
        for c in (unwrap(raw) or [])
    ]


def to_documents(raw: Any) -> list[KnowledgeDocument]:
    return [to_document(d) for d in (unwrap(raw) or [])]


def to_document(raw: Any) -> KnowledgeDocument:
    raw = raw or {}
    return KnowledgeDocument(
        uid=str(raw.get("uid", "")),
        name=raw.get("name") or "",
        type=raw.get("type"),
        description=raw.get("description"),
        collection_uid=raw.get("collection_uid"),
        created_at=_iso(raw.get("created_at")),
    )


def is_processed(raw: Any) -> bool:
    doc = unwrap(raw)
    if isinstance(doc, list):
        doc = doc[0] if doc else {}
    if not isinstance(doc, dict):
        return False
    return bool(doc.get("processed") or doc.get("embedded"))
