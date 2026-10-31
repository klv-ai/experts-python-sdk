"""Knowledge — the corpus experts retrieve from.

Served by the embedding module rather than the conversations service, which
shows in the wire format: every response is wrapped in
``{timestamp, status, message, count, data}`` and the rows inside are
snake_case with epoch-millisecond timestamps. Unwrapped and normalised here so
callers see the same shape as everywhere else.

Uploading is asynchronous. A file is accepted into an indexing JOB, then
split, embedded and indexed by a worker — so a file that has just uploaded is
NOT yet retrievable, and the upload's answer is the job, not a document. The
job's status is where the document's uid appears, and where a file that could
not be read says why.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .._transport import Request
from ..types import (
    Collection,
    KnowledgeDocument,
    SearchHit,
    UploadedFile,
    UploadJob,
    UploadStatus,
)


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


def documents_request(
    *,
    collection: str | None,
    limit: int | None,
    skip: int | None = None,
    search: str | None = None,
) -> Request:
    return Request(
        path="/api/v1/documents",
        params={
            "collection": collection,
            "limit": limit,
            "skip": skip,
            # Matches the file name, title or summary.
            "searchTerm": search,
        },
    )


def delete_document_request(uid: str) -> Request:
    # Not retried: a repeat after a delete that did land answers 404.
    return Request(
        method="DELETE", path=f"/api/v1/documents/{uid}", idempotent=False
    )


def upload_request(*, file: Any, filename: str, collection: str) -> Request:
    return Request(
        method="POST",
        path="/api/v1/documents/job-queue-upload",
        # `files`, plural: the endpoint takes several in one job.
        files={"files": (filename, file)},
        data={"collection": collection},
        # An upload is never retried — a repeat ingests the file twice, which
        # then shows up as duplicate retrieval hits.
        idempotent=False,
    )


def upload_status_request(job_id: int) -> Request:
    return Request(path=f"/api/v1/documents/job-queue-status/{job_id}")


def search_request(
    query: str, *, collections: list[str] | None, limit: int
) -> Request:
    body: dict[str, Any] = {"prompt": query, "top_k": limit}
    # Only when there are some. An empty list is not "everything": the
    # service turns it into a query that fails.
    if collections:
        body["collections"] = collections
    return Request(
        method="POST",
        # The trailing slash is part of the route.
        path="/api/v1/similarity_search/",
        json=body,
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
            min_role_id=int(c.get("min_role_id") or 2),
            is_public=bool(c.get("is_public")),
            scope_uids=[str(u) for u in (c.get("scope_uids") or [])],
        )
        for c in (unwrap(raw) or [])
    ]


def to_documents(raw: Any, *, collection: str | None = None) -> list[KnowledgeDocument]:
    return [to_document(d, collection=collection) for d in (unwrap(raw) or [])]


def to_document(raw: Any, *, collection: str | None = None) -> KnowledgeDocument:
    raw = raw or {}
    summary = raw.get("summary")
    size = raw.get("file_size")
    return KnowledgeDocument(
        uid=str(raw.get("uid", "")),
        name=raw.get("doc_name") or "",
        type=raw.get("doc_type"),
        description=summary,
        collection_uid=collection,
        created_at=_iso(raw.get("created_at")),
        title=raw.get("title"),
        summary=summary,
        size=int(size) if isinstance(size, (int, float)) else None,
    )


def to_hits(raw: Any) -> list[SearchHit]:
    hits = []
    for h in unwrap(raw) or []:
        distance = h.get("cosine_dist")
        is_num = isinstance(distance, (int, float))
        hits.append(
            SearchHit(
                text=h.get("split_text") or "",
                document_uid=str(h.get("source_doc_uid") or ""),
                document_name=h.get("doc_name") or "",
                title=h.get("title"),
                summary=h.get("summary"),
                distance=distance if is_num else None,
                similarity=(1 - distance) if is_num else None,
                page_number=h.get("page_number"),
                location=h.get("source_location"),
                start_ms=h.get("start_ms"),
                end_ms=h.get("end_ms"),
                rerank_score=h.get("rerank_score"),
                split_uid=h.get("text_split_uid"),
            )
        )
    return hits


def to_upload_job(raw: Any) -> UploadJob:
    raw = raw if isinstance(raw, dict) else {}
    return UploadJob(
        job_id=int(raw.get("job_queue_run_id") or 0),
        message=str(raw.get("message") or ""),
    )


def job_id_of(job: UploadJob | int) -> int:
    return job.job_id if isinstance(job, UploadJob) else int(job)


# The job queue's status ids, as names a caller can read.
_STATUS = {-1: "failed", 0: "pending", 1: "working", 2: "searchable", 3: "finished"}


def to_upload_status(job_id: int, raw: Any) -> UploadStatus:
    # Not enveloped, unlike the rest of this service.
    raw = raw if isinstance(raw, dict) else {}
    files = [
        UploadedFile(
            name=str(q.get("object_name") or ""),
            document_uid=str(q["object_uid"]) if q.get("object_uid") else None,
            status=_STATUS.get(int(q.get("status_id") or 0), "pending"),
            error=q.get("extra") or None,
        )
        for q in (raw.get("queue") or [])
        if isinstance(q, dict)
    ]
    total = int(raw.get("total_files") or 0)
    # Every file accounted for, and none still in the pipeline. "searchable"
    # counts as an end: the document answers searches while its summary is
    # still being written.
    settled = bool(files) and len(files) >= total and all(
        f.status in ("searchable", "finished", "failed") for f in files
    )
    return UploadStatus(job_id=job_id, files=files, done=settled)
