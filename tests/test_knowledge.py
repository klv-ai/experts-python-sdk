"""Knowledge, against the wire shapes the embedding module actually sends.

Every payload here is copied from a 1.1 install (build pack 1.1.0-rc.127). The
0.1 release mocked the shapes it expected instead, which is how its upload,
search and processing wait shipped without ever having worked.
"""

from __future__ import annotations

import io
from urllib.parse import parse_qs

import httpx
import pytest
from test_client import KEY, make_client, ok

from klavi_experts import AsyncExpertsClient, UploadJob

COLLECTION = "d0f5ac46-0000-4000-8000-000000000001"
DOC = "54137bfb-f499-4502-81d3-8bf1ab0970a6"


def envelope(data, **extra):
    return {"timestamp": 1793409212047, "status": 200, "message": "OK",
            "count": len(data), "data": data, **extra}


def job_status(*queue, total=1):
    # Not enveloped, unlike the rest of the knowledge service.
    return {"id": 24, "kind": 0, "total_files": total, "error_count": 0,
            "loaded_files": sum(1 for q in queue if q["status_id"] >= 2),
            "queue": list(queue), "queue_total": len(queue), "has_more": False}


def entry(status_id, *, uid=DOC, extra=None):
    return {"status_id": status_id, "object_uid": uid,
            "object_name": "policy.txt", "extra": extra}


class TestDocuments:
    def test_the_collection_filter_is_the_parameter_the_server_reads(self):
        client, rec = make_client([ok(envelope([]))])
        client.knowledge.list_documents(collection=COLLECTION, limit=5)
        query = parse_qs(rec.calls[0].url.query.decode())
        assert query == {"collection": [COLLECTION], "limit": ["5"]}

    def test_document_rows_are_mapped_from_their_wire_names(self):
        client, _ = make_client([ok(envelope([{
            "uid": DOC, "doc_name": "policy.pdf", "doc_type": "pdf",
            "title": "Refund policy", "summary": "How refunds work.",
            "file_size": 2048, "created_at": 1_700_000_000_000,
        }]))])
        [doc] = client.knowledge.list_documents(collection=COLLECTION)
        assert doc.name == "policy.pdf"
        assert doc.type == "pdf"
        assert doc.title == "Refund policy"
        assert doc.summary == doc.description == "How refunds work."
        assert doc.size == 2048
        # The row names no collection; the one it was listed from is kept.
        assert doc.collection_uid == COLLECTION

    def test_delete(self):
        client, rec = make_client([ok({"status": 204, "message": "Deleted"})])
        client.knowledge.delete_document(DOC)
        assert rec.paths == [f"DELETE /api/v1/documents/{DOC}"]


class TestUpload:
    def test_the_form_fields_are_the_ones_the_server_reads(self):
        client, rec = make_client([ok({
            "timestamp": 1, "status": 200,
            "message": "Processing 1 files on the embedding queue",
            "job_queue_run_id": 24,
        })])
        job = client.knowledge.upload(
            io.BytesIO(b"hello"), collection=COLLECTION, filename="policy.txt")
        assert job == UploadJob(job_id=24,
                                message="Processing 1 files on the embedding queue")
        body = rec.calls[0].content
        # `files`, plural, and `collection` — anything else is a 422, or a
        # file that lands in no collection.
        assert b'name="files"; filename="policy.txt"' in body
        assert b'name="collection"' in body and COLLECTION.encode() in body

    def test_an_upload_is_never_retried(self):
        client, rec = make_client(
            [httpx.Response(503, json={"detail": "busy"})], max_retries=3)
        with pytest.raises(Exception):  # noqa: B017 - the type is tested elsewhere
            client.knowledge.upload(io.BytesIO(b"x"), collection=COLLECTION)
        assert len(rec.calls) == 1

    def test_waiting_polls_the_job_until_the_file_is_searchable(self):
        client, rec = make_client([
            ok(job_status(total=1)),        # accepted, not yet queued
            ok(job_status(entry(1))),       # working
            ok(job_status(entry(2))),       # searchable, summary pending
        ])
        status = client.knowledge.wait_for_processing(24, interval=0)
        assert status
        assert status.done
        assert status.document_uids == [DOC]
        assert rec.paths == ["GET /api/v1/documents/job-queue-status/24"] * 3

    def test_a_file_that_failed_says_why_and_is_falsy(self):
        client, _ = make_client([ok(job_status(
            entry(-1, uid=None, extra="This PDF is password-protected.")))])
        status = client.knowledge.wait_for_processing(UploadJob(job_id=24))
        assert status.done
        assert not status
        [failed] = status.failed
        assert failed.error == "This PDF is password-protected."

    def test_a_timeout_returns_the_last_status_rather_than_raising(self):
        client, _ = make_client([ok(job_status(entry(1)))])
        status = client.knowledge.wait_for_processing(24, timeout=0, interval=0)
        assert not status.done
        assert not status
        assert status.files[0].status == "working"


HIT = {
    "cosine_dist": 0.25, "split_text": "Refunds are issued within 14 days.",
    "split_order": 0, "text_split_uid": "split-1", "source_doc_uid": DOC,
    "doc_name": "policy.txt", "title": None, "summary": None,
    "doc_uri": "", "page_number": 3, "source_location": None,
    "start_ms": None, "end_ms": None, "match_source": "both",
    "rrf_score": None, "rerank_score": 0.9,
}


class TestSearch:
    def test_the_request_is_the_rag_search(self):
        client, rec = make_client([ok(envelope([]))])
        client.knowledge.search("refunds", collections=[COLLECTION], limit=3)
        assert rec.paths == ["POST /api/v1/similarity_search/"]
        assert rec.body(0) == {"prompt": "refunds", "top_k": 3,
                               "collections": [COLLECTION]}

    def test_no_collections_means_the_field_is_left_out(self):
        # An empty list is not "everything" to the service: it builds a
        # query that fails with a 500.
        client, rec = make_client([ok(envelope([]))])
        client.knowledge.search("refunds")
        assert "collections" not in rec.body(0)

    def test_hits_are_passages_with_their_document(self):
        client, _ = make_client([ok(envelope([HIT]))])
        [hit] = client.knowledge.search("refunds")
        assert hit.text == "Refunds are issued within 14 days."
        assert hit.document_uid == DOC
        assert hit.document_name == "policy.txt"
        assert hit.similarity == pytest.approx(0.75)
        assert hit.page_number == 3
        assert hit.rerank_score == 0.9
        assert hit.split_uid == "split-1"


class TestAsyncKnowledge:
    async def test_upload_and_wait(self):
        replies = iter([
            ok({"job_queue_run_id": 7, "message": "queued"}),
            ok(job_status(entry(3))),
        ])
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return next(replies)

        async with AsyncExpertsClient(
            base_url="https://install.example", api_key=KEY, max_retries=1,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            job = await client.knowledge.upload(
                io.BytesIO(b"x"), collection=COLLECTION, filename="a.txt")
            status = await client.knowledge.wait_for_processing(job, interval=0)
        assert job.job_id == 7
        assert status and status.document_uids == [DOC]
        assert [c.url.path for c in calls] == [
            "/api/v1/documents/job-queue-upload",
            "/api/v1/documents/job-queue-status/7",
        ]

