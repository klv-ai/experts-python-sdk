"""The sync client, against a stubbed transport.

The three-call send sequence is the centrepiece: on the raw API a chat turn is
three requests in a specific order, and skipping the first two fails SILENTLY —
the model answers, the stream looks fine, and nothing persists. These assert
the calls happen, in order, with the placeholder uid threaded through.
"""

from __future__ import annotations

import json

import httpx
import pytest

from klavi_experts import (
    AuthError,
    BadRequestError,
    ExpertsClient,
    LicenseError,
    NotFoundError,
    RateLimitError,
    ServerError,
)

KEY = "sk-" + "k" * 48
CONV = "11111111-1111-1111-1111-111111111111"


def ndjson(*events: dict) -> bytes:
    return b"".join((json.dumps(e) + "\n").encode() for e in events)


class Recorder:
    """A transport stub that records requests and replies from a queue."""

    def __init__(self, replies: list[httpx.Response]) -> None:
        self.replies = replies
        self.calls: list[httpx.Request] = []
        self._index = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply

    @property
    def paths(self) -> list[str]:
        return [f"{c.method} {c.url.path}" for c in self.calls]

    def body(self, index: int) -> dict:
        return json.loads(self.calls[index].content)


def make_client(replies: list[httpx.Response], **kw) -> tuple[ExpertsClient, Recorder]:
    recorder = Recorder(replies)
    transport = httpx.MockTransport(recorder.handler)
    client = ExpertsClient(
        base_url="https://install.example",
        api_key=KEY,
        max_retries=1,
        http_client=httpx.Client(transport=transport),
        **kw,
    )
    return client, recorder


def ok(payload) -> httpx.Response:
    return httpx.Response(200, json=payload)


SEND_REPLIES = [
    ok({"uid": "user-row"}),      # 1. the user's turn
    ok({"uid": "stub-row"}),      # 2. the assistant placeholder
    httpx.Response(              # 3. the stream
        200,
        content=ndjson(
            {"type": "stream", "message": {"content": "4"}, "done": False},
            {"type": "stream", "message": {"content": ""}, "done": True,
             "done_reason": "stop", "eval_count": 1, "response": "stub-row"},
        ),
        headers={"content-type": "application/x-ndjson"},
    ),
]


class TestCredentials:
    def test_the_key_goes_out_as_a_bearer_token(self):
        client, rec = make_client([ok([])])
        client.experts.list()
        assert rec.calls[0].headers["authorization"] == f"Bearer {KEY}"

    def test_two_credentials_are_never_sent_at_once(self):
        # The gateway checks Authorization first and that branch is terminal:
        # a stale bearer token 401s and never falls through to X-API-Key.
        client, rec = make_client([ok([])])
        client.experts.list()
        assert "x-api-key" not in rec.calls[0].headers

    def test_a_credential_is_required(self):
        with pytest.raises(ValueError, match="api_key"):
            ExpertsClient(base_url="https://x.example")

    def test_a_base_url_is_required(self):
        with pytest.raises(ValueError, match="base_url"):
            ExpertsClient(base_url="", api_key=KEY)

    def test_a_trailing_slash_on_the_base_url_is_tolerated(self):
        client, rec = make_client([ok([])])
        client._transport.base_url = "https://install.example"
        client.experts.list()
        assert "//api" not in str(rec.calls[0].url)


class TestSending:
    def test_all_three_calls_happen_in_order(self):
        client, rec = make_client(SEND_REPLIES)
        client.conversations.send(CONV, "What is 2+2?").text()
        assert rec.paths == [
            "POST /api/v1/responses/",
            "POST /api/v1/responses/",
            "POST /api/v1/conversations/chat",
        ]

    def test_the_user_turn_is_settled_and_the_assistant_row_is_not(self):
        client, rec = make_client(SEND_REPLIES)
        client.conversations.send(CONV, "hi").text()
        assert rec.body(0)["agent"] is False and rec.body(0)["done"] is True
        assert rec.body(1)["agent"] is True and rec.body(1)["done"] is False

    def test_the_placeholder_uid_is_passed_to_chat(self):
        # Without it the model still answers and the stream looks fine, but
        # nothing is persisted and every event carries an empty `response`.
        client, rec = make_client(SEND_REPLIES)
        client.conversations.send(CONV, "hi").text()
        assert rec.body(2)["response"] == "stub-row"

    def test_the_answer_is_aggregated(self):
        client, _ = make_client(SEND_REPLIES)
        assert client.conversations.send(CONV, "hi").text() == "4"

    def test_events_are_typed(self):
        client, _ = make_client(SEND_REPLIES)
        stream = client.conversations.send(CONV, "hi")
        assert [e.type for e in stream] == ["token", "done"]

    def test_a_stream_cannot_be_consumed_twice(self):
        client, _ = make_client(SEND_REPLIES)
        stream = client.conversations.send(CONV, "hi")
        stream.text()
        with pytest.raises(Exception, match="already been consumed"):
            list(stream)

    def test_the_result_carries_usage_and_provenance(self):
        client, _ = make_client(SEND_REPLIES)
        result = client.conversations.send(CONV, "hi").result()
        assert result.conversation_uid == CONV
        assert result.response_uid == "stub-row"
        assert result.usage.completion_tokens == 1

    def test_ask_creates_a_conversation_first(self):
        client, rec = make_client([ok({"uid": CONV}), *SEND_REPLIES])
        client.conversations.ask("hi").text()
        assert rec.paths[0] == "POST /api/v1/conversations/start"


class TestCancelling:
    def test_cancel_calls_the_endpoint_rather_than_just_dropping(self):
        # Generation is DETACHED server-side: walking away stops the relay,
        # not the model. It keeps generating, keeps billing, still persists.
        client, rec = make_client([*SEND_REPLIES, ok({"cancelled": True})])
        stream = client.conversations.send(CONV, "hi")
        stream.cancel()
        assert rec.paths[-1] == "POST /api/v1/conversations/chat/stub-row/cancel"

    def test_detach_does_not_call_cancel(self):
        client, rec = make_client(SEND_REPLIES)
        stream = client.conversations.send(CONV, "hi")
        stream.detach()
        assert not any("cancel" in p for p in rec.paths)


class TestErrors:
    @pytest.mark.parametrize(
        "status,payload,expected",
        [
            (401, {}, AuthError),
            (404, {"detail": "gone"}, NotFoundError),
            (429, {"detail": "slow"}, RateLimitError),
            (400, {"detail": "bad"}, BadRequestError),
            (500, {}, ServerError),
        ],
    )
    def test_status_maps_to_a_type(self, status, payload, expected):
        client, _ = make_client([httpx.Response(status, json=payload)])
        with pytest.raises(expected):
            client.experts.list()

    def test_an_expired_licence_is_not_merely_a_permission_error(self):
        # Nothing about the request is wrong and no retry helps — the
        # install's administrator has to renew.
        client, _ = make_client([
            httpx.Response(403, json={"error": "license_expired",
                                      "message": "Your license has expired."})
        ])
        with pytest.raises(LicenseError, match="installation's licence"):
            client.experts.list()

    def test_retry_after_is_exposed(self):
        client, _ = make_client([
            httpx.Response(429, json={"detail": "slow"}, headers={"retry-after": "42"})
        ])
        with pytest.raises(RateLimitError) as exc:
            client.experts.list()
        assert exc.value.retry_after == 42

    def test_a_404_body_is_not_reported_as_an_empty_answer(self):
        # A 404 body is valid JSON, and on a streaming endpoint one valid
        # NDJSON line. Reading before checking the status turns "conversation
        # not found" into "the model returned nothing".
        client, _ = make_client([
            httpx.Response(404, json={"detail": "Conversation not found"})
        ])
        with pytest.raises(NotFoundError, match="Conversation not found"):
            client.conversations.get("missing")


class TestNormalisation:
    def test_expert_rows_are_camelised(self):
        client, _ = make_client([
            ok([{"uid": "e1", "name": "Support",
                 "model_file": "You are helpful.", "min_role_id": 1,
                 "output_language": "English (US)", "private": False}])
        ])
        expert = client.experts.list()[0]
        assert expert.instructions == "You are helpful."
        assert expert.min_role_id == 1
        assert expert.output_language == "English (US)"

    def test_guest_visible_filters_what_a_session_could_use(self):
        # A session token is a guest. An expert above that floor resolves to
        # nothing server-side and the chat silently answers on the site
        # default model with no persona.
        client, _ = make_client([
            ok([
                {"uid": "guest-ok", "name": "A", "min_role_id": 1, "private": False},
                {"uid": "too-high", "name": "B", "min_role_id": 2, "private": False},
                {"uid": "is-private", "name": "C", "min_role_id": 1, "private": True},
            ])
        ])
        assert [e.uid for e in client.experts.list_guest_visible()] == ["guest-ok"]

    def test_message_rows_keep_their_snake_case_source(self):
        client, _ = make_client([
            ok({"uid": CONV, "title": "T",
                "messages": [{"uid": "m1", "agent": True, "text": "hi",
                              "created_at": "2026-09-06T07:00:00Z",
                              "source_docs": [{"uid": "d", "cosine_dist": 0.4}]}]})
        ])
        message = client.conversations.get(CONV).messages[0]
        assert message.created_at == "2026-09-06T07:00:00Z"
        assert message.source_docs[0].similarity == pytest.approx(0.6)

    def test_the_knowledge_envelope_is_unwrapped(self):
        client, _ = make_client([
            ok({"timestamp": 1, "status": 200, "count": 1,
                "data": [{"uid": "c1", "name": "Policies", "file_count": 12}]})
        ])
        collection = client.knowledge.list_collections()[0]
        assert (collection.uid, collection.file_count) == ("c1", 12)

    def test_a_zero_count_envelope_without_data_is_an_empty_list(self):
        # A zero-count response omits `data` entirely. Keying off its presence
        # returns the envelope itself, and the caller iterates its keys.
        client, _ = make_client([ok({"timestamp": 1, "status": 200, "count": 0})])
        assert client.knowledge.list_collections() == []

    def test_epoch_millis_become_iso(self):
        # This service answers in epoch millis while the conversations service
        # answers in ISO strings.
        client, _ = make_client([
            ok({"timestamp": 1, "status": 200,
                "data": [{"uid": "d1", "created_at": 1788678050498}]})
        ])
        assert client.knowledge.list_documents()[0].created_at.startswith("2026-")


class TestSessions:
    def test_minting_returns_something_safe_for_a_browser(self):
        client, rec = make_client([
            ok({"token": "eyJhbGciOi.x.y", "session_id": "s_abc", "expires_in": 900,
                "conversation": CONV, "expert": {"uid": "e1", "name": "Support"}})
        ])
        session = client.sessions.create(expert="e1", origin="https://acme.com")
        assert not session.token.startswith("sk-")
        assert session.session_id == "s_abc"
        assert session.expires_in <= 900
        assert rec.body(0)["origin"] == "https://acme.com"

    def test_revoking_is_scoped_to_the_session(self):
        client, rec = make_client([ok({"revoked": True})])
        client.sessions.revoke("s_abc")
        assert rec.paths[0] == "DELETE /api/v1/public/sessions/s_abc"
