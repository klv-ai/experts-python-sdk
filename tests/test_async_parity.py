"""The async client, and that it means the same thing as the sync one.

Shipping two clients invites them to drift — one gains a parameter, one fixes
a bug, and callers on the other half never find out. Everything that is not
I/O is shared between them precisely so that cannot happen; these tests are
what hold that arrangement in place.

The parity checks compare SIGNATURES and the requests actually emitted, not
implementations, so they fail when the two stop agreeing about what a call
means rather than about how it waits.
"""

from __future__ import annotations

import inspect
import json

import httpx
import pytest

from klavi_experts import AsyncExpertsClient, ExpertsClient, NotFoundError
from klavi_experts.async_client import (
    AsyncConversations,
    AsyncExperts,
    AsyncKnowledge,
    AsyncSessions,
)
from klavi_experts.client import Conversations, Experts, Knowledge, Sessions

KEY = "sk-" + "k" * 48
CONV = "11111111-1111-1111-1111-111111111111"


def ndjson(*events: dict) -> bytes:
    return b"".join((json.dumps(e) + "\n").encode() for e in events)


class Recorder:
    def __init__(self, replies: list[httpx.Response]) -> None:
        self.replies = replies
        self.calls: list[httpx.Request] = []
        self._i = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        reply = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return reply

    @property
    def paths(self) -> list[str]:
        return [f"{c.method} {c.url.path}" for c in self.calls]

    def body(self, i: int) -> dict:
        return json.loads(self.calls[i].content)


SEND_REPLIES = [
    httpx.Response(200, json={"uid": "user-row"}),
    httpx.Response(200, json={"uid": "stub-row"}),
    httpx.Response(
        200,
        content=ndjson(
            {"type": "stream", "message": {"content": "4"}, "done": False},
            {"type": "stream", "message": {"content": ""}, "done": True,
             "done_reason": "stop", "eval_count": 1, "response": "stub-row"},
        ),
    ),
]


def sync_client(replies):
    rec = Recorder(replies)
    return ExpertsClient(
        base_url="https://install.example", api_key=KEY, max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(rec.handler)),
    ), rec


def async_client(replies):
    rec = Recorder(replies)
    return AsyncExpertsClient(
        base_url="https://install.example", api_key=KEY, max_retries=1,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(rec.handler)),
    ), rec


# ---------- the async client works ----------


class TestAsyncSending:
    async def test_all_three_calls_happen_in_order(self):
        client, rec = async_client(SEND_REPLIES)
        stream = await client.conversations.send(CONV, "What is 2+2?")
        await stream.text()
        assert rec.paths == [
            "POST /api/v1/responses/",
            "POST /api/v1/responses/",
            "POST /api/v1/conversations/chat",
        ]
        await client.aclose()

    async def test_the_answer_is_aggregated(self):
        client, _ = async_client(SEND_REPLIES)
        stream = await client.conversations.send(CONV, "hi")
        assert await stream.text() == "4"
        await client.aclose()

    async def test_events_are_typed(self):
        client, _ = async_client(SEND_REPLIES)
        stream = await client.conversations.send(CONV, "hi")
        assert [e.type async for e in stream] == ["token", "done"]
        await client.aclose()

    async def test_cancel_calls_the_endpoint(self):
        client, rec = async_client(
            [*SEND_REPLIES, httpx.Response(200, json={"cancelled": True})]
        )
        stream = await client.conversations.send(CONV, "hi")
        await stream.cancel()
        assert rec.paths[-1] == "POST /api/v1/conversations/chat/stub-row/cancel"
        await client.aclose()

    async def test_errors_are_the_same_types(self):
        client, _ = async_client([httpx.Response(404, json={"detail": "gone"})])
        with pytest.raises(NotFoundError):
            await client.conversations.get("missing")
        await client.aclose()

    async def test_it_works_as_a_context_manager(self):
        rec = Recorder([httpx.Response(200, json=[])])
        async with AsyncExpertsClient(
            base_url="https://install.example", api_key=KEY,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(rec.handler)),
        ) as client:
            assert await client.experts.list() == []


# ---------- and means the same thing as the sync one ----------


PAIRS = [
    (Experts, AsyncExperts),
    (Conversations, AsyncConversations),
    (Knowledge, AsyncKnowledge),
    (Sessions, AsyncSessions),
]


def public_methods(cls) -> dict[str, inspect.Signature]:
    return {
        name: inspect.signature(fn)
        for name, fn in vars(cls).items()
        if callable(fn) and not name.startswith("_")
    }


@pytest.mark.parametrize("sync_cls,async_cls", PAIRS, ids=lambda c: c.__name__)
def test_both_clients_expose_the_same_methods(sync_cls, async_cls):
    assert set(public_methods(sync_cls)) == set(public_methods(async_cls))


@pytest.mark.parametrize("sync_cls,async_cls", PAIRS, ids=lambda c: c.__name__)
def test_matching_methods_take_the_same_arguments(sync_cls, async_cls):
    """A parameter added to one half and not the other is the drift that
    silently strands callers on the other client."""
    sync_methods = public_methods(sync_cls)
    async_methods = public_methods(async_cls)
    for name, signature in sync_methods.items():
        assert list(signature.parameters) == list(async_methods[name].parameters), (
            f"{sync_cls.__name__}.{name} and {async_cls.__name__}.{name} disagree"
        )


@pytest.mark.parametrize("sync_cls,async_cls", PAIRS, ids=lambda c: c.__name__)
def test_the_async_half_is_actually_async(sync_cls, async_cls):
    for name, fn in vars(async_cls).items():
        if name.startswith("_") or not callable(fn):
            continue
        assert inspect.iscoroutinefunction(fn), (
            f"{async_cls.__name__}.{name} is not async"
        )


def test_the_two_clients_expose_the_same_resources():
    """Both clients hang the same four resources off themselves.

    Read off the constructors rather than by instantiating, so this needs no
    network and no credentials.
    """
    sync_source = inspect.getsource(ExpertsClient.__init__)
    async_source = inspect.getsource(AsyncExpertsClient.__init__)
    for resource in ("experts", "conversations", "knowledge", "sessions"):
        assert f"self.{resource} =" in sync_source
        assert f"self.{resource} =" in async_source


async def test_both_produce_identical_requests_for_the_same_call():
    """The strongest parity check: same call, same bytes on the wire.

    Each client gets its OWN Response objects. An httpx.Response carries a
    stream that is consumed on first read, so handing the same one to a sync
    and then an async client fails inside httpx rather than in anything this
    package does.
    """

    def replies():
        return [httpx.Response(200, json={"uid": CONV})]

    s_client, s_rec = sync_client(replies())
    s_client.conversations.create(expert="e1", title="T", hidden=True)

    a_client, a_rec = async_client(replies())
    await a_client.conversations.create(expert="e1", title="T", hidden=True)
    await a_client.aclose()

    assert s_rec.paths == a_rec.paths
    assert s_rec.body(0) == a_rec.body(0)
