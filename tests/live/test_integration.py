"""Integration tests against a real install.

Opt-in: ``pytest tests/live`` with EXPERTS_TEST_KEY set. These exist to catch
DRIFT — the SDK hand-writes its types against Python services that can change
independently, so the point is to assert response *shapes* here rather than at
a customer.

    EXPERTS_BASE_URL=http://localhost:5003 EXPERTS_TEST_KEY=sk-... pytest tests/live
"""

from __future__ import annotations

import contextlib
import os

import pytest

from klavi_experts import AsyncExpertsClient, ExpertsClient, NotFoundError

KEY = os.environ.get("EXPERTS_TEST_KEY")
BASE_URL = os.environ.get("EXPERTS_BASE_URL", "http://localhost:5003")

pytestmark = pytest.mark.skipif(not KEY, reason="EXPERTS_TEST_KEY not set")


@pytest.fixture(scope="module")
def client():
    with ExpertsClient(base_url=BASE_URL, api_key=KEY) as c:
        yield c


@pytest.fixture(scope="module")
def experts(client):
    return client.experts.list()


@pytest.fixture
def scratch(client):
    """Hidden conversations, deleted afterwards — no litter in the sidebar."""
    made = []
    yield made
    for uid in made:
        # Cleanup must never mask the failure the test was reporting.
        with contextlib.suppress(Exception):
            client.conversations.delete(uid)


class TestExperts:
    def test_the_shape_is_what_the_sdk_claims(self, experts):
        assert experts
        expert = experts[0]
        assert isinstance(expert.uid, str) and expert.uid
        assert isinstance(expert.min_role_id, int)
        assert isinstance(expert.private, bool)
        assert isinstance(expert.starters, list)

    def test_guest_visibility_is_computable(self, experts):
        for expert in experts:
            expected = not expert.private and expert.min_role_id <= 1
            assert expert.guest_visible == expected


class TestConversations:
    def test_create_read_and_list(self, client, scratch):
        conversation = client.conversations.create(title="SDK live", hidden=True)
        scratch.append(conversation.uid)

        fetched = client.conversations.get(conversation.uid)
        assert fetched.uid == conversation.uid
        assert isinstance(fetched.messages, list)

    def test_a_missing_conversation_is_not_found(self, client):
        with pytest.raises(NotFoundError):
            client.conversations.get("00000000-0000-0000-0000-000000000000")


class TestSending:
    def test_it_streams_and_persists(self, client, experts, scratch):
        # An expert is pinned to a model, and that model can be broken
        # independently of anything this SDK does — a hosted model the provider
        # retired, a tag nobody pulled. Generation then fails INSIDE a 200,
        # reporting itself only in the terminal event's `error`. So try
        # candidates until one generates, and report the collected reasons if
        # none can: "no expert on this install can answer" is a real result.
        pinned = os.environ.get("EXPERTS_TEST_EXPERT")
        candidates = (
            [e for e in experts if e.uid == pinned]
            if pinned
            else [e for e in experts if not e.private]
        )
        assert candidates, "no usable expert on this install"

        failures = []
        for expert in candidates[:4]:
            conversation = client.conversations.create(
                title="SDK live send", expert=expert.uid, hidden=True
            )
            scratch.append(conversation.uid)

            stream = client.conversations.send(
                conversation.uid, "Reply with exactly: pong", expert=expert.uid
            )
            seen = [event.type for event in stream]
            result = stream.result()

            if result.error:
                failures.append(f"{expert.name}: {result.error}")
                continue

            assert "token" in seen
            assert seen[-1] == "done"

            # The whole point of the three-call sequence: it persists. A caller
            # who skipped the placeholder rows sees an empty conversation here.
            reloaded = client.conversations.get(conversation.uid)
            assert len(reloaded.messages) >= 2
            assert any(m.agent and m.text for m in reloaded.messages)
            return

        raise AssertionError(
            "No expert on this install could generate an answer:\n  "
            + "\n  ".join(failures)
        )

    def test_usage_is_in_milliseconds_not_nanoseconds(self, client, scratch):
        conversation = client.conversations.create(title="SDK live usage", hidden=True)
        scratch.append(conversation.uid)
        result = client.conversations.send(conversation.uid, "Say: ok").result()
        assert result.usage.total_tokens > 0
        if result.usage.total_ms is not None:
            # A nanosecond value would be ~1e9 for a one-second call.
            assert result.usage.total_ms < 600_000


class TestKnowledge:
    def test_the_envelope_is_unwrapped(self, client):
        collections = client.knowledge.list_collections()
        assert isinstance(collections, list)
        if collections:
            assert isinstance(collections[0].uid, str)
            assert isinstance(collections[0].file_count, int)


class TestSessions:
    def test_a_token_is_bound_and_revocable(self, client, experts):
        guest = next((e for e in experts if e.guest_visible), None)
        origin = os.environ.get("EXPERTS_TEST_ORIGIN")
        if not guest or not origin:
            pytest.skip("needs a guest-visible expert and a registered origin")

        session = client.sessions.create(expert=guest.uid, origin=origin)
        assert session.token and not session.token.startswith("sk-")
        assert session.expires_in <= 900
        client.sessions.revoke(session.session_id)


class TestAsync:
    async def test_the_async_client_reaches_the_same_install(self):
        async with AsyncExpertsClient(base_url=BASE_URL, api_key=KEY) as client:
            experts = await client.experts.list()
            assert experts
