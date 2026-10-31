"""Reading the chat stream.

Every case here corresponds to something that goes wrong when a caller
hand-rolls this against the raw API, which is most of the reason the SDK
exists. The parsing layer is pure and shared by the sync and async clients, so
testing it once tests both.
"""

from __future__ import annotations

import json

import pytest

from klavi_experts._stream import LineBuffer, to_event
from klavi_experts.types import (
    ActionEvent,
    DoneEvent,
    KeepaliveEvent,
    QueuedEvent,
    ThinkingEvent,
    TokenEvent,
)


def collect(*chunks: bytes | str) -> list[dict]:
    buffer = LineBuffer()
    out: list[dict] = []
    for chunk in chunks:
        out.extend(buffer.feed(chunk))
    out.extend(buffer.flush())
    return out


class TestFraming:
    def test_one_event_per_line(self):
        assert collect(b'{"a":1}\n{"a":2}\n') == [{"a": 1}, {"a": 2}]

    def test_an_incomplete_line_carries_across_a_chunk_boundary(self):
        # The single most common hand-rolled bug: a chunk boundary lands
        # mid-JSON, and splitting on newlines discards the remainder. It shows
        # up as occasional dropped tokens, which reads as a model problem.
        assert collect(b'{"message":{"con', b'tent":"hi"}}\n') == [
            {"message": {"content": "hi"}}
        ]

    def test_a_boundary_inside_a_multibyte_character_survives(self):
        raw = '{"c":"né"}\n'.encode()
        split = 8  # mid "é"
        assert collect(raw[:split], raw[split:]) == [{"c": "né"}]

    def test_a_final_line_without_a_newline_is_still_emitted(self):
        assert collect(b'{"a":1}') == [{"a": 1}]

    def test_an_unparseable_line_is_skipped_not_fatal(self):
        # One corrupt event must not discard an answer that is otherwise fine.
        assert collect(b'{"a":1}\nnot json\n{"a":2}\n') == [{"a": 1}, {"a": 2}]

    def test_blank_lines_are_ignored(self):
        assert collect(b'\n\n{"a":1}\n\n') == [{"a": 1}]

    def test_a_json_array_line_is_not_mistaken_for_an_event(self):
        assert collect(b"[1,2,3]\n") == []


class TestEventMapping:
    def test_content_becomes_a_token(self):
        event = to_event(
            {"type": "stream", "message": {"content": "Hi"}, "done": False}
        )
        assert isinstance(event, TokenEvent)
        assert event.content == "Hi"

    def test_thinking_is_kept_separate_from_the_answer(self):
        event = to_event({"type": "thinking", "message": {"content": "hmm"}})
        assert isinstance(event, ThinkingEvent)

    def test_a_keepalive_carries_no_content(self):
        # A heartbeat. Appending it would inject empty strings into the answer.
        assert isinstance(to_event({"type": "keepalive", "message": {"content": ""}}),
                          KeepaliveEvent)

    def test_an_action_keeps_its_detail(self):
        event = to_event(
            {"type": "action", "action": "rag_search", "status": "start",
             "detail": {"ms": 12}}
        )
        assert isinstance(event, ActionEvent)
        assert event.action == "rag_search"
        assert event.detail == {"ms": 12}

    def test_an_empty_non_terminal_chunk_is_dropped(self):
        empty = {"type": "stream", "message": {"content": ""}, "done": False}
        assert to_event(empty) is None

    def test_the_terminal_event_has_no_content_field(self):
        # On the tool path the terminal chunk is empty BY DESIGN — the text
        # already streamed. A content field here invites double-rendering.
        event = to_event({"type": "stream", "message": {"content": ""}, "done": True})
        assert isinstance(event, DoneEvent)
        assert not hasattr(event, "content")


TERMINAL = {
    "type": "stream",
    "message": {"content": ""},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 100,
    "eval_count": 20,
    "total_duration": 5_000_000_000,
    "eval_duration": 3_000_000_000,
    "source_docs": [
        {"uid": "d1", "name": "n", "title": "t", "type": ".pdf", "text": "x",
         "description": "", "cosine_dist": 0.25}
    ],
    "tools_used": ["web_search"],
    "response": "resp-1",
}


class TestTerminalEvent:
    def test_nanoseconds_become_milliseconds(self):
        # Ollama's units, passed straight through by the API. Nobody expects
        # nanoseconds, so they are normalised once here.
        event = to_event(TERMINAL)
        assert event.usage.total_ms == 5000.0
        assert event.usage.eval_ms == 3000.0

    def test_tokens_are_totalled(self):
        usage = to_event(TERMINAL).usage
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
            100, 20, 120
        )

    def test_similarity_sits_alongside_distance(self):
        # "distance" ranks backwards — lower is better — which is a reliable
        # source of inverted relevance sorting.
        doc = to_event(TERMINAL).source_docs[0]
        assert (doc.distance, doc.similarity) == (0.25, 0.75)

    def test_tools_used_is_carried(self):
        assert to_event(TERMINAL).tools_used == ["web_search"]

    def test_an_error_inside_a_200_is_surfaced(self):
        # The stream returns HTTP 200 and then reports trouble in its terminal
        # event. A caller checking only the status sees an empty answer and no
        # reason at all.
        event = to_event({**TERMINAL, "error": "model unreachable"})
        assert event.error == "model unreachable"

    def test_a_context_overflow_is_reported(self):
        assert to_event({**TERMINAL, "context_overflow": True}).context_overflow

    def test_a_terminal_event_with_no_counters_is_fine(self):
        event = to_event({"done": True, "message": {"content": ""}})
        assert event.usage.total_tokens == 0
        assert event.source_docs == []


class TestMesh:
    """Experts Mesh (build pack 1.1): who answered, and waiting in line."""

    def test_waiting_in_line_is_its_own_event_not_content(self):
        event = to_event({"type": "mesh_waiting", "node": "n-tokyo", "name": "Tokyo",
                          "ahead": 2, "done": False, "response": "r"})
        assert event == QueuedEvent(ahead=2, node="n-tokyo", name="Tokyo")

    def test_a_reroute_is_flagged(self):
        event = to_event({"type": "mesh_waiting", "node": "n", "name": "Tokyo",
                          "ahead": 0, "rerouting": True, "done": False})
        assert isinstance(event, QueuedEvent) and event.rerouting

    def test_another_members_model_is_named(self):
        event = to_event({**TERMINAL, "model": "gemma4:12b", "served_by": {
            "node": "n-tokyo", "name": "Tokyo", "label": "Tokyo GPU",
            "model": "gemma4:12b",
            "route": [{"node": "n-lis", "name": "Lisbon"},
                      {"node": "n-tokyo", "name": "Tokyo"}]}})
        assert event.model == "gemma4:12b"
        assert event.served_by is not None
        assert event.served_by.name == "Tokyo"
        assert event.served_by.route == ["Lisbon", "Tokyo"]
        assert event.mesh_fallback is None

    def test_a_fallback_says_what_was_out_of_reach(self):
        event = to_event({**TERMINAL, "mesh_fallback": {
            "node": "n-chi", "name": "Chicago", "label": "", "model": "llama3:70b",
            "reason": "unreachable", "self_name": "Dili",
            "answered_with": "gemma4:12b"}})
        assert event.mesh_fallback is not None
        assert (event.mesh_fallback.name, event.mesh_fallback.reason,
                event.mesh_fallback.answered_with) == (
            "Chicago", "unreachable", "gemma4:12b")

    def test_a_strict_expert_that_cannot_reach_its_model_is_an_error(self):
        event = to_event({"type": "stream", "message": {"content": ""}, "done": True,
                          "error": "Chicago is unreachable",
                          "mesh_unavailable": {"node": "n-chi", "name": "Chicago"},
                          "model": "llama3:70b", "response": "r"})
        assert event.error == "Chicago is unreachable"
        assert event.mesh_unavailable == {"node": "n-chi", "name": "Chicago"}


class TestToolOutputs:
    def test_data_artifacts_and_images_are_carried(self):
        event = to_event({**TERMINAL,
                          "data_artifacts": [{"id": "a1", "kind": "chart"}, "junk"],
                          "generated_images": [{"url": "/img/1.png"}]})
        assert event.data_artifacts == [{"id": "a1", "kind": "chart"}]
        assert event.generated_images == [{"url": "/img/1.png"}]

    def test_a_cited_recording_passage_keeps_its_time(self):
        event = to_event({**TERMINAL, "source_docs": [{
            "uid": "d", "name": "call.m4a", "cosine_dist": 0.2,
            "split_uid": "s-9", "start_ms": 61000, "end_ms": 64500}]})
        doc = event.source_docs[0]
        assert (doc.split_uid, doc.start_ms, doc.end_ms) == ("s-9", 61000, 64500)


class TestGuestStrippedFields:
    def test_absent_source_docs_are_an_empty_list_not_none(self):
        # Guests never receive source_docs. A None here would make every
        # caller guard before iterating.
        assert to_event({"done": True, "message": {"content": ""}}).source_docs == []


@pytest.mark.parametrize(
    "line",
    [
        json.dumps({"type": "stream", "message": {"content": "a"}, "done": False}),
        json.dumps({"type": "keepalive", "message": {"content": ""}}),
    ],
)
def test_every_wire_line_maps_to_something_or_nothing_without_raising(line):
    to_event(json.loads(line))
