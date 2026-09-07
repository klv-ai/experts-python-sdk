"""Reading the chat stream.

The wire format is newline-delimited JSON over plain HTTP — no WebSocket is
involved, contrary to a well-travelled note in this platform's history. Each
line is one event; the last carries ``done: true``.

Everything here is pure: an incremental line splitter fed bytes, and a mapper
from one raw event to one typed one. The sync and async clients differ only in
how they obtain the bytes, so they share all of this.

Four things are easy to get wrong by hand, and all four have bitten somebody:

1. **Keep the incomplete trailing line.** A chunk boundary falls wherever TCP
   puts it, routinely mid-JSON. Splitting on newlines and discarding the
   remainder drops a token every few hundred.
2. **The terminal chunk of a tool-assisted answer has EMPTY content on
   purpose.** The text already streamed token by token; appending it again
   renders every tool-using answer twice.
3. **Durations are nanoseconds.** Ollama's units, passed straight through.
4. **Disconnecting is not cancelling.** Generation is detached from the HTTP
   request server-side, so dropping the connection stops the relay, not the
   model.
"""

from __future__ import annotations

import json
from typing import Any

from .types import (
    ActionEvent,
    ChatEvent,
    DoneEvent,
    KeepaliveEvent,
    SourceDocument,
    ThinkingEvent,
    TokenEvent,
    Usage,
)


class LineBuffer:
    """Incremental NDJSON splitter.

    Feed it whatever bytes arrive; it yields complete objects and keeps the
    partial trailing line for the next feed. ``flush()`` at the end of a stream
    releases a final line that arrived without a newline.

    Decoding is incremental too: a chunk boundary can land inside a multi-byte
    character, and decoding each chunk independently would raise or, worse,
    silently produce a replacement character in the middle of a word.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._decoder = json.JSONDecoder()
        import codecs

        self._utf8 = codecs.getincrementaldecoder("utf-8")()

    def feed(self, chunk: bytes | str) -> list[dict[str, Any]]:
        if isinstance(chunk, bytes):
            self._buffer += self._utf8.decode(chunk)
        else:
            self._buffer += chunk

        lines = self._buffer.split("\n")
        # The last element is whatever came after the final newline. It may be
        # half an object, so it stays behind.
        self._buffer = lines.pop()
        return [parsed for line in lines if (parsed := _try_parse(line)) is not None]

    def flush(self) -> list[dict[str, Any]]:
        tail, self._buffer = self._buffer, ""
        parsed = _try_parse(tail)
        return [parsed] if parsed is not None else []


def _try_parse(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        # A line that will not parse is skipped rather than fatal: one corrupt
        # event must not discard an answer that is otherwise fine.
        return None
    return value if isinstance(value, dict) else None


def to_event(raw: dict[str, Any]) -> ChatEvent | None:
    """One raw NDJSON event as a typed one, or None to skip it."""
    kind = raw.get("type")
    message = raw.get("message") or {}
    content = message.get("content") or ""

    if kind == "keepalive":
        # Heartbeat while the server is thinking or running a tool preflight.
        # Surfaced so a UI can keep a spinner honest, never as content.
        return KeepaliveEvent()

    if kind == "thinking":
        return ThinkingEvent(content=content)

    if kind == "action":
        return ActionEvent(
            action=str(raw.get("action") or ""),
            status=str(raw.get("status") or ""),
            detail=raw.get("detail") or {},
        )

    if raw.get("done") is True:
        error = raw.get("error")
        return DoneEvent(
            # Deliberately no content. On the tool path the terminal chunk is
            # empty by design because the text already streamed; anyone
            # appending it renders tool-using answers twice.
            finish_reason=str(raw.get("done_reason") or "stop"),
            usage=Usage.from_wire(raw),
            source_docs=[
                SourceDocument.from_wire(d) for d in (raw.get("source_docs") or [])
            ],
            tools_used=list(raw.get("tools_used") or []),
            aborted=raw.get("aborted") is True,
            error=error if isinstance(error, str) and error else None,
            context_overflow=raw.get("context_overflow") is True,
            response_uid=_str_or_none(raw.get("response")),
        )

    if not content:
        return None
    return TokenEvent(content=content, response_uid=_str_or_none(raw.get("response")))


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
