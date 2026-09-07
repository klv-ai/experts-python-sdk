"""A live answer, sync and async.

Iterate it for typed events, or call ``.text()`` for the whole reply. Both
shapes accumulate the same result, so the aggregation logic lives in
``_Accumulator`` and the two classes below differ only in how they pull bytes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import httpx

from ._stream import LineBuffer, to_event
from .errors import StreamError
from .types import ChatEvent, ChatResult, DoneEvent, ThinkingEvent, TokenEvent


class _Accumulator:
    """Builds the ChatResult as events go past. Shared by both stream classes."""

    def __init__(self, conversation_uid: str, response_uid: str | None) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self.conversation_uid = conversation_uid
        self.response_uid = response_uid
        self.result: ChatResult | None = None
        self.saw_done = False

    def observe(self, event: ChatEvent) -> None:
        if isinstance(event, TokenEvent):
            self._text.append(event.content)
        elif isinstance(event, ThinkingEvent):
            self._reasoning.append(event.content)
        elif isinstance(event, DoneEvent):
            self.saw_done = True
            self.result = ChatResult(
                text="".join(self._text),
                reasoning="".join(self._reasoning),
                conversation_uid=self.conversation_uid,
                response_uid=event.response_uid or self.response_uid,
                usage=event.usage,
                source_docs=event.source_docs,
                tools_used=event.tools_used,
                finish_reason=event.finish_reason,
                aborted=event.aborted,
                error=event.error,
            )

    def incomplete(self) -> StreamError:
        # The answer very likely completed on the server regardless —
        # generation is detached — so say what recovery actually looks like.
        return StreamError(
            "The stream ended without a terminal event. The answer may still "
            "have completed on the server: re-read the conversation rather "
            "than resending.",
            code="incomplete_stream",
        )


class ChatStream:
    """A streaming answer.

    Iterate for events; ``text()`` and ``result()`` drain and aggregate.
    Usable as a context manager so the connection is released even if you stop
    reading part-way.
    """

    def __init__(
        self,
        response: httpx.Response,
        *,
        conversation_uid: str,
        response_uid: str | None,
        cancel: Callable[[str | None], None],
    ) -> None:
        self._response = response
        self._cancel = cancel
        self._acc = _Accumulator(conversation_uid, response_uid)
        self._consumed = False
        self._detached = False
        self._cancelled = False

    @property
    def conversation_uid(self) -> str:
        return self._acc.conversation_uid

    @property
    def response_uid(self) -> str | None:
        """The assistant message row this answer is being written into."""
        return self._acc.response_uid

    def __enter__(self) -> ChatStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[ChatEvent]:
        if self._consumed:
            raise StreamError("This stream has already been consumed")
        self._consumed = True

        buffer = LineBuffer()
        try:
            for chunk in self._response.iter_bytes():
                for raw in buffer.feed(chunk):
                    event = to_event(raw)
                    if event is None:
                        continue
                    self._acc.observe(event)
                    yield event
                    if isinstance(event, DoneEvent):
                        return
            for raw in buffer.flush():
                event = to_event(raw)
                if event is None:
                    continue
                self._acc.observe(event)
                yield event
        finally:
            self._response.close()

        if not self._acc.saw_done and not self._cancelled and not self._detached:
            raise self._acc.incomplete()

    def result(self) -> ChatResult:
        """Drain the stream and return the finished answer."""
        if self._acc.result is None:
            for _ in self:
                pass
        if self._acc.result is None:
            raise StreamError("The stream produced no result")
        return self._acc.result

    def text(self) -> str:
        """The assistant's reply as a string."""
        return self.result().text

    def cancel(self) -> None:
        """Stop the model.

        Not the same as walking away from the stream: generation is detached
        server-side, so an abandoned request keeps generating, keeps costing
        and still persists its answer. This is the only thing that stops it.
        """
        self._cancelled = True
        self.close()
        self._cancel(self.response_uid)

    def detach(self) -> None:
        """Stop reading, deliberately leaving the answer to finish server-side.

        For "the user navigated away but should see the reply when they come
        back" — it persists and can be read from the conversation later.
        """
        self._detached = True
        self.close()

    def close(self) -> None:
        self._response.close()


class AsyncChatStream:
    """The async twin of :class:`ChatStream`."""

    def __init__(
        self,
        response: httpx.Response,
        *,
        conversation_uid: str,
        response_uid: str | None,
        cancel: Callable[[str | None], Awaitable[None]],
    ) -> None:
        self._response = response
        self._cancel = cancel
        self._acc = _Accumulator(conversation_uid, response_uid)
        self._consumed = False
        self._detached = False
        self._cancelled = False

    @property
    def conversation_uid(self) -> str:
        return self._acc.conversation_uid

    @property
    def response_uid(self) -> str | None:
        return self._acc.response_uid

    async def __aenter__(self) -> AsyncChatStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def __aiter__(self) -> AsyncIterator[ChatEvent]:
        if self._consumed:
            raise StreamError("This stream has already been consumed")
        self._consumed = True

        buffer = LineBuffer()
        try:
            async for chunk in self._response.aiter_bytes():
                for raw in buffer.feed(chunk):
                    event = to_event(raw)
                    if event is None:
                        continue
                    self._acc.observe(event)
                    yield event
                    if isinstance(event, DoneEvent):
                        return
            for raw in buffer.flush():
                event = to_event(raw)
                if event is None:
                    continue
                self._acc.observe(event)
                yield event
        finally:
            await self._response.aclose()

        if not self._acc.saw_done and not self._cancelled and not self._detached:
            raise self._acc.incomplete()

    async def result(self) -> ChatResult:
        if self._acc.result is None:
            async for _ in self:
                pass
        if self._acc.result is None:
            raise StreamError("The stream produced no result")
        return self._acc.result

    async def text(self) -> str:
        return (await self.result()).text

    async def cancel(self) -> None:
        self._cancelled = True
        await self.aclose()
        await self._cancel(self.response_uid)

    def detach(self) -> None:
        self._detached = True

    async def aclose(self) -> None:
        await self._response.aclose()
