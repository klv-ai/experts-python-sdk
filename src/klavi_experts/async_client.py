"""The asyncio client.

Same API as :class:`klavi_experts.ExpertsClient`; only the awaiting differs.
Everything that is not I/O — request building, response shaping, NDJSON
framing, event mapping — is shared with the sync client rather than written
twice, so the two cannot drift in what they mean, only in how they wait.
"""

from __future__ import annotations

import asyncio

# `Experts.list` and `Conversations.list` shadow the builtin INSIDE their class
# bodies, so a bare `list[...]` annotation there resolves to the method rather
# than to the type. Harmless at runtime under `from __future__ import
# annotations`, but it breaks `typing.get_type_hints()` and anything built on
# it. `builtins.list` says what is meant, and keeps `.list()` as the method
# name every other SDK uses.
import builtins
import contextlib
from typing import Any, BinaryIO

import httpx

from . import _chat_stream
from ._transport import AsyncTransport, Request
from .errors import ExpertsError, NotFoundError, StreamError
from .resources import conversations as _conv
from .resources import experts as _experts
from .resources import knowledge as _knowledge
from .resources import sessions as _sessions
from .types import (
    BrowserSession,
    Collection,
    Conversation,
    ConversationWithMessages,
    Expert,
    KnowledgeDocument,
    SearchHit,
    UploadJob,
    UploadStatus,
)


class AsyncExperts:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def list(self) -> builtins.list[Expert]:
        return _experts.to_experts(await self._t.request(_experts.list_request()))

    async def get(self, uid: str) -> Expert:
        for expert in await self.list():
            if expert.uid == uid:
                return expert
        raise NotFoundError(f"Expert {uid} not found", status=404)

    async def list_guest_visible(self) -> builtins.list[Expert]:
        """Experts a browser session may actually use — see the sync twin."""
        return [e for e in await self.list() if e.guest_visible]


class AsyncConversations:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def create(
        self,
        *,
        expert: str | None = None,
        title: str = "",
        model: str = "",
        initial_question: str = "",
        hidden: bool = False,
    ) -> Conversation:
        return _conv.to_conversation(
            await self._t.request(
                _conv.start_request(
                    expert=expert,
                    title=title,
                    model=model,
                    initial_question=initial_question,
                    hidden=hidden,
                )
            )
        )

    async def list(
        self, *, skip: int = 0, limit: int = 50, active: bool = True
    ) -> builtins.list[Conversation]:
        return _conv.to_conversations(
            await self._t.request(
                _conv.list_request(skip=skip, limit=limit, active=active)
            )
        )

    async def get(self, uid: str) -> ConversationWithMessages:
        return _conv.to_conversation_with_messages(
            await self._t.request(_conv.get_request(uid))
        )

    async def update(self, uid: str, **changes: Any) -> Conversation:
        return _conv.to_conversation(
            await self._t.request(_conv.update_request(uid, changes))
        )

    async def delete(self, uid: str) -> None:
        await self._t.request(_conv.delete_request(uid))

    async def fork(
        self,
        uid: str,
        *,
        title: str | None = None,
        response_uid: str | None = None,
        target_user: str | None = None,
    ) -> Conversation:
        return _conv.to_conversation(
            await self._t.request(
                _conv.fork_request(
                    uid,
                    title=title,
                    response_uid=response_uid,
                    target_user=target_user,
                )
            )
        )

    async def send(
        self,
        conversation_uid: str,
        question: str,
        *,
        expert: str | None = None,
        model: str | None = None,
        images: builtins.list[str] | None = None,
        context: builtins.list[str] | None = None,
    ) -> _chat_stream.AsyncChatStream:
        """Send a message and stream the reply. Three calls — see the sync twin."""
        await self._t.request(
            _conv.user_turn_request(
                conversation_uid, question, images=images, context=context
            )
        )
        placeholder = await self._t.request(
            _conv.placeholder_request(conversation_uid, question, expert=expert)
        )
        response_uid = str((placeholder or {}).get("uid") or "")
        if not response_uid:
            raise StreamError("The server did not return a placeholder response uid")

        try:
            response = await self._t.send(
                _conv.chat_request(
                    conversation_uid,
                    question,
                    response_uid=response_uid,
                    expert=expert,
                    model=model,
                    images=images,
                    context=context,
                )
            )
        except ExpertsError as exc:
            # Refused before it started: settle the placeholder, as the sync
            # twin does, or the conversation shows it typing forever.
            await self._settle(response_uid, exc)
            raise
        return _chat_stream.AsyncChatStream(
            response,
            conversation_uid=conversation_uid,
            response_uid=response_uid,
            cancel=self._cancel,
        )

    async def _settle(self, response_uid: str, exc: ExpertsError) -> None:
        with contextlib.suppress(ExpertsError):
            await self._t.request(_conv.settle_request(response_uid, exc.message))

    async def _cancel(self, response_uid: str | None) -> bool:
        if not response_uid:
            return False
        return _conv.was_cancelled(
            await self._t.request(_conv.cancel_request(response_uid))
        )

    async def ask(
        self,
        question: str,
        *,
        expert: str | None = None,
        model: str | None = None,
        title: str = "",
        **kw: Any,
    ) -> _chat_stream.AsyncChatStream:
        conversation = await self.create(
            expert=expert, model=model or "", title=title, initial_question=question
        )
        return await self.send(
            conversation.uid, question, expert=expert, model=model, **kw
        )


class AsyncKnowledge:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def list_collections(self) -> list[Collection]:
        return _knowledge.to_collections(
            await self._t.request(_knowledge.collections_request())
        )

    async def list_documents(
        self,
        *,
        collection: str | None = None,
        limit: int | None = None,
        skip: int | None = None,
        search: str | None = None,
    ) -> list[KnowledgeDocument]:
        """Documents this credential may read — see the sync twin."""
        return _knowledge.to_documents(
            await self._t.request(
                _knowledge.documents_request(
                    collection=collection, limit=limit, skip=skip, search=search
                )
            ),
            collection=collection,
        )

    async def upload(
        self, file: BinaryIO, *, collection: str, filename: str = "upload"
    ) -> UploadJob:
        """Accepted and queued, not yet searchable — see the sync twin."""
        return _knowledge.to_upload_job(
            await self._t.request(
                _knowledge.upload_request(
                    file=file, filename=filename, collection=collection
                )
            )
        )

    async def upload_status(self, job: UploadJob | int) -> UploadStatus:
        job_id = _knowledge.job_id_of(job)
        return _knowledge.to_upload_status(
            job_id, await self._t.request(_knowledge.upload_status_request(job_id))
        )

    async def wait_for_processing(
        self, job: UploadJob | int, *, timeout: float = 120.0, interval: float = 2.0
    ) -> UploadStatus:
        """Poll until every file is searchable or has failed — see the sync twin."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            status = await self.upload_status(job)
            if status.done or loop.time() >= deadline:
                return status
            await asyncio.sleep(interval)

    async def search(
        self, query: str, *, collections: list[str] | None = None, limit: int = 8
    ) -> list[SearchHit]:
        return _knowledge.to_hits(
            await self._t.request(
                _knowledge.search_request(query, collections=collections, limit=limit)
            )
        )

    async def delete_document(self, uid: str) -> None:
        await self._t.request(_knowledge.delete_document_request(uid))


class AsyncSessions:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def create(
        self,
        *,
        expert: str,
        origin: str,
        conversation: str | None = None,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BrowserSession:
        return _sessions.to_session(
            await self._t.request(
                _sessions.create_request(
                    expert=expert,
                    origin=origin,
                    conversation=conversation,
                    ttl=ttl,
                    metadata=metadata,
                )
            )
        )

    async def revoke(self, session_id: str) -> None:
        await self._t.request(_sessions.revoke_request(session_id))


class AsyncExpertsClient:
    """Talk to a Klavi Experts installation, from asyncio."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        token: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
        stream_timeout: float = 300.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._transport = AsyncTransport(
            base_url=base_url,
            api_key=api_key,
            token=token,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            stream_timeout=stream_timeout,
            client=http_client,
        )
        self.experts = AsyncExperts(self._transport)
        self.conversations = AsyncConversations(self._transport)
        self.knowledge = AsyncKnowledge(self._transport)
        self.sessions = AsyncSessions(self._transport)

    @property
    def transport(self) -> AsyncTransport:
        return self._transport

    async def request(self, method: str, path: str, **kw: Any) -> Any:
        return await self._transport.request(Request(method=method, path=path, **kw))

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncExpertsClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
