"""The synchronous client.

Holds the ``sk-`` API key, which is a full user identity on the install — so it
belongs on a server and nowhere else. To put a chatbox on a public page, use
``sessions.create()`` to mint a short-lived browser token and hand THAT to the
page; the browser then talks to the install directly.

For asyncio, use :class:`klavi_experts.AsyncExpertsClient`. The two are the
same API; only the awaiting differs.
"""

from __future__ import annotations

# `Experts.list` and `Conversations.list` shadow the builtin INSIDE their class
# bodies, so a bare `list[...]` annotation there resolves to the method rather
# than to the type. Harmless at runtime under `from __future__ import
# annotations`, but it breaks `typing.get_type_hints()` and anything built on
# it. `builtins.list` says what is meant, and keeps `.list()` as the method
# name every other SDK uses.
import builtins
import time
from typing import Any, BinaryIO

import httpx

from . import _chat_stream
from ._transport import Request, SyncTransport
from .errors import NotFoundError, StreamError
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
)


class Experts:
    def __init__(self, transport: SyncTransport) -> None:
        self._t = transport

    def list(self) -> builtins.list[Expert]:
        """Every expert this credential may use."""
        return _experts.to_experts(self._t.request(_experts.list_request()))

    def get(self, uid: str) -> Expert:
        for expert in self.list():
            if expert.uid == uid:
                return expert
        raise NotFoundError(f"Expert {uid} not found", status=404)

    def list_guest_visible(self) -> builtins.list[Expert]:
        """Experts a browser session may actually use.

        A session token carries the guest role, and an expert above that floor
        resolves to nothing server-side — the chat then answers on the site
        default model with no persona and no knowledge, silently. Use this to
        build a picker that only offers experts that will work.
        """
        return [e for e in self.list() if e.guest_visible]


class Conversations:
    def __init__(self, transport: SyncTransport) -> None:
        self._t = transport

    def create(
        self,
        *,
        expert: str | None = None,
        title: str = "",
        model: str = "",
        initial_question: str = "",
        hidden: bool = False,
    ) -> Conversation:
        return _conv.to_conversation(
            self._t.request(
                _conv.start_request(
                    expert=expert,
                    title=title,
                    model=model,
                    initial_question=initial_question,
                    hidden=hidden,
                )
            )
        )

    def list(
        self, *, skip: int = 0, limit: int = 50, active: bool = True
    ) -> builtins.list[Conversation]:
        return _conv.to_conversations(
            self._t.request(_conv.list_request(skip=skip, limit=limit, active=active))
        )

    def get(self, uid: str) -> ConversationWithMessages:
        return _conv.to_conversation_with_messages(
            self._t.request(_conv.get_request(uid))
        )

    def update(self, uid: str, **changes: Any) -> Conversation:
        return _conv.to_conversation(
            self._t.request(_conv.update_request(uid, changes))
        )

    def delete(self, uid: str) -> None:
        """Soft delete — the row is deactivated, not destroyed."""
        self._t.request(_conv.delete_request(uid))

    def fork(
        self,
        uid: str,
        *,
        title: str | None = None,
        response_uid: str | None = None,
        target_user: str | None = None,
    ) -> Conversation:
        return _conv.to_conversation(
            self._t.request(
                _conv.fork_request(
                    uid,
                    title=title,
                    response_uid=response_uid,
                    target_user=target_user,
                )
            )
        )

    def send(
        self,
        conversation_uid: str,
        question: str,
        *,
        expert: str | None = None,
        model: str | None = None,
        images: builtins.list[str] | None = None,
        context: builtins.list[str] | None = None,
    ) -> _chat_stream.ChatStream:
        """Send a message and stream the reply.

        Wraps the three-call sequence documented in
        :mod:`klavi_experts.resources.conversations`. The returned stream is
        iterable for events and has ``.text()`` for the whole reply.
        """
        # 1. The user's turn, already settled.
        self._t.request(
            _conv.user_turn_request(
                conversation_uid, question, images=images, context=context
            )
        )
        # 2. The assistant placeholder. Its uid is what /chat writes into, and
        #    what makes cancelling possible.
        placeholder = self._t.request(
            _conv.placeholder_request(conversation_uid, question, expert=expert)
        )
        response_uid = str((placeholder or {}).get("uid") or "")
        if not response_uid:
            raise StreamError("The server did not return a placeholder response uid")

        # 3. Generate. Streams NDJSON over plain HTTP.
        response = self._t.send(
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
        return _chat_stream.ChatStream(
            response,
            conversation_uid=conversation_uid,
            response_uid=response_uid,
            cancel=self._cancel,
        )

    def _cancel(self, response_uid: str | None) -> None:
        if response_uid:
            self._t.request(_conv.cancel_request(response_uid))

    def ask(
        self,
        question: str,
        *,
        expert: str | None = None,
        model: str | None = None,
        title: str = "",
        **kw: Any,
    ) -> _chat_stream.ChatStream:
        """Create a conversation and send the first message in one call."""
        conversation = self.create(
            expert=expert, model=model or "", title=title, initial_question=question
        )
        return self.send(conversation.uid, question, expert=expert, model=model, **kw)


class Knowledge:
    def __init__(self, transport: SyncTransport) -> None:
        self._t = transport

    def list_collections(self) -> list[Collection]:
        return _knowledge.to_collections(
            self._t.request(_knowledge.collections_request())
        )

    def list_documents(
        self, *, collection: str | None = None, limit: int | None = None
    ) -> list[KnowledgeDocument]:
        return _knowledge.to_documents(
            self._t.request(
                _knowledge.documents_request(collection=collection, limit=limit)
            )
        )

    def upload(
        self, file: BinaryIO, *, collection: str, filename: str = "upload"
    ) -> KnowledgeDocument:
        """Upload a file into a collection.

        Returns as soon as the file is accepted. It is NOT searchable yet — the
        worker still has to split, embed and index it. Follow with
        :meth:`wait_for_processing` if the next thing you do is query.
        """
        response = self._t.send(
            _knowledge.upload_request(
                file=file, filename=filename, collection=collection
            )
        )
        try:
            response.read()
            return _knowledge.to_document(_knowledge.unwrap(response.json()))
        finally:
            response.close()

    def search(
        self,
        query: str,
        *,
        collections: list[str] | None = None,
        limit: int = 8,
    ) -> list[KnowledgeDocument]:
        return _knowledge.to_documents(
            self._t.request(
                _knowledge.search_request(query, collections=collections, limit=limit)
            )
        )

    def wait_for_processing(
        self, uid: str, *, timeout: float = 120.0, interval: float = 2.0
    ) -> bool:
        """Poll until a document is embedded. True when ready, False on timeout.

        There is no push signal for this, so polling is the honest answer
        rather than a workaround.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self._t.request(_knowledge.document_request(uid))
            if _knowledge.is_processed(raw):
                return True
            time.sleep(interval)
        return False


class Sessions:
    def __init__(self, transport: SyncTransport) -> None:
        self._t = transport

    def create(
        self,
        *,
        expert: str,
        origin: str,
        conversation: str | None = None,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BrowserSession:
        """Mint a token to hand to one browser."""
        return _sessions.to_session(
            self._t.request(
                _sessions.create_request(
                    expert=expert,
                    origin=origin,
                    conversation=conversation,
                    ttl=ttl,
                    metadata=metadata,
                )
            )
        )

    def revoke(self, session_id: str) -> None:
        """Kill a session immediately, before its token would expire."""
        self._t.request(_sessions.revoke_request(session_id))


class ExpertsClient:
    """Talk to a Klavi Experts installation.

    ``api_key`` is a full user identity on the install. Keep it on a server.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        token: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._transport = SyncTransport(
            base_url=base_url,
            api_key=api_key,
            token=token,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            client=http_client,
        )
        self.experts = Experts(self._transport)
        self.conversations = Conversations(self._transport)
        self.knowledge = Knowledge(self._transport)
        self.sessions = Sessions(self._transport)

    @property
    def transport(self) -> SyncTransport:
        """Escape hatch for endpoints this SDK does not wrap yet."""
        return self._transport

    def request(self, method: str, path: str, **kw: Any) -> Any:
        """Call an endpoint directly, with auth and error handling applied."""
        return self._transport.request(Request(method=method, path=path, **kw))

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> ExpertsClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
