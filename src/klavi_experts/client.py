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
import contextlib
import time
from typing import Any, BinaryIO

import httpx

from . import _chat_stream
from ._transport import Request, SyncTransport
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

        A session token carries the guest role, so ``sessions.create()``
        refuses an expert above that floor, a private one, or one limited to
        org profiles (a visitor is in none of them). Before build pack 1.1 such
        an expert silently answered on the site default model with no persona
        and no knowledge instead. Use this to build a picker that only offers
        experts that will work.
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
        try:
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
        except ExpertsError as exc:
            # Refused before it started (an expert this key may not use, a
            # read-only licence). Nothing will ever write the placeholder, so
            # settle it with the reason, or the conversation shows it typing.
            self._settle(response_uid, exc)
            raise
        return _chat_stream.ChatStream(
            response,
            conversation_uid=conversation_uid,
            response_uid=response_uid,
            cancel=self._cancel,
        )

    def _settle(self, response_uid: str, exc: ExpertsError) -> None:
        # Best effort: the refusal is what the caller needs to see, never a
        # failure to tidy up after it.
        with contextlib.suppress(ExpertsError):
            self._t.request(_conv.settle_request(response_uid, exc.message))

    def _cancel(self, response_uid: str | None) -> bool:
        if not response_uid:
            return False
        return _conv.was_cancelled(self._t.request(_conv.cancel_request(response_uid)))

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
        self,
        *,
        collection: str | None = None,
        limit: int | None = None,
        skip: int | None = None,
        search: str | None = None,
    ) -> list[KnowledgeDocument]:
        """Documents this credential may read, optionally in one collection.

        ``search`` matches the file name, title or summary — it is a filter,
        not a semantic search; for that, use :meth:`search`.
        """
        return _knowledge.to_documents(
            self._t.request(
                _knowledge.documents_request(
                    collection=collection, limit=limit, skip=skip, search=search
                )
            ),
            collection=collection,
        )

    def upload(
        self, file: BinaryIO, *, collection: str, filename: str = "upload"
    ) -> UploadJob:
        """Upload a file into a collection.

        Returns as soon as the file is accepted, with the indexing job it
        joined. It is NOT searchable yet — the worker still has to split,
        embed and index it. Follow with :meth:`wait_for_processing` if the
        next thing you do is query; that is also where the new document's uid
        and any read failure come back.
        """
        return _knowledge.to_upload_job(
            self._t.request(
                _knowledge.upload_request(
                    file=file, filename=filename, collection=collection
                )
            )
        )

    def upload_status(self, job: UploadJob | int) -> UploadStatus:
        """Where an upload's indexing job has got to, without waiting."""
        job_id = _knowledge.job_id_of(job)
        return _knowledge.to_upload_status(
            job_id, self._t.request(_knowledge.upload_status_request(job_id))
        )

    def wait_for_processing(
        self, job: UploadJob | int, *, timeout: float = 120.0, interval: float = 2.0
    ) -> UploadStatus:
        """Poll until every file in the job is searchable or has failed.

        Returns the last status either way: truthy when every file is
        searchable, falsy on a failure or on timeout (check ``.done`` and
        ``.failed`` to tell which). There is no push signal for this, so
        polling is the honest answer rather than a workaround.

        Large files take longer than the default allows, and a recording is
        transcribed before it becomes a document at all — raise ``timeout``
        for those.
        """
        deadline = time.monotonic() + timeout
        while True:
            status = self.upload_status(job)
            if status.done or time.monotonic() >= deadline:
                return status
            time.sleep(interval)

    def search(
        self,
        query: str,
        *,
        collections: list[str] | None = None,
        limit: int = 8,
    ) -> list[SearchHit]:
        """The passages that best match ``query``, closest first.

        Searches every collection this credential may read unless
        ``collections`` narrows it.
        """
        return _knowledge.to_hits(
            self._t.request(
                _knowledge.search_request(query, collections=collections, limit=limit)
            )
        )

    def delete_document(self, uid: str) -> None:
        """Delete a document, from every collection it is in."""
        self._t.request(_knowledge.delete_document_request(uid))


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

    ``stream_timeout`` is the longest silence, in seconds, tolerated between
    two pieces of a streamed answer. On an Experts Mesh a turn can wait in
    line for a busy member's model without a word; raise it if yours do.
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
        stream_timeout: float = 300.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._transport = SyncTransport(
            base_url=base_url,
            api_key=api_key,
            token=token,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            stream_timeout=stream_timeout,
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
