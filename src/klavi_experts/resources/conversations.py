"""Conversations, and sending a message.

``send()`` is the reason this SDK exists. On the raw API, one chat turn is
THREE calls in a specific order, and getting it wrong fails silently rather
than loudly::

    POST /api/v1/responses          the user's turn      {agent: False, done: True}
    POST /api/v1/responses          assistant placeholder {agent: True, done: False}
    POST /api/v1/conversations/chat {response: <placeholder uid>, ...}

``/chat`` never creates message rows — it UPDATES the placeholder you made.
Skip the first two calls and the model still answers and the stream still looks
fine, but nothing is persisted and every event carries an empty ``response``,
so the conversation is empty when you reload it. That is the most expensive
thing to discover on your own, and it is why this method exists.

The request BUILDING below is shared by the sync and async resources; only the
awaiting differs.
"""

from __future__ import annotations

from typing import Any

from .._transport import Request
from ..types import (
    Conversation,
    ConversationWithMessages,
    Message,
)

# --- pure request builders, shared by both flavours ---------------------


def start_request(
    *,
    expert: str | None,
    title: str,
    model: str,
    initial_question: str,
    hidden: bool,
) -> Request:
    return Request(
        method="POST",
        path="/api/v1/conversations/start",
        json={
            "title": title,
            # The server requires the field; an expert's own model wins anyway.
            "model": model,
            "initialQuestion": initial_question,
            "configId": 1,
            "modelProfile": expert,
            "hidden": hidden,
        },
    )


def list_request(*, skip: int, limit: int, active: bool) -> Request:
    # The trailing slash matters: without it the gateway answers a 307.
    return Request(
        path="/api/v1/conversations/",
        params={"skip": skip, "limit": limit, "active": active},
    )


def get_request(uid: str) -> Request:
    return Request(path=f"/api/v1/conversations/{uid}")


def update_request(uid: str, changes: dict[str, Any]) -> Request:
    return Request(
        method="PUT", path="/api/v1/conversations/", json={"uid": uid, **changes}
    )


def delete_request(uid: str) -> Request:
    # A body-bearing DELETE. Unusual, but it is what the API takes.
    return Request(method="DELETE", path="/api/v1/conversations/", json={"uid": uid})


def fork_request(
    uid: str, *, title: str | None, response_uid: str | None, target_user: str | None
) -> Request:
    return Request(
        method="POST",
        path=f"/api/v1/conversations/{uid}/fork",
        json={
            "title": title,
            "responseUid": response_uid,
            "targetUser": target_user,
        },
    )


def user_turn_request(
    conversation_uid: str,
    question: str,
    *,
    images: list[str] | None,
    context: list[str] | None,
) -> Request:
    return Request(
        method="POST",
        path="/api/v1/responses/",
        json={
            "text": question,
            "conversation": conversation_uid,
            "agent": False,
            "done": True,
            "question": "",
            "images": images or [],
            "context": context or [],
        },
        # Never retried: a duplicate here posts the question twice.
        idempotent=False,
    )


def placeholder_request(
    conversation_uid: str, question: str, *, expert: str | None
) -> Request:
    return Request(
        method="POST",
        path="/api/v1/responses/",
        json={
            "text": "",
            "conversation": conversation_uid,
            "agent": True,
            "done": False,
            "question": question,
            "profileId": expert,
            "context": [],
        },
        idempotent=False,
    )


def chat_request(
    conversation_uid: str,
    question: str,
    *,
    response_uid: str,
    expert: str | None,
    model: str | None,
    images: list[str] | None,
    context: list[str] | None,
) -> Request:
    return Request(
        method="POST",
        path="/api/v1/conversations/chat",
        json={
            "conversationUid": conversation_uid,
            "question": question,
            "response": response_uid,
            "expert": expert,
            "model": model,
            "images": images,
            "context": context,
        },
        stream=True,
        headers={"accept": "application/x-ndjson"},
    )


def cancel_request(response_uid: str) -> Request:
    return Request(
        method="POST",
        path=f"/api/v1/conversations/chat/{response_uid}/cancel",
        idempotent=False,
    )


# --- response shaping ----------------------------------------------------


def to_conversation(raw: Any) -> Conversation:
    return Conversation.from_wire(raw or {})


def to_conversation_with_messages(raw: Any) -> ConversationWithMessages:
    raw = raw or {}
    return ConversationWithMessages(
        conversation=Conversation.from_wire(raw),
        messages=[Message.from_wire(m) for m in (raw.get("messages") or [])],
    )


def to_conversations(raw: Any) -> list[Conversation]:
    return [Conversation.from_wire(c) for c in (raw or [])]
