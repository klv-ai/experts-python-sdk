"""Wire types.

Naming note: the API is inconsistent by design, and this SDK is where that
stops. Conversation rows come back camelCased; message rows and expert rows
come back snake_case (``source_docs``, ``min_role_id``); the knowledge service
wraps everything in ``{timestamp, status, count, data}``. All three are
normalised here so a caller never has to know which service answered.

Plain dataclasses rather than pydantic models, so the package pulls in nothing
but httpx and works the same inside a project pinned to pydantic v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

UUIDStr = str


@dataclass(slots=True)
class Expert:
    uid: UUIDStr
    name: str
    description: str | None = None
    #: The persona/system prompt. ``model_file`` on the wire.
    instructions: str | None = None
    model: str | None = None
    avatar: str | None = None
    #: Suggested opening questions, for a picker UI.
    starters: list[str] = field(default_factory=list)
    #: Knowledge collections this expert can retrieve from.
    collections: list[UUIDStr] = field(default_factory=list)
    temperature: float | None = None
    voice: str | None = None
    output_language: str | None = None
    #: Minimum role that may use this expert. 1 = guest.
    #:
    #: Load-bearing for browser sessions: a session token is a guest, and an
    #: expert above this floor silently resolves to the site default model with
    #: no persona. ``sessions.create()`` refuses such an expert up front.
    min_role_id: int = 2
    private: bool = False

    @property
    def guest_visible(self) -> bool:
        """Whether a browser session token could actually use this expert."""
        return not self.private and self.min_role_id <= 1

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> Expert:
        return cls(
            uid=str(raw.get("uid", "")),
            name=raw.get("name") or "",
            description=raw.get("description"),
            instructions=raw.get("model_file"),
            model=raw.get("model"),
            avatar=raw.get("avatar"),
            starters=list(raw.get("starters") or []),
            collections=[str(c) for c in (raw.get("collections") or [])],
            temperature=raw.get("temperature"),
            voice=raw.get("voice"),
            output_language=raw.get("output_language"),
            min_role_id=int(raw.get("min_role_id") or 2),
            private=bool(raw.get("private")),
        )


@dataclass(slots=True)
class Conversation:
    uid: UUIDStr
    title: str = ""
    model: str | None = None
    initial_question: str = ""
    #: Expert attached to this conversation, if any.
    expert_uid: UUIDStr | None = None
    active: bool = True
    #: Hidden conversations are excluded from list views, not from access.
    hidden: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    forked_from: UUIDStr | None = None
    forked_by_name: str | None = None

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> Conversation:
        return cls(
            uid=str(raw.get("uid", "")),
            title=raw.get("title") or "",
            model=raw.get("model"),
            initial_question=raw.get("initialQuestion") or "",
            expert_uid=raw.get("modelProfile"),
            active=bool(raw.get("active", True)),
            hidden=bool(raw.get("hidden", False)),
            created_at=raw.get("createdAt"),
            updated_at=raw.get("updatedAt"),
            forked_from=raw.get("forkedFrom"),
            forked_by_name=raw.get("forkedByName"),
        )


@dataclass(slots=True)
class SourceDocument:
    uid: str
    name: str = ""
    title: str = ""
    type: str = ""
    #: The retrieved passage, not the whole document.
    text: str = ""
    description: str = ""
    #: Cosine distance — lower is closer.
    distance: float | None = None
    #: ``1 - distance``, because "distance" reads backwards when ranking.
    similarity: float | None = None

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> SourceDocument:
        distance = raw.get("cosine_dist")
        return cls(
            uid=str(raw.get("uid", "")),
            name=raw.get("name") or "",
            title=raw.get("title") or "",
            type=raw.get("type") or "",
            text=raw.get("text") or "",
            description=raw.get("description") or "",
            distance=distance if isinstance(distance, (int, float)) else None,
            similarity=(1 - distance) if isinstance(distance, (int, float)) else None,
        )


@dataclass(slots=True)
class Message:
    uid: UUIDStr
    #: True for the assistant, false for the user.
    agent: bool = True
    text: str = ""
    question: str | None = None
    #: False while an answer is still being written.
    done: bool = True
    index: int = 0
    created_at: str | None = None
    source_docs: list[SourceDocument] = field(default_factory=list)
    error: str | None = None
    aborted: bool = False

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> Message:
        return cls(
            uid=str(raw.get("uid", "")),
            agent=bool(raw.get("agent", True)),
            text=raw.get("text") or "",
            question=raw.get("question"),
            done=bool(raw.get("done", True)),
            index=int(raw.get("index") or 0),
            # snake_case here, unlike the conversation row wrapping it: message
            # rows come from a raw SQL projection the frontend reads directly.
            created_at=raw.get("created_at"),
            source_docs=[
                SourceDocument.from_wire(d) for d in (raw.get("source_docs") or [])
            ],
            error=raw.get("error"),
            aborted=bool(raw.get("aborted", False)),
        )


@dataclass(slots=True)
class ConversationWithMessages:
    conversation: Conversation
    messages: list[Message] = field(default_factory=list)

    # Convenience passthroughs, so callers are not forced through `.conversation`
    # for the field they were already thinking about.
    @property
    def uid(self) -> UUIDStr:
        return self.conversation.uid

    @property
    def title(self) -> str:
        return self.conversation.title


NS_PER_MS = 1_000_000


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    #: Milliseconds. The wire carries NANOSECONDS; converted once, here.
    total_ms: float | None = None
    prompt_ms: float | None = None
    eval_ms: float | None = None
    load_ms: float | None = None

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> Usage:
        def ms(key: str) -> float | None:
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return round(value / NS_PER_MS, 2)
            return None

        prompt = int(raw.get("prompt_eval_count") or 0)
        completion = int(raw.get("eval_count") or 0)
        reasoning = raw.get("reasoning_tokens")
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            reasoning_tokens=int(reasoning) if isinstance(reasoning, int) else None,
            total_ms=ms("total_duration"),
            prompt_ms=ms("prompt_eval_duration"),
            eval_ms=ms("eval_duration"),
            load_ms=ms("load_duration"),
        )


# --- stream events -------------------------------------------------------


@dataclass(slots=True)
class TokenEvent:
    """A piece of the answer. Concatenate these."""

    content: str
    response_uid: str | None = None
    type: Literal["token"] = "token"


@dataclass(slots=True)
class ThinkingEvent:
    """Reasoning tokens, when the model exposes them and the expert allows it."""

    content: str
    type: Literal["thinking"] = "thinking"


@dataclass(slots=True)
class ActionEvent:
    """Progress: retrieval, tool calls, compression. Useful for a status line."""

    action: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)
    type: Literal["action"] = "action"


@dataclass(slots=True)
class KeepaliveEvent:
    """Heartbeat. Never content — surfaced so a spinner can stay honest."""

    type: Literal["keepalive"] = "keepalive"


@dataclass(slots=True)
class DoneEvent:
    """The end.

    Carries no content on purpose: on the tool path the text already streamed,
    so a terminal chunk with content would render the answer twice.
    """

    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    source_docs: list[SourceDocument] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    aborted: bool = False
    error: str | None = None
    context_overflow: bool = False
    response_uid: str | None = None
    type: Literal["done"] = "done"


ChatEvent = (
    TokenEvent | ThinkingEvent | ActionEvent | KeepaliveEvent | DoneEvent
)


@dataclass(slots=True)
class ChatResult:
    text: str = ""
    #: Reasoning tokens, if any were streamed.
    reasoning: str = ""
    conversation_uid: UUIDStr = ""
    response_uid: str | None = None
    usage: Usage = field(default_factory=Usage)
    source_docs: list[SourceDocument] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    finish_reason: str = "stop"
    aborted: bool = False
    #: Set when generation failed *inside* a 200 response.
    #:
    #: The stream returns HTTP 200 and then reports trouble in its terminal
    #: event, so a caller checking only the status code sees an empty answer
    #: and no reason. Always worth checking.
    error: str | None = None


@dataclass(slots=True)
class Collection:
    uid: UUIDStr
    name: str = ""
    description: str | None = None
    file_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class KnowledgeDocument:
    uid: UUIDStr
    name: str = ""
    type: str | None = None
    description: str | None = None
    collection_uid: UUIDStr | None = None
    created_at: str | None = None


@dataclass(slots=True)
class BrowserSession:
    """A credential safe to hand to one browser. Never the API key."""

    token: str
    session_id: str
    expires_in: int
    conversation: UUIDStr
    expert_uid: UUIDStr
    expert_name: str = ""
