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
    #:
    #: Only the expert's owner and administrators see it (build pack 1.1); for
    #: anyone else this is None.
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
    #: Load-bearing for browser sessions: a session token is a guest, and
    #: ``sessions.create()`` refuses an expert above this floor up front.
    #: Chatting with one is refused too (`ExpertNotAvailableError`); before
    #: build pack 1.1 it silently answered on the site default model instead.
    min_role_id: int = 2
    private: bool = False
    #: Org profiles this expert is limited to; empty means everyone who meets
    #: ``min_role_id``. A browser visitor belongs to none, so a limited expert
    #: cannot back a chatbox.
    scope_uids: list[UUIDStr] = field(default_factory=list)

    @property
    def guest_visible(self) -> bool:
        """Whether a browser session token could actually use this expert."""
        return not self.private and self.min_role_id <= 1 and not self.scope_uids

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
            scope_uids=[str(s) for s in (raw.get("scope_uids") or [])],
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
    #: The passage that was cited, for opening a viewer at it.
    split_uid: str | None = None
    #: For a recording: where in it the passage was said, in milliseconds.
    start_ms: int | None = None
    end_ms: int | None = None

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
            split_uid=_opt_str(raw.get("split_uid")),
            start_ms=_opt_int(raw.get("start_ms")),
            end_ms=_opt_int(raw.get("end_ms")),
        )


def _opt_str(value: Any) -> str | None:
    return str(value) if value else None


def _opt_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


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
class QueuedEvent:
    """The answer is waiting in line for another server's model (a mesh).

    Sent when the expert's model runs on another member of an Experts Mesh and
    that member is busy, each time the place in line moves. Nothing streams
    while it waits, so this is the event to show "waiting on Tokyo, 2 ahead".
    Never content.
    """

    #: Requests ahead of this one. 0 with ``rerouting`` when another route
    #: failed and the request is being sent a different way.
    ahead: int = 0
    #: The member whose model will answer: its node id and display name.
    node: str | None = None
    name: str | None = None
    rerouting: bool = False
    type: Literal["queued"] = "queued"


@dataclass(slots=True)
class ServedBy:
    """Another server's model answered (an Experts Mesh member)."""

    node: str = ""
    #: The member's display name, e.g. "Tokyo".
    name: str = ""
    label: str = ""
    model: str = ""
    #: The members the request passed through, in order, when it was relayed.
    route: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, raw: Any) -> ServedBy | None:
        if not isinstance(raw, dict):
            return None
        return cls(
            node=str(raw.get("node") or ""),
            name=str(raw.get("name") or ""),
            label=str(raw.get("label") or ""),
            model=str(raw.get("model") or ""),
            route=[
                str(r.get("name") or r.get("node") or "")
                for r in (raw.get("route") or [])
                if isinstance(r, dict)
            ],
        )


@dataclass(slots=True)
class MeshFallback:
    """The expert's own model was out of reach, so this server's answered.

    Only for an expert set to fall back; a strict one fails instead, with
    ``ChatResult.error`` set and the details in ``mesh_unavailable``.
    """

    #: The member whose model was out of reach.
    node: str = ""
    name: str = ""
    model: str = ""
    #: "unreachable", or "not_shared" when that member stopped sharing it.
    reason: str = ""
    #: The model that answered instead.
    answered_with: str = ""

    @classmethod
    def from_wire(cls, raw: Any) -> MeshFallback | None:
        if not isinstance(raw, dict):
            return None
        return cls(
            node=str(raw.get("node") or ""),
            name=str(raw.get("name") or ""),
            model=str(raw.get("model") or ""),
            reason=str(raw.get("reason") or ""),
            answered_with=str(raw.get("answered_with") or ""),
        )


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
    #: The model that answered.
    model: str | None = None
    #: Set when another mesh member's model answered.
    served_by: ServedBy | None = None
    #: Set when the expert's own mesh model was out of reach and this server's
    #: answered instead.
    mesh_fallback: MeshFallback | None = None
    #: A strict expert's mesh model was out of reach: what was unreachable.
    #: ``error`` carries the message.
    mesh_unavailable: dict[str, Any] | None = None
    #: Images a tool generated for this answer.
    generated_images: list[Any] = field(default_factory=list)
    #: Tables, charts and proposed sheet edits a tool produced, each a dict
    #: with ``id`` and ``kind`` ("query", "chart" or "sheet_proposal").
    data_artifacts: list[dict[str, Any]] = field(default_factory=list)
    type: Literal["done"] = "done"


ChatEvent = (
    TokenEvent
    | ThinkingEvent
    | ActionEvent
    | KeepaliveEvent
    | QueuedEvent
    | DoneEvent
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
    #: The model that answered, and — on an Experts Mesh — whose it was.
    model: str | None = None
    served_by: ServedBy | None = None
    mesh_fallback: MeshFallback | None = None
    mesh_unavailable: dict[str, Any] | None = None
    generated_images: list[Any] = field(default_factory=list)
    data_artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Collection:
    uid: UUIDStr
    name: str = ""
    description: str | None = None
    #: Not reported by the API, so always 0. Kept so existing code does not
    #: break; count with ``list_documents(collection=...)`` instead.
    file_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    #: Minimum role that may read it, when it is public.
    min_role_id: int = 2
    is_public: bool = False
    #: Org profiles it is limited to; empty means no limit.
    scope_uids: list[UUIDStr] = field(default_factory=list)


@dataclass(slots=True)
class KnowledgeDocument:
    uid: UUIDStr
    #: The file name, e.g. "policy.pdf".
    name: str = ""
    #: The file type, e.g. "pdf".
    type: str | None = None
    #: The summary written after indexing. Same as ``summary``.
    description: str | None = None
    #: The collection it was listed from, when ``list_documents()`` was
    #: filtered by one. A document can sit in several, and its row names none.
    collection_uid: UUIDStr | None = None
    created_at: str | None = None
    title: str | None = None
    summary: str | None = None
    #: Bytes.
    size: int | None = None


@dataclass(slots=True)
class SearchHit:
    """One passage that matched a search, with the document it came from."""

    #: The passage itself.
    text: str
    document_uid: UUIDStr = ""
    document_name: str = ""
    title: str | None = None
    summary: str | None = None
    #: Cosine distance — lower is closer.
    distance: float | None = None
    #: ``1 - distance``.
    similarity: float | None = None
    page_number: int | None = None
    #: Where in the document, when the format has one (a slide, a sheet).
    location: str | None = None
    #: For a recording: where the passage was said, in milliseconds.
    start_ms: int | None = None
    end_ms: int | None = None
    #: The reranker's score, when the install reranks.
    rerank_score: float | None = None
    #: The id of the passage.
    split_uid: str | None = None


@dataclass(slots=True)
class UploadJob:
    """An upload the server has accepted and queued for indexing."""

    #: The indexing job. Pass this object (or the id) to
    #: ``knowledge.wait_for_processing()`` or ``knowledge.upload_status()``.
    job_id: int
    message: str = ""


@dataclass(slots=True)
class UploadedFile:
    """One file in an indexing job."""

    name: str
    #: The document's uid, once the server has made one.
    document_uid: UUIDStr | None = None
    #: "pending", "working", "searchable" (embedded; the summary is still being
    #: written), "finished", or "failed".
    status: str = "pending"
    error: str | None = None

    @property
    def searchable(self) -> bool:
        return self.status in ("searchable", "finished")


@dataclass(slots=True)
class UploadStatus:
    """Where an indexing job has got to.

    Truthy when every file is searchable, so
    ``if knowledge.wait_for_processing(job):`` reads the way it sounds.
    """

    job_id: int
    files: list[UploadedFile] = field(default_factory=list)
    #: Every file has finished one way or the other: searchable or failed.
    done: bool = False

    @property
    def document_uids(self) -> list[UUIDStr]:
        return [f.document_uid for f in self.files if f.searchable and f.document_uid]

    @property
    def failed(self) -> list[UploadedFile]:
        return [f for f in self.files if f.status == "failed"]

    def __bool__(self) -> bool:
        return self.done and bool(self.files) and not self.failed


@dataclass(slots=True)
class BrowserSession:
    """A credential safe to hand to one browser. Never the API key."""

    token: str
    session_id: str
    expires_in: int
    conversation: UUIDStr
    expert_uid: UUIDStr
    expert_name: str = ""
