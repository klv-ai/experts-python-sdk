"""klavi-experts — the Klavi Experts API for Python.

For servers. An API key is a full user identity on an install, so it belongs
here and not in anything a visitor can read. For a browser, mint a session
token with ``client.sessions.create()`` and hand that to the page.

    from klavi_experts import ExpertsClient

    experts = ExpertsClient(base_url="https://experts.acme.com", api_key="sk-...")
    for event in experts.conversations.ask("What is our refund policy?"):
        if event.type == "token":
            print(event.content, end="", flush=True)
"""

from ._chat_stream import AsyncChatStream, ChatStream
from ._stream import LineBuffer, to_event
from ._transport import AsyncTransport, Request, SyncTransport
from .async_client import AsyncExpertsClient
from .client import ExpertsClient
from .errors import (
    AbortError,
    AuthError,
    BadRequestError,
    ExpertsError,
    ExpertsPermissionError,
    LicenseError,
    NotFoundError,
    RateLimitError,
    ServerError,
    StreamError,
)
from .types import (
    ActionEvent,
    BrowserSession,
    ChatEvent,
    ChatResult,
    Collection,
    Conversation,
    ConversationWithMessages,
    DoneEvent,
    Expert,
    KeepaliveEvent,
    KnowledgeDocument,
    Message,
    SourceDocument,
    ThinkingEvent,
    TokenEvent,
    Usage,
)

__version__ = "0.1.0"

# Grouped by what a reader is looking for — clients, then errors, then wire
# types — rather than alphabetised. RUF022 wants the latter; the grouping
# with its comments is the more useful order in a file people skim.
__all__ = [  # noqa: RUF022
    "ExpertsClient",
    "AsyncExpertsClient",
    "ChatStream",
    "AsyncChatStream",
    "LineBuffer",
    "to_event",
    "Request",
    "SyncTransport",
    "AsyncTransport",
    # errors
    "ExpertsError",
    "AuthError",
    "ExpertsPermissionError",
    "LicenseError",
    "NotFoundError",
    "RateLimitError",
    "ServerError",
    "BadRequestError",
    "StreamError",
    "AbortError",
    # types
    "Expert",
    "Conversation",
    "ConversationWithMessages",
    "Message",
    "SourceDocument",
    "Usage",
    "ChatEvent",
    "ChatResult",
    "TokenEvent",
    "ThinkingEvent",
    "ActionEvent",
    "KeepaliveEvent",
    "DoneEvent",
    "Collection",
    "KnowledgeDocument",
    "BrowserSession",
    "__version__",
]
