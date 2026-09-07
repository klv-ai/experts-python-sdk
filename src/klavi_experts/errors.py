"""Typed errors.

The distinctions here are the ones a caller has to act on differently, not a
taxonomy for its own sake. `LicenseError` in particular is not a variety of
"forbidden": it says the *installation's* licence has lapsed, so no amount of
fixing the request will help and the person who can fix it is the install's
administrator, not whoever is reading the traceback.
"""

from __future__ import annotations

from typing import Any


class ExpertsError(Exception):
    """Base for everything this SDK raises."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        body: Any = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        #: The parsed response body, when there was one.
        self.body = body
        #: Server request id, if the install returned one. Quote it in a bug report.
        self.request_id = request_id

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.message!r}, status={self.status})"


class AuthError(ExpertsError):
    """401 — the credential is missing, malformed, expired or revoked."""


class PermissionError_(ExpertsError):
    """403 — authenticated, but not allowed to do this.

    Named with a trailing underscore so it cannot shadow the builtin
    ``PermissionError`` for anyone doing ``from klavi_experts.errors import *``.
    Exported as ``ExpertsPermissionError`` too, which is the name to prefer.
    """


ExpertsPermissionError = PermissionError_


class LicenseError(ExpertsError):
    """403 with ``{"error": "license_expired"}`` — the INSTALL's licence lapsed.

    Separate from a permission error on purpose. Nothing about the request is
    wrong and no retry or credential change will help; the install's
    administrator has to renew. Reporting "forbidden" here sends a developer to
    debug their own code for an hour.
    """


class NotFoundError(ExpertsError):
    """404 — or a resource this credential may not see.

    The API answers "not found" rather than "forbidden" for a conversation the
    caller has no access to, deliberately: a uid probe must not reveal whether
    something exists. So this can mean either, and the SDK does not guess.
    """


class RateLimitError(ExpertsError):
    """429 — slow down. ``retry_after`` is in seconds when the server said."""

    def __init__(
        self, message: str, *, retry_after: float | None = None, **kw: Any
    ) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class ServerError(ExpertsError):
    """5xx, or a transport failure. Retried automatically before it surfaces."""


class BadRequestError(ExpertsError):
    """4xx that is none of the above — a malformed request."""


class StreamError(ExpertsError):
    """The stream ended without a terminal event.

    Usually a dropped connection. Note that generation is DETACHED from the
    HTTP request on this platform: the answer very likely completed on the
    server and was persisted, so re-reading the conversation is the right
    recovery, not resending the question.
    """


class AbortError(ExpertsError):
    """The caller stopped the stream."""


def _message_from(body: Any, status: int) -> tuple[str, str | None]:
    """Pull a human message and a code out of whichever envelope arrived.

    The API speaks three of them depending on which service answered, and the
    difference is not cosmetic: reading the wrong one turns "your licence has
    expired" into "Request failed with status 403".
    """
    if isinstance(body, str) and body.strip():
        return body.strip()[:500], None

    if isinstance(body, dict):
        error = body.get("error")

        # OpenAI-compatible surface: {"error": {"message", "type", "code"}}
        if isinstance(error, dict):
            code = error.get("type")
            return (
                str(error.get("message", "Request failed")),
                code if isinstance(code, str) else None,
            )

        # Keyguard and the session middleware: {"error": "...", "detail": "..."}
        if isinstance(error, str):
            detail = body.get("message") or body.get("detail")
            return str(detail or error), error

        # FastAPI's default: {"detail": ...}
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail, None
        if isinstance(detail, list):
            import json

            return json.dumps(detail)[:500], None

    return f"Request failed with status {status}", None


def error_from_response(
    status: int,
    body: Any,
    *,
    request_id: str | None = None,
    retry_after: float | None = None,
) -> ExpertsError:
    """Build the right exception for a failed response."""
    message, code = _message_from(body, status)
    common: dict[str, Any] = {
        "status": status,
        "code": code,
        "body": body,
        "request_id": request_id,
    }

    if status == 401:
        return AuthError(message, **common)
    if status == 403:
        # An install-level failure wearing a request-level status code.
        if code == "license_expired":
            return LicenseError(
                f"{message} (the installation's licence has expired — this is not "
                f"something your request or credentials can fix)",
                **common,
            )
        return PermissionError_(message, **common)
    if status == 404:
        return NotFoundError(message, **common)
    if status == 429:
        return RateLimitError(message, retry_after=retry_after, **common)
    if status >= 500:
        return ServerError(message, **common)
    return BadRequestError(message, **common)
