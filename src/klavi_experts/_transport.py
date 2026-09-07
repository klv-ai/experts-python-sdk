"""HTTP, sync and async.

Three details are not obvious, and each was learned against the real API:

1. **Never send two credentials.** The gateway checks ``Authorization`` first
   and that branch is terminal: a stale or malformed bearer token 401s and
   never falls through to ``X-API-Key``. Sending both means the wrong one
   silently decides the outcome, so exactly one goes out.

2. **Check the status before reading the body.** An error body parses as
   perfectly valid JSON — and, on the streaming endpoints, as one valid NDJSON
   line. A reader that starts consuming before checking reports "the model
   returned nothing" for what was a 404.

3. **Retry only what is safe to retry.** 429 and 5xx, with backoff that
   honours ``Retry-After``. Never a 4xx (the request is wrong and will stay
   wrong), and never a generation call — retrying a chat turn bills twice and
   can produce two answers.

The sync and async classes share the pure parts below and differ only in how
they move bytes. That is the whole reason `_prepare` and `_outcome` exist as
free functions rather than methods.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .errors import ExpertsError, ServerError, error_from_response

RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
#: Streaming has no meaningful overall deadline — a long answer is not a hung
#: request — but a stalled connection still has to end. httpx's read timeout
#: applies between chunks, which is the right knob.
STREAM_READ_TIMEOUT = 300.0


@dataclass(slots=True)
class Request:
    method: str = "GET"
    path: str = "/"
    params: dict[str, Any] | None = None
    json: Any = None
    files: Any = None
    data: Any = None
    headers: dict[str, str] | None = None
    #: Streams opt out of the overall timeout and of retries.
    stream: bool = False
    #: Force-disable retries for a call that must not be repeated.
    idempotent: bool = True


def _auth_header(api_key: str | None, token: str | None) -> dict[str, str]:
    """Exactly one credential, as a Bearer token.

    ``Authorization: Bearer sk-...`` works as of build pack 1.0.49 and is the
    form every off-the-shelf client uses. ``X-API-Key`` is the older header and
    still works, but sending both would let a stale Authorization decide the
    request, so it is never used alongside one.
    """
    secret = token or api_key
    return {"authorization": f"Bearer {secret}"} if secret else {}


def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop None rather than sending the literal string 'None'."""
    return {k: v for k, v in (params or {}).items() if v is not None}


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _backoff(attempt: int, retry_after: float | None) -> float:
    # The server knows when its window resets; prefer its answer to guessing.
    if retry_after is not None:
        return min(retry_after, 60.0)
    # Jitter, so a fleet of retries does not synchronise.
    return min(2.0 ** (attempt - 1) * 0.5, 8.0) + random.random() * 0.25


def _body(response: httpx.Response) -> Any:
    text = response.text
    if not text:
        return None
    try:
        return response.json()
    except ValueError:
        return text


def _outcome(response: httpx.Response) -> ExpertsError | None:
    """The error this response represents, or None if it succeeded."""
    if response.is_success:
        return None
    return error_from_response(
        response.status_code,
        _body(response),
        request_id=response.headers.get("x-request-id"),
        retry_after=_retry_after(response),
    )


class _Base:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key and not token:
            raise ValueError(
                "Provide either api_key (server) or token (a session token)"
            )
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._token = token
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._extra_headers = dict(headers or {})

    def _headers(self, request: Request) -> dict[str, str]:
        headers = {"accept": "application/json"}
        headers.update(_auth_header(self._api_key, self._token))
        headers.update(self._extra_headers)
        headers.update(request.headers or {})
        return headers

    def _attempts(self, request: Request) -> int:
        if request.stream or not request.idempotent:
            return 1
        return self._max_retries

    def _build(self, request: Request) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "method": request.method,
            "url": self.base_url + (
                request.path if request.path.startswith("/") else f"/{request.path}"
            ),
            "headers": self._headers(request),
            "params": _clean_params(request.params),
        }
        if request.json is not None:
            kwargs["json"] = request.json
        if request.files is not None:
            # Multipart: httpx sets the content-type and boundary itself.
            kwargs["files"] = request.files
        if request.data is not None:
            kwargs["data"] = request.data
        return kwargs


class SyncTransport(_Base):
    def __init__(self, *, client: httpx.Client | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._client = client or httpx.Client(timeout=self._timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def request(self, request: Request) -> Any:
        response = self.send(request)
        try:
            if response.status_code == 204:
                return None
            return _body(response)
        finally:
            response.close()

    def send(self, request: Request) -> httpx.Response:
        """The raw response, status already checked. The caller owns the body."""
        attempts = self._attempts(request)
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                if request.stream:
                    # Stream mode: httpx wants an explicit request object so the
                    # body is not read into memory before we see the status.
                    req = self._client.build_request(
                        **self._build(request),
                        timeout=httpx.Timeout(None, read=STREAM_READ_TIMEOUT),
                    )
                    response = self._client.send(req, stream=True)
                else:
                    response = self._client.request(**self._build(request))
            except httpx.HTTPError as exc:
                last = exc
                if attempt < attempts:
                    time.sleep(_backoff(attempt, None))
                    continue
                raise ServerError(str(exc) or "Network request failed") from exc

            error = _outcome(response)
            if error is None:
                return response

            # Read and release before deciding, or a streamed error response
            # holds the connection open for the whole retry window.
            response.read()
            response.close()
            if attempt < attempts and response.status_code in RETRYABLE:
                last = error
                time.sleep(_backoff(attempt, _retry_after(response)))
                continue
            raise error

        raise last if isinstance(last, Exception) else ServerError(
            "Request failed after retries"
        )


class AsyncTransport(_Base):
    def __init__(self, *, client: httpx.AsyncClient | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._client = client or httpx.AsyncClient(timeout=self._timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def request(self, request: Request) -> Any:
        response = await self.send(request)
        try:
            if response.status_code == 204:
                return None
            return _body(response)
        finally:
            await response.aclose()

    async def send(self, request: Request) -> httpx.Response:
        attempts = self._attempts(request)
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                if request.stream:
                    req = self._client.build_request(
                        **self._build(request),
                        timeout=httpx.Timeout(None, read=STREAM_READ_TIMEOUT),
                    )
                    response = await self._client.send(req, stream=True)
                else:
                    response = await self._client.request(**self._build(request))
            except httpx.HTTPError as exc:
                last = exc
                if attempt < attempts:
                    await asyncio.sleep(_backoff(attempt, None))
                    continue
                raise ServerError(str(exc) or "Network request failed") from exc

            error = _outcome(response)
            if error is None:
                return response

            await response.aread()
            await response.aclose()
            if attempt < attempts and response.status_code in RETRYABLE:
                last = error
                await asyncio.sleep(_backoff(attempt, _retry_after(response)))
                continue
            raise error

        raise last if isinstance(last, Exception) else ServerError(
            "Request failed after retries"
        )
