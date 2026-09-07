"""Mint browser session tokens from FastAPI.

The only place the API key appears. Everything the visitor's browser can do is
decided here, at mint time: one expert, one conversation, one origin, fifteen
minutes. The page then talks to the install directly — you are not proxying
every token through your own process.

    uvicorn app:app --reload

Before it works:

  1. Register the origin against your API key in the install's admin. That
     registration is also what supplies CORS for the surface.
  2. Make the expert guest-visible — minimum role Guest, not private. A session
     token IS a guest, and an expert above that floor resolves to nothing
     server-side: the chat would answer on the site default model with no
     persona and no knowledge, silently, with a 200. sessions.create() refuses
     such an expert and says so.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from klavi_experts import AsyncExpertsClient, ExpertsError

app = FastAPI()

experts = AsyncExpertsClient(
    base_url=os.environ["EXPERTS_BASE_URL"],
    api_key=os.environ["EXPERTS_API_KEY"],
)

EXPERT_UID = os.environ["EXPERTS_EXPERT_UID"]
SITE_ORIGIN = os.environ["SITE_ORIGIN"]


class SessionRequest(BaseModel):
    # Passing the previous conversation back keeps a visitor's transcript
    # across a token refresh or a page reload.
    conversation: str | None = None
    page: str | None = None


@app.post("/api/chat/session")
async def create_session(body: SessionRequest) -> dict:
    # Rate limit by IP here if the page is genuinely public — minting is cheap,
    # but every token is a licence to spend inference.
    try:
        session = await experts.sessions.create(
            expert=EXPERT_UID,
            origin=SITE_ORIGIN,
            conversation=body.conversation,
            metadata={"page": body.page},
        )
    except ExpertsError as exc:
        # Do not leak the install's message to a public page.
        raise HTTPException(503, "Chat is unavailable right now") from exc

    # Deliberately NOT returning session_id: it is the revocation handle, and
    # the browser has no use for it.
    return {
        "token": session.token,
        "conversation": session.conversation,
        "expires_in": session.expires_in,
    }


@app.on_event("shutdown")
async def _close() -> None:
    await experts.aclose()
