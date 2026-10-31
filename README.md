# klavi-experts

The official Python SDK for the [Klavi Experts](https://github.com/klv-ai) API — AI experts with retrieval over your own documents, running on your own infrastructure.

Sync and async. One dependency (`httpx`). Python 3.10+.

```bash
pip install klavi-experts
```

## Thirty seconds

```python
from klavi_experts import ExpertsClient

experts = ExpertsClient(
    base_url="https://experts.acme.com",
    api_key=os.environ["EXPERTS_API_KEY"],   # server-side only
)

for event in experts.conversations.ask("What's our refund policy?"):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

Or, when you just want the answer:

```python
answer = experts.conversations.ask("Summarise Q3.").text()
```

Async is the same API:

```python
from klavi_experts import AsyncExpertsClient

async with AsyncExpertsClient(base_url=..., api_key=...) as experts:
    stream = await experts.conversations.ask("Summarise Q3.")
    async for event in stream:
        ...
```

## Why not just call the API

Sending one message is three HTTP calls in a specific order, and getting it wrong fails *silently* — the model answers, the stream looks fine, and nothing is saved:

```
POST /api/v1/responses           the user's turn        {agent: False, done: True}
POST /api/v1/responses           assistant placeholder  {agent: True, done: False}  ← keep uid
POST /api/v1/conversations/chat  {response: <placeholder uid>, ...}
```

`conversations.send()` is that sequence. The rest of the SDK is the same idea applied to the parts of the API that are easy to get subtly wrong — see [Things worth knowing](#things-worth-knowing).

## Talking to an expert

```python
experts_list = experts.experts.list()

conversation = experts.conversations.create(
    expert=experts_list[0].uid,
    title="Support",
)

with experts.conversations.send(conversation.uid, "Hello") as stream:
    for event in stream:
        match event.type:
            case "token":    print(event.content, end="", flush=True)
            case "thinking": ...   # reasoning tokens, when the model exposes them
            case "queued":   ...   # waiting in line for another mesh server's model
            case "done":     print(event.usage, event.source_docs)
```

Everything the answer was grounded in comes back on the terminal event:

```python
result = stream.result()
result.text            # the full reply
result.source_docs     # documents retrieved, with similarity scores
result.tools_used      # e.g. ["web_search"]
result.usage.total_ms  # milliseconds (the API speaks nanoseconds)
result.error           # set when generation failed inside a 200 — always check
result.model           # the model that answered
result.data_artifacts  # tables, charts and sheet edits a tool produced
```

On an [Experts Mesh](https://klv.ai/mesh), an answer can come from another server's model. `result.served_by` names that server; `result.mesh_fallback` says when the expert's own model was out of reach and this server's answered instead. While a busy server's model has the request in line, the stream sends `queued` events (`event.ahead`, `event.name`) and nothing else. Raise `stream_timeout` on the client if yours wait longer than five minutes.

### Stopping

```python
stream.cancel()   # stops the model; True if the server found it running
stream.detach()   # stop reading; let it finish and persist
```

These are genuinely different. Generation is **detached** from the HTTP request server-side, so simply walking away stops the relay, not the model — it keeps generating, keeps costing you, and still writes its answer. `cancel()` is the only thing that stops it.

## A chatbox on your website

An API key is a full user identity on the install, so it must never reach a browser. Instead your Python server mints a short-lived session token bound to one expert, one conversation and one origin.

**Register the origin first** against your API key, in the install's admin. That registration is also what supplies CORS for this surface.

```python
# Django / Flask / FastAPI — your /api/chat/session route
session = experts.sessions.create(expert=EXPERT_UID, origin="https://acme.com")
return {"token": session.token}
```

The browser then talks to the install directly, using [`@klv-ai/experts/browser`](https://www.npmjs.com/package/@klv-ai/experts) or plain `fetch`. Your server is involved once, to mint — you are not proxying every token.

**The expert must be guest-visible**: minimum role Guest, not private, and not limited to org profiles. A session token carries the guest role, so `sessions.create()` refuses any other expert up front and tells you how to fix it. `experts.experts.list_guest_visible()` gives you the ones that will work.

**On an Experts Mesh, the browser talks to the server you minted on.** A session lives on the server that issued it; another member of the mesh does not know it and answers 401.

## Knowledge

```python
collections = experts.knowledge.list_collections()

with open("policy.pdf", "rb") as f:
    job = experts.knowledge.upload(f, collection=collections[0].uid, filename="policy.pdf")

# Uploading is asynchronous: the file is accepted into an indexing job, then
# split, embedded and indexed by a worker. It is NOT searchable until that
# finishes — and the job is where the document's uid and any failure appear.
status = experts.knowledge.wait_for_processing(job)
if not status:
    print(status.failed)       # e.g. a password-protected PDF, and why

for hit in experts.knowledge.search("refund policy", collections=[collections[0].uid]):
    print(hit.document_name, hit.similarity, hit.text)
```

`search()` returns passages, closest first, each with the document it came from. `list_documents(collection=..., search=...)` lists and filters documents by name, title or summary; `delete_document(uid)` removes one. Allow longer than the two-minute default for large files, and much longer for recordings, which are transcribed before they become documents.

## Errors

```python
from klavi_experts import ExpertNotAvailableError, LicenseError, RateLimitError

try:
    experts.conversations.ask("hi", expert=EXPERT_UID).text()
except RateLimitError as e:
    time.sleep(e.retry_after or 5)
except ExpertNotAvailableError:
    # This key's user may not use that expert (its role, privacy or org
    # profiles). Pick another from experts.experts.list().
    ...
except LicenseError as e:
    # The INSTALL's licence is not valid. Nothing about your request is wrong
    # and no retry will help — its administrator has to act. e.read_only is
    # True for a recently expired licence: reads still work, writes do not.
    ...
```

For 30 days after its licence expires, an install is read-only: anything sent as a GET still works, and everything else raises `LicenseReadOnlyError` (a `LicenseError`). That includes sending a message and `knowledge.search()`, which read but go out as a POST.

429 and 5xx are retried automatically with backoff that honours `Retry-After`. A 4xx never is. Neither is a chat turn — retrying one bills twice and can produce two answers.

## Things worth knowing

These are the parts of the API that surprise people. The SDK handles each of them; they are listed so you know what it is doing on your behalf.

| | |
|---|---|
| **Send both credentials and the wrong one wins** | The gateway checks `Authorization` first and that check is terminal. The SDK sends exactly one. |
| **A terminal chunk can be empty on purpose** | On the tool path the text already streamed. Appending terminal content renders tool-using answers twice. |
| **Durations are nanoseconds** | Ollama's units, passed straight through. Normalised to `*_ms` fields here. |
| **HTTP 200 can still be a failure** | Generation errors arrive in the terminal event, not the status code. Check `result.error`. |
| **404 can mean "forbidden"** | A conversation you may not see answers 404 so a uid probe reveals nothing. The SDK does not guess which it was. |
| **Casing is mixed by design** | Conversation rows are camelCase; message and expert rows are snake_case; knowledge responses are wrapped in an envelope. All normalised. |
| **Streaming is plain HTTP** | NDJSON over `POST`. No WebSocket is required for chat, contrary to some older notes. |
| **A refused turn leaves a placeholder behind** | `/chat` updates a message row you created first. When it refuses to start, nothing settles that row, and the conversation shows the reply "typing" forever. The SDK settles it with the reason. |
| **An expert's instructions are for its managers** | Since 1.1, `Expert.instructions` is only filled in for the expert's owner and administrators. |
| **Knowledge says "forbidden" as 401** | The knowledge service answers a document you may not read with 401. The SDK raises `ExpertsPermissionError`, not `AuthError`: the key is fine. |

## OpenAI-compatible endpoint

If you already have code written against OpenAI, you may not need this SDK at all. An install also speaks the OpenAI chat-completions protocol, with an expert's uid as the model:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EXPERTS_API_KEY"],
    base_url="https://experts.acme.com/api/openai/v1",
)

client.chat.completions.create(
    model=EXPERT_UID,
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
```

Retrieval still happens; source documents come back under a `klavi` key on the final chunk. Use this SDK when you want conversations, knowledge management or browser sessions; use the OpenAI client when you want to drop an install into tooling that already speaks that protocol.

## Requirements

Python 3.10 or later, and an install running build pack **1.0.49** or later. Tested against **1.1**; the mesh fields and the read-only licence error are only sent by a 1.1 install.

## Contributing

Issues and pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go to [SECURITY.md](SECURITY.md), not the issue tracker.

## Licence

MIT — see [LICENSE](LICENSE).
