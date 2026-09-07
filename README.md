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
            case "action":   ...   # "rag_search", "tool_call" — for a status line
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
```

### Stopping

```python
stream.cancel()   # stops the model
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

**The expert must be guest-visible.** A session token carries the guest role, and an expert above that floor resolves to nothing server-side — the chat then answers on the site's default model with no persona and no knowledge, silently, with a normal 200. `sessions.create()` refuses such an expert up front and tells you how to fix it. `experts.experts.list_guest_visible()` gives you the ones that will work.

## Knowledge

```python
collections = experts.knowledge.list_collections()

with open("policy.pdf", "rb") as f:
    doc = experts.knowledge.upload(f, collection=collections[0].uid, filename="policy.pdf")

# Uploading is asynchronous: the file is accepted, then split, embedded and
# indexed by a worker. It is NOT searchable until that finishes.
experts.knowledge.wait_for_processing(doc.uid)

hits = experts.knowledge.search("refund policy")
```

## Errors

```python
from klavi_experts import RateLimitError, LicenseError

try:
    experts.conversations.ask("hi").text()
except RateLimitError as e:
    time.sleep(e.retry_after or 5)
except LicenseError:
    # The INSTALL's licence has lapsed. Nothing about your request is wrong and
    # no retry will help — its administrator has to renew.
    ...
```

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

Python 3.10 or later, and an install running build pack **1.0.49** or later.

## Contributing

Issues and pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go to [SECURITY.md](SECURITY.md), not the issue tracker.

## Licence

MIT — see [LICENSE](LICENSE).
