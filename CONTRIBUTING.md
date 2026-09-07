# Contributing

Thanks for helping. This SDK wraps a HTTP API that it does not control, so the
most valuable contributions are usually about *behaviour we got wrong* rather
than features we are missing.

## Getting set up

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest            # unit tests, no install required
ruff check src tests
mypy src
```

## Running against a real install

The unit tests use a stubbed transport, which cannot catch drift between this
SDK and the Python services it talks to. The live suite can:

```bash
cp .env.example .env      # fill in EXPERTS_BASE_URL and EXPERTS_TEST_KEY
pytest tests/live
```

It creates only hidden conversations and deletes them afterwards. Point it at
a development install, never production.

## Sync and async

There are two clients, and they must not drift. Everything that is not I/O —
request building, response shaping, NDJSON framing, event mapping — is shared,
so a change to what a call MEANS is made once. `tests/test_async_parity.py`
compares their signatures and the requests they emit; if you add a parameter
to one half, that suite tells you about the other.

## What we look for in a change

- **A test that fails without it.** For a bug, that test should describe the
  behaviour, not the fix.
- **Comments that explain why, not what.** Most of the surprising code here is
  surprising because the API is, and the comment should say which part.
- **No new runtime dependencies.** `httpx` is the only one, on purpose.

## Reporting an API problem

If the API itself misbehaves — a shape that does not match its documentation, a
silent failure — open an issue with the raw request and response. Those are
often fixed on the server rather than papered over here, and knowing which is
the point of the report.

## Security

Do not open an issue for a vulnerability. See [SECURITY.md](SECURITY.md).
