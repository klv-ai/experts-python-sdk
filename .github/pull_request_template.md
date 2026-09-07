## What this changes

<!-- And why. If it fixes a bug, say what the wrong behaviour was. -->

## Checklist

- [ ] A test that fails without this change
- [ ] `ruff check`, `mypy src` and `pytest` pass
- [ ] No new runtime dependencies (httpx is the only one)
- [ ] Sync and async halves still agree (`tests/test_async_parity.py`)
- [ ] Comments explain *why*, where the reason is not obvious from the code
