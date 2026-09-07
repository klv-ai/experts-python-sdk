# spec/

`openapi.json` is the API contract this SDK is written against. It is
**generated**, not hand-written — see `knowledge/tools/build_openapi.py` in the
platform repository, which reads the pydantic models of the services behind the
gateway and rewrites them into the paths a customer actually calls.

It is vendored here so the SDK can be typechecked, reviewed and released
without a running install.

## Keeping it honest

The SDK's types are hand-written on top of this rather than generated from it,
deliberately: generated names read like a database and the ergonomic layer is
most of this package's value. What that costs is the risk of drift, which two
things guard against:

- `pytest tests/live` asserts real response *shapes* against a real install;
- the platform repo's `build_openapi.py --check` fails CI when the committed
  spec disagrees with the services.

If you change this file by hand, you have papered over one of those.
