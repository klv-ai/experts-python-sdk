# Security

## Reporting a vulnerability

Please report security issues privately to **security@klavi.ai**, or through
GitHub's private vulnerability reporting on this repository. Do not open a
public issue.

Include what you can: what you found, how to reproduce it, and what an attacker
could do with it. We will acknowledge within three working days and keep you
updated as we work on a fix.

## Scope

This repository is the Python client SDK. Issues in an Experts *installation* — the
gateway, the conversations service — should be reported the same way; say which
you mean and we will route it.

## Handling credentials

Two rules matter more than anything else here:

- **An API key (`sk-…`) is a full user identity on an install.** It belongs on
  a server. It must never be shipped in front-end code, committed, or put in a
  URL — credentials in URLs end up in access logs, `Referer` headers and
  browser history.
- **For anything browser-facing, mint a session token.** `sessions.create()`
  returns a short-lived credential bound to one expert, one conversation and
  one origin. The Python SDK never handles a browser credential itself — it mints
  one and hands it to your page.

If a key is exposed, rotate it in the install's admin. Rotation invalidates the
old key immediately.
