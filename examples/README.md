# Examples

Each is self-contained and runnable against a development install. All read
`EXPERTS_BASE_URL` and `EXPERTS_API_KEY` from the environment; none commits a key.

| | What it shows |
|---|---|
| [`cli`](./cli) | One-shot questions from a terminal, with sources and usage |
| [`fastapi_chatbox`](./fastapi_chatbox) | Minting browser session tokens so a public page can chat directly |
| [`django_ingest`](./django_ingest) | Feeding a corpus in and waiting for it to become searchable |
