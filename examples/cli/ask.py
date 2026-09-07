"""Ask an expert a question from the terminal.

    EXPERTS_BASE_URL=... EXPERTS_API_KEY=... python ask.py "what is our refund policy?"
"""

from __future__ import annotations

import os
import signal
import sys

from klavi_experts import ExpertsClient, LicenseError, RateLimitError


def main() -> int:
    question = " ".join(sys.argv[1:])
    if not question:
        print("usage: python ask.py <question>", file=sys.stderr)
        return 1

    experts = ExpertsClient(
        base_url=os.environ.get("EXPERTS_BASE_URL", "http://localhost:5003"),
        api_key=os.environ["EXPERTS_API_KEY"],
    )

    try:
        stream = experts.conversations.ask(
            question, expert=os.environ.get("EXPERTS_EXPERT_UID")
        )

        # Ctrl-C should STOP the model, not just close the pipe. Generation is
        # detached server-side: walking away leaves it running and billing.
        def stop(*_):
            stream.cancel()
            raise SystemExit(130)

        signal.signal(signal.SIGINT, stop)

        for event in stream:
            if event.type == "token":
                print(event.content, end="", flush=True)
        # Flush before the summary goes to stderr, or the two interleave
        # whenever stdout is piped and therefore block-buffered.
        print(flush=True)

        result = stream.result()
        if result.error:
            print(f"\nGeneration failed: {result.error}", file=sys.stderr)
        for doc in result.source_docs:
            score = f"{doc.similarity:.2f}" if doc.similarity is not None else "?"
            print(f"  {doc.title or doc.name}  ({score})", file=sys.stderr)
        print(
            f"\n{result.usage.total_tokens} tokens, {result.usage.total_ms or '?'}ms",
            file=sys.stderr,
        )
    except RateLimitError as e:
        wait = e.retry_after or "a moment"
        print(f"Rate limited. Retry after {wait}s.", file=sys.stderr)
        return 1
    except LicenseError:
        print(
            "This install's licence has expired. Contact its administrator.",
            file=sys.stderr,
        )
        return 1
    finally:
        experts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
