"""Feed a corpus into an expert's knowledge, and wait for it to be searchable.

Uploading is ASYNCHRONOUS. The file is accepted, then split, embedded and
indexed by a worker — so a document that has just uploaded is not retrievable
yet, and querying immediately returns nothing with no error to explain it.

    python ingest.py ./policies/*.pdf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from klavi_experts import ExpertsClient


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: python ingest.py <file>...", file=sys.stderr)
        return 1

    experts = ExpertsClient(
        base_url=os.environ["EXPERTS_BASE_URL"],
        api_key=os.environ["EXPERTS_API_KEY"],
    )

    collection = os.environ.get("EXPERTS_COLLECTION_UID")
    if not collection:
        collections = experts.knowledge.list_collections()
        if not collections:
            print("No collections on this install; make one first.", file=sys.stderr)
            return 1
        collection = collections[0].uid
        print(f"Using collection {collections[0].name} ({collection})")

    pending = []
    for path in paths:
        with path.open("rb") as handle:
            doc = experts.knowledge.upload(
                handle, collection=collection, filename=path.name
            )
        print(f"uploaded {path.name} -> {doc.uid}")
        pending.append(doc)

    # Accepted is not the same as searchable.
    for doc in pending:
        ready = experts.knowledge.wait_for_processing(doc.uid, timeout=300)
        print(f"{'indexed ' if ready else 'TIMEOUT '} {doc.name}")

    experts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
