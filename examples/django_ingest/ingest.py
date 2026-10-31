"""Feed a corpus into an expert's knowledge, and wait for it to be searchable.

Uploading is ASYNCHRONOUS. The file is accepted into an indexing job, then
split, embedded and indexed by a worker — so a document that has just uploaded
is not retrievable yet, and querying immediately returns nothing with no error
to explain it. The job is also where a file that could not be read says why.

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

    jobs = []
    for path in paths:
        with path.open("rb") as handle:
            job = experts.knowledge.upload(
                handle, collection=collection, filename=path.name
            )
        print(f"uploaded {path.name} (job {job.job_id})")
        jobs.append(job)

    # Accepted is not the same as searchable.
    failures = 0
    for job in jobs:
        status = experts.knowledge.wait_for_processing(job, timeout=300)
        for file in status.files:
            if file.searchable:
                print(f"indexed  {file.name} -> {file.document_uid}")
            elif file.status == "failed":
                failures += 1
                print(f"FAILED   {file.name}: {file.error}")
            else:
                failures += 1
                print(f"TIMEOUT  {file.name} (still {file.status})")

    experts.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
