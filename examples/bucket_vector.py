# Copyright (c) 2026 Burak Sezer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Similarity search with ``BUCKET.VECTOR``.

The index is ready with the bucket, but the graph it searches is fed after
the commit. So the search is retried while the server says the index is not
ready or answers nothing.

Run with ``uv run examples/bucket_vector.py``.
"""

import time

from kronotop import Kronotop, VectorIndex, VectorIndexNotReadyError

BUCKET = "examples-vector"


def main() -> None:
    with Kronotop() as client, client.session() as session:
        # ``if_not_exists`` answers True either way, so the list says whether
        # the bucket, and with it the index, is already there.
        if BUCKET not in session.bucket.list():
            # A vector index needs one shard, and ``create`` without ``shards``
            # picks one.
            session.bucket.create(
                BUCKET, indexes=[VectorIndex("embedding", 3, "cosine")]
            )
            session.bucket.insert(
                BUCKET,
                {"label": "alpha", "embedding": [0.1, 0.2, 0.3]},
                {"label": "beta", "embedding": [0.4, 0.5, 0.6]},
                {"label": "gamma", "embedding": [0.7, 0.8, 0.9]},
            )

        # A ready index that is not fed yet answers an empty list, so that is
        # retried too.
        deadline = time.monotonic() + 10
        while True:
            try:
                results = session.bucket.vector(
                    BUCKET,
                    "embedding",
                    [0.4, 0.5, 0.6],
                    top=2,
                    projection={"_id": 0, "label": 1},
                )
            except VectorIndexNotReadyError:
                results = []
            if results:
                break
            if time.monotonic() > deadline:
                raise SystemExit("the vector index did not get ready in 10 seconds")
            time.sleep(0.1)

        # Best match first. Beta is the query vector itself, so it scores 1.0.
        for score, entry in results:
            print(f"{score:.4f}  {entry}")

        # ``filter_`` is applied after the ranking, so the score is unchanged.
        results = session.bucket.vector(
            BUCKET,
            "embedding",
            [0.4, 0.5, 0.6],
            filter_={"label": "gamma"},
            projection={"_id": 0, "label": 1},
        )
        for score, entry in results:
            print(f"{score:.4f}  {entry}  (filtered)")


if __name__ == "__main__":
    main()
