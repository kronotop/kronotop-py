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

"""Read documents with ``BUCKET.QUERY``, on the asyncio client.

The query is a mapping. Results arrive in batches through a cursor, and
``batches()`` sends ``BUCKET.ADVANCE`` for each batch after the first.

``batches()`` stops after the batch that carries cursor id ``-1``. The server
has dropped the cursor by then, so the ``with`` block has nothing to close.
A cursor you stop reading early stays open, and leaving the ``with`` block
sends ``BUCKET.CLOSE``.

Run with ``uv run examples/bucket_query.py``.
"""

import asyncio

from kronotop.asyncio import Kronotop

BUCKET = "examples-query"


async def main() -> None:
    async with Kronotop() as client, client.session() as session:
        await session.bucket.create(BUCKET, if_not_exists=True)
        await session.bucket.insert(
            BUCKET,
            {"name": "Ada", "age": 36},
            {"name": "Linus", "age": 55},
            {"name": "Grace", "age": 45},
        )

        # The command goes out here, and the first batch is in
        # ``cursor.batch``. ``batch`` is the size of one batch, ``limit``
        # would cap the whole result.
        async with session.bucket.query(
            BUCKET,
            {"age": {"$gt": 30}},
            projection={"_id": 0, "name": 1, "age": 1},
            batch=2,
        ) as cursor:
            print("cursor  :", cursor.cursor_id)
            # The first turn is the batch already in hand.
            async for batch in cursor.batches():
                print("batch   :", batch)

            # The last reply carried cursor id -1.
            print("closed  :", cursor.closed, cursor.cursor_id)

        # The server dropped the cursor, so the session lists none.
        print("cursors :", await session.bucket.cursors("query"))

        # Stop early. The first batch is full, so the cursor stays open on the
        # server, and leaving the block sends BUCKET.CLOSE.
        async with session.bucket.query(BUCKET, {}, batch=1) as cursor:
            print("first   :", cursor.batch)
            print("open    :", await session.bucket.cursors("query"))
        print("cursors :", await session.bucket.cursors("query"))

        # ``remove`` marks the bucket, ``purge`` deletes its data.
        await session.bucket.remove(BUCKET)
        await session.bucket.purge(BUCKET)


if __name__ == "__main__":
    asyncio.run(main())
