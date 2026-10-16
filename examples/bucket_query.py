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

An empty batch does not mean the cursor is done, so ``batches()`` never ends on
its own. This example stops on the first empty batch, which is enough for a
small bucket.

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
                if not batch:
                    break
                print("batch   :", batch)

            # The cursor is still open here, so the session lists it.
            print("cursors :", await session.bucket.cursors("query"))

        # Leaving the block sent BUCKET.CLOSE.
        print("cursors :", await session.bucket.cursors("query"))


if __name__ == "__main__":
    asyncio.run(main())
