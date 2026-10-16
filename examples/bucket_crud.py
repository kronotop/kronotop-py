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

"""The full lifecycle of a document, on the asyncio client.

Insert, query, update and delete on one bucket. ``insert`` returns the ids.
The other three open a cursor. A query batch holds documents. An update or
delete batch holds the ids of the documents it changed.

Run with ``uv run examples/bucket_crud.py``.
"""

import asyncio

from kronotop.asyncio import Kronotop

BUCKET = "examples-crud"


async def main() -> None:
    async with Kronotop() as client, client.session() as session:
        await session.bucket.create(BUCKET, if_not_exists=True)

        # Create. One ObjectId per document, in the order given.
        ids = await session.bucket.insert(
            BUCKET,
            {"name": "Ada", "status": "pending"},
            {"name": "Linus", "status": "pending"},
            {"name": "Grace", "status": "active"},
        )
        print("inserted:", ids)

        # Read. The query is a mapping, and the first batch is in
        # ``cursor.batch``. Three documents fit in one batch here.
        async with session.bucket.query(BUCKET, {"status": "pending"}) as cursor:
            for document in cursor.batch:
                print("read    :", document["name"], document["status"])

        # Update. The update mapping carries the operators. The batch holds
        # the id of each modified document: Grace is already active, so only
        # Ada and Linus are in it.
        async with session.bucket.update(
            BUCKET, {}, {"$set": {"status": "active"}}
        ) as cursor:
            print("updated :", cursor.batch)

        # Delete. The batch holds the id of each deleted document.
        async with session.bucket.delete(BUCKET, {"name": "Linus"}) as cursor:
            print("deleted :", cursor.batch)

        async with session.bucket.query(BUCKET, {}) as cursor:
            print("left    :", [document["name"] for document in cursor.batch])

        # ``remove`` marks the bucket, ``purge`` deletes its data.
        await session.bucket.remove(BUCKET)
        await session.bucket.purge(BUCKET)


if __name__ == "__main__":
    asyncio.run(main())
