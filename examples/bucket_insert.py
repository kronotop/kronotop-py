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

"""Store documents in a bucket with ``BUCKET.INSERT``, on the asyncio client.

Each document is a mapping. The client encodes it in the format the session
attribute ``input_type`` names. The reply is one ``bson.ObjectId`` per document,
in the order the documents were given.

Run with ``uv run examples/bucket_insert.py``.
"""

import asyncio

from kronotop.asyncio import Kronotop

BUCKET = "examples-users"


async def main() -> None:
    # ``namespace`` picks the namespace the bucket is created in and read from.
    # ``NAMESPACE.USE`` does the same on an open session.
    async with (
        Kronotop() as client,
        client.session(namespace="global") as session,
    ):
        await session.bucket.create(BUCKET, if_not_exists=True)
        ids = await session.bucket.insert(
            BUCKET,
            {"name": "Ada", "age": 36},
            {"name": "Linus", "age": 55},
        )
        for document_id in ids:
            print("inserted:", document_id)
        print("count   :", len(ids))


if __name__ == "__main__":
    asyncio.run(main())
