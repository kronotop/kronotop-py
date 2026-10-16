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

"""Open a session on the member that holds the buckets, with ``session_for``.

A bucket command that reaches another member comes back as ``RejectError``.
``session_for`` picks a member that holds all the named buckets, so they share
one transaction. When no member holds them all, it raises
``NoCommonMemberError``.

Run with ``uv run examples/session_for.py``.
"""

from kronotop import Kronotop

NAMESPACE = "global"
FIRST = "examples-users"
SECOND = "examples-orders"


def main() -> None:
    with Kronotop() as client:
        with client.session(namespace=NAMESPACE) as session:
            session.bucket.create(FIRST, if_not_exists=True)
            # Name the first bucket's shard, so one member holds both.
            shard_id = session.bucket.locate(FIRST).shards[0].shard_id
            session.bucket.create(SECOND, shards=[shard_id], if_not_exists=True)

        # This sends commands: one BUCKET.LOCATE per bucket.
        with client.session_for(buckets={NAMESPACE: [FIRST, SECOND]}) as session:
            # ``buckets`` does not set the namespace.
            session.namespace.use(NAMESPACE)
            with session.transaction():
                print("users :", session.bucket.insert(FIRST, {"name": "Ada"}))
                print("orders:", session.bucket.insert(SECOND, {"total": 42}))


if __name__ == "__main__":
    main()
