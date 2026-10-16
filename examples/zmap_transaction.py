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

"""ZMap inside a transaction block. Many commands, one atomic unit.

The block sends ``BEGIN`` on entry and ``COMMIT`` on a clean exit. If it raises,
it sends ``ROLLBACK`` instead and none of the writes become visible. See
``examples/zmap_autocommit.py`` for the other mode.

Run with ``uv run examples/zmap_transaction.py``.
"""

from kronotop import Kronotop

USER = {
    "examples:zmap:user:1:name": "Burak",
    "examples:zmap:user:1:city": "Istanbul",
    "examples:zmap:user:1:role": "engineer",
}


def main() -> None:
    with Kronotop() as client, client.session() as session:
        with session.transaction(returning="committed-version") as tx:
            for key, value in USER.items():
                session.zmap.set(key, value)

            # A read inside the block sees the writes the block just made.
            print("inside the block :", session.zmap.get("examples:zmap:user:1:name"))

        print("committed version:", tx.committed_version)

        # Back to auto-commit: one transaction per read.
        print("after commit     :", [session.zmap.get(key) for key in USER])


if __name__ == "__main__":
    main()
