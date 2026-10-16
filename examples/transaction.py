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

"""Open a session and run a transaction.

Run with ``uv run examples/transaction.py``.
"""

from kronotop import Kronotop


def main():
    # A session pins one connection. Kronotop keeps the transaction on the
    # connection, so every command in the block below reaches the same session.
    with Kronotop() as client, client.session() as session:
        with session.transaction(returning="committed-version") as tx:
            print("tick:", session.tick("fresh"))
            print("read version:", session.get_read_version())
            print("approximate size:", session.get_approximate_size())

        # Nothing was written, so the committed version is -1.
        print("committed version:", tx.committed_version)


if __name__ == "__main__":
    main()
