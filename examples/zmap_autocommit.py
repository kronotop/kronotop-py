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

"""ZMap with no transaction block. Every command commits on its own.

Kronotop calls this auto-commit. With no ``BEGIN``, the server opens a
transaction for one command, runs it, and commits right away.

Run with ``uv run examples/zmap_autocommit.py``.
"""

from kronotop import Kronotop

KEY = "examples:zmap:greeting"


def main() -> None:
    with Kronotop() as client, client.session() as session:
        # One transaction per command. Each one is durable when the call
        # returns.
        session.zmap.set(KEY, "Hello")
        print("get         :", session.zmap.get(KEY))

        session.zmap.set(KEY, "World")
        print("overwritten :", session.zmap.get(KEY))

        print("missing key :", session.zmap.get("examples:zmap:no-such-key"))

        session.zmap.delete(KEY)
        print("deleted     :", session.zmap.get(KEY))

        # True says the command ran, not that the key existed.
        print("delete again:", session.zmap.delete(KEY))


if __name__ == "__main__":
    main()
