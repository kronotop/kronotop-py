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

"""Wait on one key and print it every time it changes.

Run with ``uv run examples/zmap_watch.py``. It only waits. Change the key
from another terminal to wake it:

    kronotop-cli -p 5484 ZSET examples:zmap:watched "a new value"

The signal has no value in it, so each wake reads the key. Ctrl-C ends it.
"""

from kronotop import Kronotop

KEY = "examples:zmap:watched"


def main() -> None:
    with Kronotop() as client, client.session() as session:
        print(f"watching {KEY}, now holding {session.zmap.get(KEY)!r}")
        try:
            while True:
                # The session sends nothing else until this returns. A program
                # with other work to do gives the watch its own session.
                session.zmap.watch(KEY)
                print("changed to :", session.zmap.get(KEY))
        except KeyboardInterrupt:
            # Ctrl-C interrupted the blocked read, which costs the socket. The
            # session ends here anyway.
            print()


if __name__ == "__main__":
    main()
