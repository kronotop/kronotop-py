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

"""Hello world: connect to Kronotop and PING it.

Every command is on the session, ``ping`` included. The client only owns the
pools and opens sessions.

Run with ``uv run examples/hello_world.py``.
"""

import asyncio

from kronotop.asyncio import Kronotop


async def main() -> None:
    async with Kronotop() as client, client.session() as session:
        print(await session.ping("Hello world"))
        print(await session.info())


if __name__ == "__main__":
    asyncio.run(main())
