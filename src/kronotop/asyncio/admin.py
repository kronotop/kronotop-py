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

"""Asynchronous cluster administration client."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Literal

import redis.asyncio
from redis.exceptions import ResponseError

from kronotop._callbacks import register_callbacks
from kronotop._options import (
    DEFAULT_ADMIN_PORT,
    apply_defaults,
    check_kwargs,
    parse_host,
)
from kronotop.commands.admin import AdminCommands
from kronotop.commands.volume import AsyncVolume
from kronotop.exceptions import wrap_response_error

__all__ = ["Admin"]


class Admin(AdminCommands):
    """The async twin of :class:`kronotop.Admin`. Same arguments, every
    command awaited.

    ```python
    async with Admin("10.0.0.1") as admin:
        await admin.describe_cluster()
    ```
    """

    _is_async_client: Literal[True] = True

    def __init__(self, host: str = f"localhost:{DEFAULT_ADMIN_PORT}", **kwargs: Any):
        check_kwargs(kwargs, "host")
        _, hostname, port = parse_host(host, "host", DEFAULT_ADMIN_PORT)
        apply_defaults(kwargs)
        self._redis = redis.asyncio.Redis(host=hostname, port=port, **kwargs)
        register_callbacks(self._redis)
        #: Volume commands, e.g. ``await admin.volume.describe("bucket-shard-0")``.
        self.volume = AsyncVolume(self)

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        """Send a raw command. Every method funnels through here."""
        try:
            return await self._redis.execute_command(*args, **options)  # type: ignore[no-untyped-call]
        except ResponseError as exc:
            raise wrap_response_error(exc) from None

    async def aclose(self) -> None:
        await self._redis.aclose()

    async def __aenter__(self) -> Admin:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
