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

"""Connection commands: ``PING`` and ``ECHO``."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, overload

from kronotop.commands.base import AsyncClient, CommandsBase, SyncClient

__all__ = ["ConnectionCommands"]


class ConnectionCommands(CommandsBase):
    """``PING`` and ``ECHO``."""

    @overload
    def ping(self: SyncClient, message: str | None = None) -> bool | bytes | str: ...

    @overload
    def ping(
        self: AsyncClient, message: str | None = None
    ) -> Awaitable[bool | bytes | str]: ...

    def ping(self, message: str | None = None) -> Any:
        """``PING``. Returns ``True``, or the echo of ``message`` when one is given."""
        if message is None:
            return self.execute_command("PING")
        return self.execute_command("PING", message, ping_message=True)

    @overload
    def echo(self: SyncClient, message: str) -> bytes | str: ...

    @overload
    def echo(self: AsyncClient, message: str) -> Awaitable[bytes | str]: ...

    def echo(self, message: str) -> Any:
        """``ECHO``. Send a message and get it back.

        The reply is ``bytes`` unless the client was built with
        ``decode_responses=True``.
        """
        return self.execute_command("ECHO", message)
