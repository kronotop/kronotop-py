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

"""Session administration commands."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Literal, overload

from kronotop.commands.base import AsyncClient, CommandsBase, SyncClient

__all__ = [
    "InputType",
    "ObjectIdFormat",
    "ReplyType",
    "SessionAdminCommands",
    "SessionAttributeName",
]

#: Encoding of the documents a session sends. Server default ``bson``.
InputType = Literal["json", "bson"]

#: Encoding of the query results a session receives. Server default ``bson``.
ReplyType = Literal["json", "bson"]

#: How ObjectIds come back from DELETE and UPDATE. Server default ``bytes``.
ObjectIdFormat = Literal["bytes", "hex"]

#: The attributes ``SESSION.ATTRIBUTE`` knows. ``LIST`` echoes them in this
#: lowercase form.
SessionAttributeName = Literal["input_type", "reply_type", "batch", "object_id_format"]


class SessionAdminCommands(CommandsBase):
    """``SESSION.ATTRIBUTE SET`` and ``LIST``."""

    @overload
    def attribute_set(
        self: SyncClient, name: SessionAttributeName, value: str | int
    ) -> bool: ...

    @overload
    def attribute_set(
        self: AsyncClient, name: SessionAttributeName, value: str | int
    ) -> Awaitable[bool]: ...

    def attribute_set(self, name: SessionAttributeName, value: str | int) -> Any:
        """``SESSION.ATTRIBUTE SET``. Set one attribute on this session.

        The server rejects a bad name or value. :class:`kronotop.KronotopConfig`
        checks the pair before any command goes out.
        """
        return self.execute_command("SESSION.ATTRIBUTE", "SET", name, value)

    @overload
    def attribute_list(self: SyncClient) -> dict[Any, Any]: ...

    @overload
    def attribute_list(self: AsyncClient) -> Awaitable[dict[Any, Any]]: ...

    def attribute_list(self) -> Any:
        """``SESSION.ATTRIBUTE LIST``. Every attribute and its current value.

        Always a dict, on RESP2 and RESP3. Keys are ``bytes`` unless the client
        was built with ``decode_responses=True``. ``batch`` is an int, the other
        three are strings.
        """
        return self.execute_command("SESSION.ATTRIBUTE", "LIST")
