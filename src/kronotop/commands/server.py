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

"""Commands that ask the server about itself.

They read server state, so the connection they run on does not matter. The
``COMMAND`` family is ``session.command``, see :mod:`kronotop.commands.command`.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, overload

from kronotop.commands.base import AsyncClient, CommandsBase, SyncClient

__all__ = ["ServerCommands"]


class ServerCommands(CommandsBase):
    """``TIME`` and ``INFO``."""

    @overload
    def time(self: SyncClient) -> tuple[int, int]: ...

    @overload
    def time(self: AsyncClient) -> Awaitable[tuple[int, int]]: ...

    def time(self) -> Any:
        """``TIME``. Unix seconds and microseconds of the node that answers.

        Two nodes can report two times, and a clock adjustment can move the
        value backwards. For a value that only grows, use ``session.tick()``.
        """
        return self.execute_command("TIME")

    @overload
    def info(self: SyncClient, *sections: str) -> dict[str, str]: ...

    @overload
    def info(self: AsyncClient, *sections: str) -> Awaitable[dict[str, str]]: ...

    def info(self, *sections: str) -> Any:
        """``INFO``. Server information as one flat dict.

        The ``# Section`` headers are dropped, so every key sits in one dict.
        No name means every section. Names are case insensitive, ``all``,
        ``default`` and ``everything`` mean every section, and an unknown name
        is ignored.

        Keys and values are always ``str``, whatever ``decode_responses`` says.
        A value is never parsed, so a version such as ``1.10`` keeps its zero
        and ``listener0`` stays one ``name=...,bind=...`` string.
        """
        return self.execute_command("INFO", *sections)
