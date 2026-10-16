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

"""``COMMAND`` commands, reached through ``session.command``.

The replies come from the command definitions the server loaded at startup, so
a command without a definition is not listed. A subcommand is named
``container|subcommand``, for example ``kr.admin|describe-cluster``.

Every name, flag and text field comes back as ``str``, on either protocol and
whatever ``decode_responses`` says.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Literal, overload

from kronotop.commands.base import (
    AsyncClient,
    CommandGroup,
    CommandsBase,
    SyncClient,
)

__all__ = ["AsyncCommand", "Command", "CommandCommands", "CommandDoc", "CommandInfo"]

#: One ``COMMAND INFO`` entry: ``name``, ``arity``, ``flags``, ``first_key``,
#: ``last_key``, ``step``, ``acl_categories``, ``tips``, ``key_specifications``
#: and ``subcommands``.
CommandInfo = dict[str, Any]

#: One ``COMMAND DOCS`` entry: ``summary``, ``since``, ``group``,
#: ``complexity``, ``doc_flags``, ``history``, ``reply_schema``, ``arguments``
#: and ``subcommands``. A field the definition leaves empty is left out.
CommandDoc = dict[str, Any]

#: What ``list``, ``getkeys`` and ``help`` return. Spelled out here because
#: inside the class ``list`` is the method, and an IDE reads the annotation
#: that way.
_Names = list[str]
_KeysAndFlags = list[tuple[str, list[str]]]


class CommandCommands(CommandsBase):
    """``COMMAND INFO``, ``COUNT``, ``DOCS``, ``LIST``, ``GETKEYS``,
    ``GETKEYSANDFLAGS`` and ``HELP``.

    Each subcommand is sent as one string, ``"COMMAND INFO"``, because the
    response callback is looked up by the whole first argument and the one for
    bare ``COMMAND`` reads the ``INFO`` shape only. The string is split in two
    on the way out.
    """

    @overload
    def info(self: SyncClient, *names: str) -> dict[str, CommandInfo | None]: ...

    @overload
    def info(
        self: AsyncClient, *names: str
    ) -> Awaitable[dict[str, CommandInfo | None]]: ...

    def info(self, *names: str) -> Any:
        """``COMMAND INFO``. What the server knows about its commands.

        No name means every command. The reply is a dict keyed by lowercase
        command name, names matched case insensitively. An unknown name maps
        to ``None``.

        ``flags``, ``acl_categories`` and ``tips`` are lists of ``str``.
        ``key_specifications`` is a list of dicts with ``flags``,
        ``begin_search``, ``find_keys`` and an optional ``notes``.
        ``subcommands`` is a dict of the same shape, keyed by
        ``container|subcommand``.
        """
        return self.execute_command("COMMAND INFO", *names, command_names=names)

    @overload
    def count(self: SyncClient) -> int: ...

    @overload
    def count(self: AsyncClient) -> Awaitable[int]: ...

    def count(self) -> Any:
        """``COMMAND COUNT``. How many commands have a definition."""
        return self.execute_command("COMMAND COUNT")

    @overload
    def docs(self: SyncClient, *names: str) -> dict[str, CommandDoc]: ...

    @overload
    def docs(self: AsyncClient, *names: str) -> Awaitable[dict[str, CommandDoc]]: ...

    def docs(self, *names: str) -> Any:
        """``COMMAND DOCS``. The documentation of commands.

        No name means every documented command. The reply is a dict keyed by
        lowercase command name, names matched case insensitively, unknown
        names skipped.

        ``group`` is always present, the other fields only when the definition
        fills them. ``history`` is a list of ``(version, change)`` tuples.
        ``arguments`` is a list of dicts with ``name``, ``type``,
        ``display_text``, ``token``, ``since``, ``flags`` and, for a ``oneof``
        or a ``block``, a nested ``arguments`` list. ``reply_schema`` is the
        JSON Schema of the reply. ``subcommands`` is a dict of the same shape.
        """
        return self.execute_command("COMMAND DOCS", *names)

    @overload
    def list(
        self: SyncClient,
        *,
        module: str | None = None,
        category: str | None = None,
        pattern: str | None = None,
    ) -> _Names: ...

    @overload
    def list(
        self: AsyncClient,
        *,
        module: str | None = None,
        category: str | None = None,
        pattern: str | None = None,
    ) -> Awaitable[_Names]: ...

    def list(
        self,
        *,
        module: str | None = None,
        category: str | None = None,
        pattern: str | None = None,
    ) -> Any:
        """``COMMAND LIST``. The names of every command, subcommands included.

        One filter at most, sent as ``FILTERBY``. ``module`` is ``MODULE`` and
        always gives an empty list, there are no modules. ``category`` is
        ``ACLCAT`` without the ``@``, unknown gives an empty list. ``pattern``
        is ``PATTERN``, a case insensitive glob. Two filters are a ``syntax
        error`` on the server.
        """
        args: list[str] = []
        if module is not None:
            args += ["MODULE", module]
        if category is not None:
            args += ["ACLCAT", category]
        if pattern is not None:
            args += ["PATTERN", pattern]
        if args:
            args.insert(0, "FILTERBY")
        return self.execute_command("COMMAND LIST", *args)

    @overload
    def getkeys(self: SyncClient, *args: Any) -> _Names: ...

    @overload
    def getkeys(self: AsyncClient, *args: Any) -> Awaitable[_Names]: ...

    def getkeys(self, *args: Any) -> Any:
        """``COMMAND GETKEYS``. The key arguments of a full command.

        ``args`` is the command as it would be sent, name first. The server
        refuses an unknown command, one without key specifications, or a wrong
        argument count with a plain ``redis.ResponseError``.
        """
        return self.execute_command("COMMAND GETKEYS", *args)

    @overload
    def getkeysandflags(self: SyncClient, *args: Any) -> _KeysAndFlags: ...

    @overload
    def getkeysandflags(self: AsyncClient, *args: Any) -> Awaitable[_KeysAndFlags]: ...

    def getkeysandflags(self, *args: Any) -> Any:
        """``COMMAND GETKEYSANDFLAGS``. As :meth:`getkeys`, with flags.

        Every item is a ``(key, flags)`` tuple, the flags a list of ``str``
        such as ``RO`` and ``access``.
        """
        return self.execute_command("COMMAND GETKEYSANDFLAGS", *args)

    @overload
    def help(self: SyncClient) -> _Names: ...

    @overload
    def help(self: AsyncClient) -> Awaitable[_Names]: ...

    def help(self) -> Any:
        """``COMMAND HELP``. One line per subcommand, as the server words it."""
        return self.execute_command("COMMAND HELP")


class Command(CommandGroup, CommandCommands):
    """``session.command`` on the synchronous session."""

    _is_async_client: Literal[False] = False


class AsyncCommand(CommandGroup, CommandCommands):
    """``session.command`` on the asyncio session."""

    _is_async_client: Literal[True] = True
