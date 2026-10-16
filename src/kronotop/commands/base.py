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

"""Shared parts of every command mixin.

Kept out of :mod:`kronotop.commands.core`, which imports the families. A
family that imported ``CommandsBase`` from there would make an import loop.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

__all__ = [
    "AsyncClient",
    "CommandGroup",
    "CommandsBase",
    "SyncClient",
    "namespace_args",
]


def namespace_args(namespace: str | None) -> tuple[str, ...]:
    """The ``NAMESPACE`` argument of a command, or nothing for ``None``."""
    return () if namespace is None else ("NAMESPACE", namespace)


class SyncClient(Protocol):
    """Structural type of a synchronous command target."""

    _is_async_client: Literal[False]

    def execute_command(self, *args: Any, **options: Any) -> Any: ...

    def execute_blocking_command(self, *args: Any) -> Any: ...


class AsyncClient(Protocol):
    """Structural type of an asyncio command target."""

    _is_async_client: Literal[True]

    def execute_command(self, *args: Any, **options: Any) -> Any: ...

    def execute_blocking_command(self, *args: Any) -> Any: ...


class CommandsBase:
    """Base for every command mixin.

    The client supplies ``execute_command``. It is declared here so the mixins
    type-check on their own.

    Each command has two ``@overload`` stanzas over one shared body. The type
    of ``self``, through ``_is_async_client``, picks one. The sync client sees
    a plain return type and the async client an ``Awaitable``, so
    ``print(client.ping())`` is never flagged as possibly awaitable.
    """

    #: Each client and command group redeclares this as a ``Literal``. Declared
    #: here too, so a shared body can branch on it. Only a command that reads
    #: the session attributes needs that, see ``BucketCommands``.
    _is_async_client: bool = False

    def execute_command(self, *args: Any, **options: Any) -> Any:
        raise NotImplementedError

    def execute_blocking_command(self, *args: Any) -> Any:
        """Send a command that blocks the server.

        Only a session supplies this. A blocking command holds its connection
        while it waits, and the client returns every connection to the pool
        after one command.
        """
        raise NotImplementedError


class CommandGroup:
    """Binds one command family to the object that owns the connection.

    Kronotop names its commands by family: ``BUCKET.INSERT``, ``ZSET.I64``,
    ``NAMESPACE USE``. The attribute carries the family name, so ``ZSET`` is
    ``session.zmap.set`` and ``NAMESPACE USE`` is ``session.namespace.use``.

    A group holds no connection. It forwards every command to its owner, so the
    session's transaction tracking, error mapping and pinned connection still
    apply.

    A concrete group lists this class first, then the mixin, so the forwarding
    ``execute_command`` wins over the one in ``CommandsBase`` that raises
    ``NotImplementedError``.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    @property
    def attributes(self) -> Any:
        """The session attributes, as the owner stores them.

        ``BucketCommands.insert`` reads ``input_type`` and ``object_id_format``
        from here. ``Any``, because the sync session gives a ``dict`` and the
        asyncio one a coroutine.
        """
        return self._owner.attributes

    def execute_command(self, *args: Any, **options: Any) -> Any:
        return self._owner.execute_command(*args, **options)

    def execute_blocking_command(self, *args: Any) -> Any:
        return self._owner.execute_blocking_command(*args)
