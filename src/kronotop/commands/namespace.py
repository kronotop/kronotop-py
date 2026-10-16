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

"""Namespace commands, reached through ``session.namespace``.

A namespace isolates data inside one cluster. Every bucket and ZMap key
resolves against the namespace the session uses, a dotted path such as
``app.prod``.

The current namespace is session state. It starts at the server's
``default_namespace``, and ``SESSION.CLOSE`` puts it back there.
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

__all__ = ["AsyncNamespace", "Namespace", "NamespaceCommands"]

#: What ``list`` returns. Spelled out here because inside the class ``list`` is
#: the method, and an IDE reads the annotation that way.
_Names = list[str]


class NamespaceCommands(CommandsBase):
    """``NAMESPACE USE``, ``CURRENT``, ``CREATE``, ``REMOVE``, ``EXISTS``,
    ``LIST``, ``MOVE`` and ``PURGE``.
    """

    @overload
    def use(self: SyncClient, namespace: str) -> bool: ...

    @overload
    def use(self: AsyncClient, namespace: str) -> Awaitable[bool]: ...

    def use(self, namespace: str) -> Any:
        """``NAMESPACE USE``. Point this session at ``namespace``.

        The namespace has to exist, or the server raises
        :class:`~kronotop.exceptions.NoSuchNamespaceError`. ``SESSION.CLOSE``
        resets this.
        """
        return self.execute_command("NAMESPACE", "USE", namespace, namespace_ok=True)

    @overload
    def current(self: SyncClient) -> str: ...

    @overload
    def current(self: AsyncClient) -> Awaitable[str]: ...

    def current(self) -> Any:
        """``NAMESPACE CURRENT``. The namespace this session uses, always a ``str``."""
        return self.execute_command("NAMESPACE", "CURRENT")

    @overload
    def create(
        self: SyncClient, namespace: str, *, if_not_exists: bool = False
    ) -> bool: ...

    @overload
    def create(
        self: AsyncClient, namespace: str, *, if_not_exists: bool = False
    ) -> Awaitable[bool]: ...

    def create(self, namespace: str, *, if_not_exists: bool = False) -> Any:
        """``NAMESPACE CREATE``. Create ``namespace``.

        A dotted path creates the levels above it too, so ``app.prod.api``
        creates ``app`` and ``app.prod`` on the way. Ten levels is the maximum,
        and ``__internal__`` is reserved at every level.

        With ``if_not_exists=True`` an existing name answers ``OK`` and stays
        as it is. A name that ``remove`` marked is refused with
        :class:`~kronotop.exceptions.NamespaceBeingRemovedError`, flag or not.

        Creating a namespace does not switch to it. Call ``use`` for that.
        """
        args: list[Any] = ["NAMESPACE", "CREATE", namespace]
        if if_not_exists:
            args.append("IF-NOT-EXISTS")
        return self.execute_command(*args, namespace_ok=True)

    @overload
    def remove(self: SyncClient, namespace: str) -> bool: ...

    @overload
    def remove(self: AsyncClient, namespace: str) -> Awaitable[bool]: ...

    def remove(self, namespace: str) -> Any:
        """``NAMESPACE REMOVE``. Mark ``namespace`` as removed.

        A logical removal. The data stays until ``NAMESPACE PURGE``, and until
        then the server answers ``NAMESPACEBEINGREMOVED`` to a new ``create``
        of the same name. Removing a namespace removes the ones under it, and
        the server's ``default_namespace`` cannot be removed.
        """
        return self.execute_command("NAMESPACE", "REMOVE", namespace, namespace_ok=True)

    @overload
    def exists(self: SyncClient, namespace: str) -> bool: ...

    @overload
    def exists(self: AsyncClient, namespace: str) -> Awaitable[bool]: ...

    def exists(self, namespace: str) -> Any:
        """``NAMESPACE EXISTS``. Whether ``namespace`` is there.

        A namespace that ``remove`` marked and ``purge`` has not deleted raises
        :class:`~kronotop.exceptions.NamespaceBeingRemovedError` instead of
        giving ``False``.
        """
        return self.execute_command(
            "NAMESPACE", "EXISTS", namespace, namespace_exists=True
        )

    @overload
    def list(self: SyncClient, namespace: str | None = None) -> _Names: ...

    @overload
    def list(self: AsyncClient, namespace: str | None = None) -> Awaitable[_Names]: ...

    def list(self, namespace: str | None = None) -> Any:
        """``NAMESPACE LIST``. The names one level under ``namespace``.

        Without an argument, the top level. A name comes back as a leaf, so
        ``list("app")`` on ``app.prod`` gives ``['prod']``. The names are
        ``str`` on any client, and ``__internal__`` is never among them.

        A namespace with nothing under it gives an empty list. A namespace that
        is not there raises :class:`~kronotop.exceptions.NoSuchNamespaceError`,
        except at the top level, where an empty cluster gives an empty list.
        """
        if namespace is None:
            return self.execute_command("NAMESPACE", "LIST")
        return self.execute_command("NAMESPACE", "LIST", namespace)

    @overload
    def move(self: SyncClient, old: str, new: str) -> bool: ...

    @overload
    def move(self: AsyncClient, old: str, new: str) -> Awaitable[bool]: ...

    def move(self, old: str, new: str) -> Any:
        """``NAMESPACE MOVE``. Rename ``old`` to ``new``.

        The data goes with the name. ``new`` has to be free, and ``old`` has to
        exist and not be marked for removal.

        The server keeps a tombstone under ``old`` and refuses ``create`` on it
        until every live member has seen the move. A session using ``old``
        loses it, because the name no longer resolves.
        """
        return self.execute_command("NAMESPACE", "MOVE", old, new, namespace_ok=True)

    @overload
    def purge(self: SyncClient, namespace: str) -> bool: ...

    @overload
    def purge(self: AsyncClient, namespace: str) -> Awaitable[bool]: ...

    def purge(self, namespace: str) -> Any:
        """``NAMESPACE PURGE``. Delete ``namespace`` for good.

        The second half of a removal, so ``remove`` has to run first. Purging a
        namespace purges the ones under it, and the server's
        ``default_namespace`` cannot be purged.

        Every live member has to have seen the removal, and Kronotop waits
        about five seconds for that. Otherwise this raises
        :class:`~kronotop.exceptions.BarrierNotSatisfiedError`, and one retry
        is usually enough.

        This deletes the directory. The volumes reclaim their bytes in a
        separate server job.
        """
        return self.execute_command("NAMESPACE", "PURGE", namespace, namespace_ok=True)


class Namespace(CommandGroup, NamespaceCommands):
    """``session.namespace`` on the synchronous session."""

    _is_async_client: Literal[False] = False


class AsyncNamespace(CommandGroup, NamespaceCommands):
    """``session.namespace`` on the asyncio session."""

    _is_async_client: Literal[True] = True
