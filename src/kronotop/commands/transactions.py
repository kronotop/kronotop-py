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

"""Transaction commands.

A transaction is bound to a session, so these are on
:class:`kronotop.session.Session` only. Prefer ``with session.transaction():``
over ``begin`` and ``commit``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Literal, overload

from kronotop.commands.base import AsyncClient, CommandsBase, SyncClient

__all__ = ["Returning", "TickMode", "TransactionCommands"]

#: What ``COMMIT RETURNING`` can ask for.
Returning = Literal["committed-version", "versionstamp"]

#: Where ``TICK`` reads its value from.
TickMode = Literal["fresh", "cached"]


class TransactionCommands(CommandsBase):
    """``BEGIN``, ``COMMIT``, ``ROLLBACK`` and transaction inspection.

    ``TICK`` sits here too, because the Kronotop docs group it with them.
    """

    @overload
    def begin(self: SyncClient) -> bool: ...

    @overload
    def begin(self: AsyncClient) -> Awaitable[bool]: ...

    def begin(self) -> Any:
        """``BEGIN``. Open a transaction on this session.

        Until ``commit`` or ``rollback``, every command that touches
        FoundationDB joins it. Kronotop has no savepoints, so a second
        ``begin`` raises :class:`~kronotop.exceptions.TransactionError`.
        """
        return self.execute_command("BEGIN")

    @overload
    def commit(self: SyncClient, returning: None = None) -> bool: ...

    @overload
    def commit(self: SyncClient, returning: Returning) -> int | bytes: ...

    @overload
    def commit(self: AsyncClient, returning: None = None) -> Awaitable[bool]: ...

    @overload
    def commit(self: AsyncClient, returning: Returning) -> Awaitable[int | bytes]: ...

    def commit(self, returning: Returning | None = None) -> Any:
        """``COMMIT``. Commit the open transaction.

        Returns ``True``, or with ``returning`` a value from the commit:

        * ``"committed-version"`` gives the commit version as an ``int``. It is
          ``-1`` when the transaction wrote nothing.
        * ``"versionstamp"`` gives the 10 byte versionstamp. The transaction
          must have written at least once, otherwise the server raises
          :class:`~kronotop.exceptions.NoCommitVersionError`.
        """
        if returning is None:
            return self.execute_command("COMMIT")
        # See ``parse_commit`` in kronotop._callbacks.
        return self.execute_command(
            "COMMIT", "RETURNING", returning, commit_returning=True
        )

    @overload
    def rollback(self: SyncClient) -> bool: ...

    @overload
    def rollback(self: AsyncClient) -> Awaitable[bool]: ...

    def rollback(self) -> Any:
        """``ROLLBACK``. Abort the open transaction and drop its changes."""
        return self.execute_command("ROLLBACK")

    @overload
    def get_read_version(self: SyncClient) -> int: ...

    @overload
    def get_read_version(self: AsyncClient) -> Awaitable[int]: ...

    def get_read_version(self) -> Any:
        """``GETREADVERSION``. The read version of the open transaction.

        Takes the read version if no read has happened yet, which starts
        FoundationDB's five second window.
        """
        return self.execute_command("GETREADVERSION")

    @overload
    def get_approximate_size(self: SyncClient) -> int: ...

    @overload
    def get_approximate_size(self: AsyncClient) -> Awaitable[int]: ...

    def get_approximate_size(self) -> Any:
        """``GETAPPROXIMATESIZE``. Approximate byte size of the mutations so far.

        FoundationDB caps a transaction at 10 MB, so use this to split large
        work before the commit fails.
        """
        return self.execute_command("GETAPPROXIMATESIZE")

    @overload
    def snapshot_read(self: SyncClient, enabled: bool) -> bool: ...

    @overload
    def snapshot_read(self: AsyncClient, enabled: bool) -> Awaitable[bool]: ...

    def snapshot_read(self, enabled: bool) -> Any:
        """``SNAPSHOTREAD``. Turn snapshot reads on or off.

        Snapshot reads take no read conflict ranges, so concurrent writes to
        the same keys do not make the transaction conflict. The setting is
        session state: it survives ``commit`` and ``rollback``.
        """
        return self.execute_command("SNAPSHOTREAD", "ON" if enabled else "OFF")

    @overload
    def tick(self: SyncClient, mode: TickMode) -> int: ...

    @overload
    def tick(self: AsyncClient, mode: TickMode) -> Awaitable[int]: ...

    def tick(self, mode: TickMode) -> Any:
        """``TICK``. A 64 bit integer that never goes backwards.

        ``"fresh"`` reads the cluster value, one round trip, monotonic cluster
        wide. ``"cached"`` reads a node-local value up to one second behind
        with no round trip, so a client that moves to another node can see a
        smaller one.

        The same value can come back twice, even with ``"fresh"``, so it is
        not an ID generator.
        """
        return self.execute_command("TICK", mode)

    @overload
    def list_transactions(self: SyncClient) -> list[dict[str, Any]]: ...

    @overload
    def list_transactions(self: AsyncClient) -> Awaitable[list[dict[str, Any]]]: ...

    def list_transactions(self) -> Any:
        """``TRANSACTION LIST``. The open transactions of this session's member.

        One dict per transaction, ordered by ``id``. Other members are not
        listed.
        """
        return self.execute_command("TRANSACTION", "LIST")

    @overload
    def slow_transactions(self: SyncClient) -> list[dict[str, Any]]: ...

    @overload
    def slow_transactions(self: AsyncClient) -> Awaitable[list[dict[str, Any]]]: ...

    def slow_transactions(self) -> Any:
        """``TRANSACTION SLOW``. Closed transactions that ran past the threshold.

        The server sets the threshold. Newest first, this member only, and a
        restart clears the log.
        """
        return self.execute_command("TRANSACTION", "SLOW")

    @overload
    def kill_transaction(self: SyncClient, transaction_id: int) -> bool: ...

    @overload
    def kill_transaction(self: AsyncClient, transaction_id: int) -> Awaitable[bool]: ...

    def kill_transaction(self, transaction_id: int) -> Any:
        """``TRANSACTION KILL``. Cancel a transaction of this session's member.

        ``transaction_id`` is the ``id`` of a :meth:`list_transactions` entry.
        The owner session gets
        :class:`~kronotop.exceptions.TransactionCancelledError` on its next
        read or commit and has to roll back.
        """
        return self.execute_command("TRANSACTION", "KILL", transaction_id)
