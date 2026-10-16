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

"""Asynchronous Kronotop session.

The async twin of :mod:`kronotop.session`, which explains why a session pins
one connection.

The finalizer differs. ``disconnect`` and ``ConnectionPool.release`` are
coroutines here, so ``__del__`` closes the transport synchronously and leaves
the release to the next session on the pool. See :mod:`kronotop._pending`.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Callable
from types import TracebackType
from typing import Any, Literal, cast

import redis.asyncio
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from kronotop._callbacks import register_callbacks
from kronotop._cleanup import ignore_cleanup_error
from kronotop._pending import arelease_pending
from kronotop._session_base import (
    SessionState,
    TransactionState,
    apply_callback,
    logger,
)
from kronotop.commands import AsyncKronotopSessionCommands, Returning
from kronotop.commands.bucket import AsyncBucket
from kronotop.commands.command import AsyncCommand
from kronotop.commands.namespace import AsyncNamespace
from kronotop.commands.zmap import AsyncZMap
from kronotop.config import KronotopConfig
from kronotop.exceptions import TransactionError

__all__ = ["Session", "Transaction"]


class Session(SessionState, AsyncKronotopSessionCommands):
    """A Kronotop session, backed by one pinned connection.

    Build one with :meth:`kronotop.asyncio.Kronotop.session`. A session is not
    safe to share between concurrent tasks, so open one per task.

    ```python
    async with client.session() as s:
        async with s.transaction():
            ...
    ```
    """

    _is_async_client: Literal[True] = True

    #: The concrete types behind the ``Any`` in :class:`SessionState`.
    _redis: redis.asyncio.Redis
    _pool: redis.asyncio.ConnectionPool
    _saved_retry: Retry | None

    def __init__(
        self,
        pool: redis.asyncio.ConnectionPool,
        config: KronotopConfig | None = None,
        *,
        namespace: str | None = None,
        on_reject: Callable[[], None] | None = None,
    ) -> None:
        self._redis = redis.asyncio.Redis(connection_pool=pool)
        register_callbacks(self._redis)
        #: ZMap commands, e.g. ``await session.zmap.set(key, value)``.
        self.zmap = AsyncZMap(self)
        #: Namespace commands, e.g. ``await session.namespace.use("app.prod")``.
        self.namespace = AsyncNamespace(self)
        #: Command metadata, e.g. ``await session.command.info("ping")``.
        self.command = AsyncCommand(self)
        #: Bucket commands, e.g. ``await session.bucket.create("users")``.
        self.bucket = AsyncBucket(self)
        # Last. See ``_init_state``.
        self._init_state(pool, config, namespace, on_reject)

    async def _acquire(self) -> None:
        """Pin a connection and configure it, on the first command.

        See :meth:`kronotop.session.Session._acquire`.
        """
        self._check_usable()
        if self._redis.connection is None:
            await arelease_pending(self._pool, self._pending)
            # get_connection is declared as AbstractConnection, but the pool
            # builds Connection objects, which is what Redis.connection wants.
            conn = cast(
                "redis.asyncio.Connection",
                await self._pool.get_connection(),  # type: ignore[no-untyped-call]
            )
            self._pin(conn, Retry(NoBackoff(), 0))
        if self._config_pending:
            await self._apply_config()

    async def _apply_config(self) -> None:
        """Send the session configuration, then read the attributes back.

        See :meth:`kronotop.session.Session._apply_config`.
        """
        with self._sending_config() as commands:
            for args in commands:
                await self.execute_command(*args)
            self._store_attributes(
                await self.execute_command("SESSION.ATTRIBUTE", "LIST")
            )

    @property
    async def attributes(self) -> dict[str, str | int]:
        self._check_usable()
        if len(self._attributes) == 0:
            await self._apply_config()
        return self._attributes

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        """Send a raw command on this session. Every command funnels through here."""
        await self._acquire()
        with self._tracking(str(args[0]).upper(), args):
            return await self._redis.execute_command(*args, **options)  # type: ignore[no-untyped-call]

    async def execute_blocking_command(self, *args: Any) -> Any:
        """Send a command that blocks the server. See the sync twin."""
        name = str(args[0]).upper()
        await self._acquire()
        with self._tracking(name, args):
            conn = self._redis.connection
            assert conn is not None  # ``_acquire`` pinned one.
            await conn.send_command(*args)
            # ``math.inf``, not ``None``. The async ``read_response`` reads
            # ``None`` as "use socket_timeout". redis-py 8.1.0 says it will swap
            # the two in its next major version.
            response = await conn.read_response(timeout=math.inf)
            return apply_callback(self._redis, name, response)

    def transaction(self, returning: Returning | None = None) -> Transaction:
        """An async context manager that wraps a transaction.

        With ``returning``, the result lands on the transaction object, and a
        transaction that wrote nothing reports version ``-1``:

        ```python
        async with s.transaction(returning="committed-version") as tx:
            ...
        print(tx.committed_version)
        ```
        """
        return Transaction(self, returning)

    async def _release(self, *, reset: bool) -> None:
        """Give the pinned connection back to the pool.

        See :meth:`kronotop.session.Session._release` for every step below.
        """
        conn: redis.asyncio.Connection | None = self._redis.connection
        if conn is None:
            return
        try:
            if reset and not self._broken:
                await self._redis.execute_command("SESSION.CLOSE")  # type: ignore[no-untyped-call]
        except Exception as exc:
            self._broken = True
            logger.warning("error ignored while releasing %r: %s", self, exc)
        finally:
            self._attributes.clear()
            self._unpin(conn)
            if self._broken or not reset:
                await conn.disconnect()
            if reset:
                await self._pool.release(conn)
            else:
                self._pending.append(conn)

    async def close(self) -> None:
        """End the session and give the connection back to the pool."""
        if self._closed:
            return
        self._closed = True
        await self._release(reset=True)

    def __del__(self, _warnings: Any = warnings) -> None:
        """Stop a forgotten session from taking a pool slot with it.

        Warns and closes the transport. Nothing can be awaited here, so the
        release waits for the next session on this pool. See
        :meth:`kronotop.session.Session.__del__` for the rest.
        """
        if getattr(self, "_closed", True):
            return
        conn: redis.asyncio.Connection | None = self._redis.connection
        if conn is None:
            return
        _warnings.warn(f"unclosed session {self!r}", ResourceWarning, source=self)
        self._closed = True
        # Without this, redis-py's ``Redis.__del__`` would close the socket and
        # never release the slot.
        self._unpin(conn)
        with contextlib.suppress(Exception):
            # What redis-py's own ``AbstractConnection.__del__`` calls. It
            # raises once the event loop is gone, so the append below stays
            # outside the block.
            # noinspection PyProtectedMember
            conn._close()  # type: ignore[no-untyped-call]
        self._pending.append(conn)

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


class Transaction(TransactionState):
    """The ``async with`` form of ``BEGIN`` and ``COMMIT``.

    Kronotop has no savepoints, so these blocks do not nest.
    """

    _session: Session

    async def __aenter__(self) -> Transaction:
        if self._session.in_transaction:
            raise TransactionError(
                "a transaction is already in progress on this session; "
                "Kronotop has no savepoints"
            )
        await self._session.begin()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # See :meth:`kronotop.session.Transaction.__exit__`.
            if self._session.in_transaction:
                with ignore_cleanup_error("rollback", self._session):
                    await self._session.rollback()
            return
        if self._returning is None:
            await self._session.commit()
        elif self._returning == "committed-version":
            self.committed_version = int(await self._session.commit(self._returning))
        else:
            self.versionstamp = bytes(await self._session.commit(self._returning))
