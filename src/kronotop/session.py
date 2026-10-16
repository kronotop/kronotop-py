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

"""Synchronous Kronotop session.

One TCP connection is one session. The open transaction, the cursors and the
session attributes live on it. The client hands a pooled connection back after
every command, which would let ``BEGIN`` and the next command land on
different connections. A ``Session`` pins one connection for its whole life.

``close`` sends ``SESSION.CLOSE``, which resets the state without dropping the
socket, then releases the connection. A session that loses its socket is
finished and refuses further commands, because a reconnect would open a new
server side session.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from types import TracebackType
from typing import Any, Literal

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from kronotop._callbacks import register_callbacks
from kronotop._cleanup import ignore_cleanup_error
from kronotop._pending import release_pending
from kronotop._session_base import (
    SessionState,
    TransactionState,
    apply_callback,
    logger,
)
from kronotop.commands import KronotopSessionCommands, Returning
from kronotop.commands.bucket import Bucket
from kronotop.commands.command import Command
from kronotop.commands.namespace import Namespace
from kronotop.commands.zmap import ZMap
from kronotop.config import KronotopConfig
from kronotop.exceptions import TransactionError

__all__ = ["Session", "Transaction"]


class Session(SessionState, KronotopSessionCommands):
    _is_async_client: Literal[False] = False

    #: The concrete types behind the ``Any`` in :class:`SessionState`.
    _redis: redis.Redis
    _pool: redis.ConnectionPool
    _saved_retry: Retry | None

    def __init__(
        self,
        pool: redis.ConnectionPool,
        config: KronotopConfig | None = None,
        *,
        namespace: str | None = None,
        on_reject: Callable[[], None] | None = None,
    ) -> None:
        self._redis = redis.Redis(connection_pool=pool)
        register_callbacks(self._redis)
        #: ZMap commands, e.g. ``session.zmap.set(key, value)``.
        self.zmap = ZMap(self)
        #: Namespace commands, e.g. ``session.namespace.use("app.prod")``.
        self.namespace = Namespace(self)
        #: Command metadata, e.g. ``session.command.info("ping")``.
        self.command = Command(self)
        #: Bucket commands, e.g. ``session.bucket.create("users")``.
        self.bucket = Bucket(self)
        # Last. See ``_init_state``.
        self._init_state(pool, config, namespace, on_reject)

    def _acquire(self) -> None:
        """Pin a connection and configure it, on the first command.

        Running here keeps ``session()`` free of I/O. The two guards stay
        separate: the connection is pinned before the configuration goes out,
        and a single guard would never retry a configuration that failed
        halfway.
        """
        self._check_usable()
        if self._redis.connection is None:
            # A session that died without ``close`` left its connection in the
            # queue. Handing it back now keeps the pool from running out.
            release_pending(self._pool, self._pending)
            self._pin(self._pool.get_connection(), Retry(NoBackoff(), 0))
        if self._config_pending:
            self._apply_config()

    def _apply_config(self) -> None:
        """Send the session configuration, then read the attributes back.

        The ``LIST`` runs inside the block, so a failure on it sends the whole
        set again on the next command. A session without a config still sends
        the ``LIST``, because ``insert`` needs the attributes and the server
        defaults come from ``kronotop.toml``.
        """
        with self._sending_config() as commands:
            for args in commands:
                self.execute_command(*args)
            self._store_attributes(self.execute_command("SESSION.ATTRIBUTE", "LIST"))

    @property
    def attributes(self) -> dict[str, str | int]:
        # A closed session raises here, like every other operation on it.
        self._check_usable()
        if len(self._attributes) == 0:
            self._apply_config()
        return self._attributes

    def execute_command(self, *args: Any, **options: Any) -> Any:
        """Send a raw command on this session. Every command funnels through here."""
        self._acquire()
        with self._tracking(str(args[0]).upper(), args):
            return self._redis.execute_command(*args, **options)  # type: ignore[no-untyped-call]

    def execute_blocking_command(self, *args: Any) -> Any:
        """Send a command that blocks the server. The read has no timeout.

        Only ``ZWATCH`` needs this. A read that hits ``socket_timeout`` costs
        the socket and ends the session, so a client built with
        ``socket_timeout`` would lose the session on every quiet key.
        """
        name = str(args[0]).upper()
        self._acquire()
        with self._tracking(name, args):
            conn = self._redis.connection
            assert conn is not None  # ``_acquire`` pinned one.
            # By hand, because ``Redis.parse_response`` takes no timeout.
            conn.send_command(*args)  # type: ignore[no-untyped-call]
            # ``timeout=None`` drops the socket timeout for this one read and
            # puts it back after.
            response = conn.read_response(timeout=None)
            return apply_callback(self._redis, name, response)

    def transaction(self, returning: Returning | None = None) -> Transaction:
        """A context manager that wraps a transaction.

        ``BEGIN`` on entry, ``COMMIT`` on a clean exit, ``ROLLBACK`` when the
        block raises. With ``returning``, the result lands on the transaction
        object, and a transaction that wrote nothing reports version ``-1``:

        ```python
        with s.transaction(returning="committed-version") as tx:
            ...
        print(tx.committed_version)
        ```
        """
        return Transaction(self, returning)

    def _release(self, *, reset: bool) -> None:
        """Give the pinned connection back to the pool.

        ``reset`` sends ``SESSION.CLOSE`` first, so the connection goes back
        clean. Without it the socket is dropped, and the server discards the
        session when the channel unregisters, which rolls back the transaction.
        """
        # ``_acquire`` is the only writer of ``Redis.connection`` here, and it
        # writes a Connection. redis-py leaves the attribute unannotated.
        conn: redis.Connection | None = self._redis.connection
        if conn is None:
            return
        try:
            if reset and not self._broken:
                self._redis.execute_command("SESSION.CLOSE")  # type: ignore[no-untyped-call]
        except Exception as exc:
            # A release must not raise, whatever the socket does.
            self._broken = True
            logger.warning("error ignored while releasing %r: %s", self, exc)
        finally:
            self._attributes.clear()
            self._unpin(conn)
            if self._broken or not reset:
                conn.disconnect()  # type: ignore[no-untyped-call]
            if reset:
                self._pool.release(conn)
            else:
                # ``reset=False`` only comes from the finalizer. See
                # :mod:`kronotop._pending`.
                self._pending.append(conn)

    def close(self) -> None:
        """End the session and give the connection back to the pool.

        Sends ``SESSION.CLOSE`` first, which rolls back an open transaction and
        clears the rest of the state. Calling this twice is safe.
        """
        if self._closed:
            return
        self._closed = True
        self._release(reset=True)

    def __del__(self, _warnings: Any = warnings) -> None:
        """Stop a forgotten session from damaging the pool.

        Warns and drops the socket. A finalizer must not do I/O, so it cannot
        send ``SESSION.CLOSE``. The session sits in a reference cycle through
        ``self.zmap``, so this runs on a cyclic GC pass.
        """
        # ``_warnings`` is a default argument so the module is still reachable
        # during interpreter shutdown, as in redis-py's
        # ``AbstractConnection.__del__``. The ``True`` default covers a session
        # whose ``__init__`` failed before ``_init_state``.
        if getattr(self, "_closed", True):
            return
        if self._redis.connection is None:
            return
        _warnings.warn(f"unclosed session {self!r}", ResourceWarning, source=self)
        self._closed = True
        # Without this, redis-py's own ``Redis.__del__`` would release the
        # connection with the transaction still on it. The ``suppress`` is
        # there because a finalizer must not raise.
        with contextlib.suppress(Exception):
            self._release(reset=False)

    def __enter__(self) -> Session:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class Transaction(TransactionState):
    """The ``with`` form of ``BEGIN`` and ``COMMIT``.

    Kronotop has no savepoints, so these blocks do not nest. Entering one while
    a transaction is open raises :class:`~kronotop.exceptions.TransactionError`.
    """

    _session: Session

    def __enter__(self) -> Transaction:
        if self._session.in_transaction:
            raise TransactionError(
                "a transaction is already in progress on this session; "
                "Kronotop has no savepoints"
            )
        self._session.begin()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # SESSION.CLOSE, a failed COMMIT and a lost connection all clear the
            # flag, so this skips the cases with nothing left to roll back.
            if self._session.in_transaction:
                # The socket can also die between the last command and here,
                # which ``_acquire`` only notices now.
                with ignore_cleanup_error("rollback", self._session):
                    self._session.rollback()
            return
        if self._returning is None:
            self._session.commit()
        elif self._returning == "committed-version":
            self.committed_version = int(self._session.commit(self._returning))
        else:
            self.versionstamp = bytes(self._session.commit(self._returning))
