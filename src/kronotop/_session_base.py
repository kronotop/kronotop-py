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

"""The part of a session that is the same on both clients.

:mod:`kronotop.session` and :mod:`kronotop.asyncio.session` are twins, because
a body with ``await`` in it cannot be reused by sync code. The state and the
bookkeeping that touch no I/O live here, so they exist once.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from redis.exceptions import ConnectionError as _RedisConnectionError
from redis.exceptions import ResponseError
from redis.utils import str_if_bytes

from kronotop._pending import queue_for
from kronotop._transactions import (
    CLOSES_TRANSACTION,
    CLOSES_TRANSACTION_ON_ERROR,
    OPENS_TRANSACTION,
)
from kronotop.commands import Returning
from kronotop.exceptions import (
    KronotopConnectionError,
    RejectError,
    wrap_response_error,
)

if TYPE_CHECKING:
    from kronotop.config import KronotopConfig

#: Used for errors that must not replace the exception already on its way out.
logger = logging.getLogger("kronotop")


def apply_callback(redis_client: Any, name: str, response: Any) -> Any:
    """Run the response callback ``redis_client`` holds for ``name``.

    A blocking command reads its reply by hand, which skips ``parse_response``
    where the callback usually runs. A name with no callback gets the raw
    reply. Every Kronotop callback is a plain function, so the async session
    has nothing to await here.
    """
    callback = redis_client.response_callbacks.get(name)
    return response if callback is None else callback(response)


def attribute_name(name: Any) -> str:
    """One session attribute name, as the session stores it.

    ``LIST`` answers with ``bytes`` keys unless the client decodes responses.
    """
    return str(str_if_bytes(name)).lower()


def attribute_value(name: str, value: Any) -> str | int:
    """One session attribute value, as the session stores it.

    Runs on a ``LIST`` reply and on the argument of a ``SET``, so the same
    attribute never sits there as ``50`` from one side and ``"50"`` from the
    other. ``batch`` is an ``int``, the other three lowercase words.
    """
    text = str(str_if_bytes(value))
    return int(text) if name == "batch" else text.lower()


class SessionState:
    """Session state, and the reads and writes of it that need no ``await``.

    The attributes are ``Any`` here. Each session redeclares them with the
    redis-py types it wraps, so a strict type check still sees concrete types.
    """

    _redis: Any
    _pool: Any
    _saved_retry: Any
    _pending: Any
    #: What a session sends before its first command, or ``None``.
    _config: KronotopConfig | None
    #: Whether the pinned connection still needs the configuration and the
    #: attribute read. Set again after ``SESSION.CLOSE``, which restores the
    #: server defaults.
    _config_pending: bool
    #: What ``SESSION.ATTRIBUTE LIST`` answered after the configuration went
    #: out. Empty until then, and empty again after ``SESSION.CLOSE``.
    _attributes: dict[str, str | int]
    #: The namespace the session was opened with. Falls back to the one in the
    #: configuration, so it can stay ``None``.
    _namespace: str | None
    #: Called once per ``RejectError``. ``session_for`` passes one that drops
    #: the placement the session was opened from.
    _on_reject: Callable[[], None] | None
    _closed: bool
    _broken: bool
    _in_transaction: bool

    def _init_state(
        self,
        pool: Any,
        config: KronotopConfig | None = None,
        namespace: str | None = None,
        on_reject: Callable[[], None] | None = None,
    ) -> None:
        """Set the state up. Call this as the last statement of ``__init__``.

        Last, because the finalizer reads ``_closed`` first. A failure above
        this line leaves it unset, and the finalizer skips a half-built
        session.
        """
        self._pool = pool
        #: Connections a finalizer could not hand back. See
        #: :mod:`kronotop._pending`.
        self._pending = queue_for(pool)
        self._saved_retry = None
        self._config = config
        if namespace is None and config is not None:
            namespace = config.namespace
        self._namespace = namespace
        self._on_reject = on_reject
        # True even without a config: the ``LIST`` still has to run, because the
        # server defaults come from ``kronotop.toml`` and ``insert`` reads them.
        self._config_pending = True
        self._attributes = {}
        self._closed = False
        self._broken = False
        self._in_transaction = False

    def __repr__(self) -> str:
        """Name the class and the flags that are set, e.g. ``<... open>``.

        Every flag has a default, because the finalizer and the rollback path
        log this and can run on a session whose ``__init__`` failed halfway.
        """
        cls = type(self)
        flags = " ".join(
            name
            for name in ("closed", "broken", "in_transaction")
            if getattr(self, "_" + name, False)
        )
        return f"<{cls.__module__}.{cls.__qualname__} {flags or 'open'}>"

    @property
    def in_transaction(self) -> bool:
        """Whether a transaction is open, as far as this client knows."""
        return self._in_transaction

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def broken(self) -> bool:
        """Whether the pinned connection was lost. A broken session stays broken."""
        return self._broken

    def _forget_lost_connection(self) -> None:
        """Break the session when the pinned connection lost its socket.

        redis-py disconnects and re-raises on a timeout, a cancellation or a
        socket error. Without this the next command would reconnect and run
        in a new server side session, in auto-commit.
        """
        conn = self._redis.connection
        if conn is not None and not conn.is_connected:
            self._broken = True
            self._in_transaction = False

    def _check_usable(self) -> None:
        """Raise unless the session can still take a command.

        A connection can go away between two commands, so this looks at the
        socket every time.
        """
        if self._closed:
            raise KronotopConnectionError("session is closed")
        self._forget_lost_connection()
        if self._broken:
            raise KronotopConnectionError(
                "session connection was lost; open a new session"
            )

    def _pin(self, conn: Any, no_retry: Any) -> None:
        """Take ``conn`` for the life of the session.

        redis-py retries by reconnecting and sending again, which on a pinned
        connection drops the server side session and re-runs the command in
        auto-commit. The caller builds ``no_retry`` because the sync and async
        ``Retry`` classes differ.
        """
        self._saved_retry = conn.retry
        conn.retry = no_retry
        self._redis.connection = conn

    def _unpin(self, conn: Any) -> None:
        """Let go of ``conn``. It is still open here, and still needs a release."""
        if self._saved_retry is not None:
            conn.retry = self._saved_retry
            # Cleared with the connection it came from, so a later ``_unpin``
            # cannot put it on a connection that never had it.
            self._saved_retry = None
        # redis-py's own finalizer works on whatever ``connection`` points at.
        # Cleared, so it cannot touch a connection the session handed back.
        self._redis.connection = None
        self._in_transaction = False

    @contextlib.contextmanager
    def _tracking(self, name: str, args: Sequence[Any]) -> Iterator[None]:
        """Track what one command did, and map what it raised.

        Wraps the send in ``execute_command``, the only place a command reaches
        the socket.
        """
        try:
            yield
        except _RedisConnectionError:
            # The session died with the socket. Mark it broken, so the next
            # command raises instead of starting a fresh session.
            self._broken = True
            self._in_transaction = False
            raise
        except ResponseError as exc:
            self._track_error(name)
            wrapped = wrap_response_error(exc)
            if isinstance(wrapped, RejectError) and self._on_reject is not None:
                self._on_reject()
            raise wrapped from None
        except BaseException:
            # ``BaseException`` because asyncio.CancelledError is not an
            # ``Exception``. An error that never reached the socket, an encoding
            # error for one, leaves the session usable.
            self._forget_lost_connection()
            raise
        self._track(name, args)

    @contextlib.contextmanager
    def _sending_config(self) -> Iterator[list[tuple[str | int, ...]]]:
        """Hold the configuration flag while the caller sends the commands.

        The flag is cleared first, because each command calls ``_acquire``
        again and has to fall through the guard. A failure puts it back, so the
        next command tries the whole set again. The ``SET`` commands are
        idempotent, and the caller's command never runs under a configuration
        they did not get.

        A session without a config yields no command. The caller still reads
        the attributes back.
        """
        self._config_pending = False
        try:
            if self._config is not None:
                yield self._config.to_commands(self._namespace)
            elif self._namespace is not None:
                yield [("NAMESPACE", "USE", self._namespace)]
            else:
                yield []
        except BaseException:
            self._config_pending = True
            raise

    def _store_attributes(self, reply: dict[Any, Any]) -> None:
        """Keep a ``SESSION.ATTRIBUTE LIST`` reply, in one shape."""
        attributes: dict[str, str | int] = {}
        for name, value in reply.items():
            key = attribute_name(name)
            attributes[key] = attribute_value(key, value)
        self._attributes = attributes

    def _track(self, name: str, args: Sequence[Any]) -> None:
        if name in OPENS_TRANSACTION:
            self._in_transaction = True
        elif name in CLOSES_TRANSACTION:
            self._in_transaction = False
        if name == "SESSION.CLOSE":
            # The server restored its defaults. There is no method for the
            # command, so this only fires when a caller sends it raw.
            self._config_pending = True
            # The stored values are the ones the defaults replaced.
            self._attributes = {}
        elif (
            name == "SESSION.ATTRIBUTE"
            and len(args) == 4
            and attribute_name(args[1]) == "set"
        ):
            # The server took the value, so no second ``LIST`` is needed. A
            # caller who sends the command raw comes through here too.
            key = attribute_name(args[2])
            self._attributes[key] = attribute_value(key, args[3])

    def _track_error(self, name: str) -> None:
        if name in CLOSES_TRANSACTION_ON_ERROR:
            self._in_transaction = False


class TransactionState:
    """What a transaction block holds, before any ``BEGIN`` is sent.

    Only the shared state. The ``with`` and ``async with`` bodies differ, so
    each module writes its own.
    """

    _session: Any

    def __init__(self, session: Any, returning: Returning | None = None) -> None:
        self._session = session
        self._returning = returning
        #: Set after the commit, when ``returning="committed-version"`` was asked
        #: for. ``-1`` means the transaction wrote nothing.
        self.committed_version: int | None = None
        #: Set after the commit, when ``returning="versionstamp"`` was asked for.
        self.versionstamp: bytes | None = None
