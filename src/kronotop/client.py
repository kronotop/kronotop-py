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

"""Synchronous Kronotop client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import redis
from redis.connection import parse_url

from kronotop._members import Members
from kronotop._options import (
    DEFAULT_PORT,
    build_clients,
    check_kwargs,
    split_url,
)
from kronotop._pending import release_pending
from kronotop._routing import (
    Found,
    Placement,
    check_targets,
    common_members,
    primaries,
)
from kronotop.config import KronotopConfig
from kronotop.session import Session

__all__ = ["DEFAULT_PORT", "Kronotop", "from_url"]


# Wraps ``redis.Redis`` instead of subclassing it, because a subclass would
# inherit the whole Redis command set.
class Kronotop:
    """A Kronotop client.

    The client owns the connection pools and opens sessions. Every command,
    ``ping`` included, is on :class:`kronotop.Session`: the pool hands a
    connection back after every command, and a command belongs to the session
    on that connection. :meth:`execute_command` sends a raw command without a
    session.

    The constructor takes the same keyword arguments as ``redis.Redis``, minus
    ``host``, ``port`` and ``connection_pool``, because ``bootstrap`` names the
    members. A few of the rest, such as ``db`` and client side caching, are
    Redis features Kronotop does not implement, and a non-default value raises
    :class:`~kronotop.exceptions.ConfigurationError`.

    ``config`` carries the session attributes and the namespace. Every session
    opened on this client starts from it.

    ``bootstrap`` names the members the client starts from, each ``host`` or
    ``host:port``, one pool each. The port defaults to ``5484`` and the list to
    ``localhost:5484``. There is no default member. A command or a session goes
    to each member in turn unless :meth:`session` names one, and :attr:`hosts`
    is the list as it stands.

    ```python
    client = Kronotop()  # localhost:5484
    with client.session() as s:
        s.ping()  # True

    client = Kronotop(bootstrap=["localhost:5484", "localhost:5485"])
    ```
    """

    def __init__(
        self,
        *,
        config: KronotopConfig | None = None,
        bootstrap: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Before redis.Redis, because single_connection_client opens a socket
        # inside its constructor.
        check_kwargs(kwargs)
        self._config = config
        #: The cluster members. See :mod:`kronotop._members`.
        self._members = Members(
            redis.Redis, build_clients(redis.Redis, bootstrap, kwargs)
        )
        #: What ``session_for`` learned from ``LOCATE``. See :mod:`kronotop._routing`.
        self._placement = Placement()

    @classmethod
    def from_url(
        cls, url: str, *, config: KronotopConfig | None = None, **kwargs: Any
    ) -> Kronotop:
        """Build a client from a ``kronotop://`` URL.

        The URL names one member. ``kronotops://`` is the TLS scheme. Any other
        scheme raises :class:`~kronotop.exceptions.ConfigurationError`. The port
        falls back to ``5484`` when the URL names none. A keyword argument wins
        over the same setting in the query string.

        ```python
        client = Kronotop.from_url("kronotop://localhost")
        client = Kronotop.from_url("kronotops://user:pw@kronotop.example.com")
        ```
        """
        check_kwargs(kwargs)
        host, url_kwargs = split_url(url, parse_url)
        return cls(config=config, bootstrap=[host], **{**url_kwargs, **kwargs})

    @property
    def config(self) -> KronotopConfig | None:
        """The configuration every session on this client starts from."""
        return self._config

    @property
    def hosts(self) -> tuple[str, ...]:
        """Every ``host:port`` this client knows, in the order they were added."""
        return tuple(member.name for member in self._members.all)

    @property
    def _pools(self) -> dict[str, redis.ConnectionPool]:
        """``host:port`` -> the pool that member's connections come from."""
        return {member.name: member.pool for member in self._members.all}

    def session(
        self,
        config: KronotopConfig | None = None,
        *,
        host: str | None = None,
        namespace: str | None = None,
    ) -> Session:
        """Open a session.

        A session is one pinned connection, taken on the first command and
        handed back by :meth:`Session.close`. The namespace, the session
        attributes, the open transaction and the cursors live on it.

        ``config`` is what the session sends before its first command. It
        replaces the client configuration, it does not merge with it. Derive
        one with ``dataclasses.replace``, or pass ``KronotopConfig()`` to send
        nothing.

        ``namespace`` is the dotted path the session works in, e.g.
        ``app.prod``. It has to exist, and it wins over the one in ``config``.

        ``host`` picks the member by name, one of :attr:`hosts`. Without it the
        member whose turn it is takes the session. A session stays on the
        member it opened on. A bucket command needs the primary of the
        bucket's shard, see :meth:`session_for`.

        ```python
        with client.session() as s:
            with s.transaction():
                ...
        ```
        """
        config = self._config if config is None else config
        member = self._members.next() if host is None else self._members.get(host)
        return Session(member.pool, config, namespace=namespace)

    def session_for(
        self,
        *,
        buckets: Mapping[str, Sequence[str]],
        config: KronotopConfig | None = None,
    ) -> Session:
        """Open a session on a member that holds every bucket named.

        A bucket command has to reach the primary of one of the bucket's
        shards. Any other member, a standby too, raises
        :class:`~kronotop.exceptions.RejectError`. This
        reads where each bucket lives and picks a member that holds them all,
        so one transaction can work on any of them. ``buckets`` maps a
        namespace to the bucket names in it, and there is no ``host``.

        Unlike :meth:`session`, this sends commands: one ``BUCKET.LOCATE`` per
        bucket it has not located before, on a session of its own. The answer
        is kept on the client. A bucket that moved raises ``RejectError`` on
        the session, which drops what was kept for this call's buckets, so the
        next call reads them again.

        Raises :class:`~kronotop.exceptions.NoCommonMemberError` when no
        member holds them all, and :class:`~kronotop.exceptions.NoSuchBucketError`
        for a bucket that does not exist. The session starts in the namespace
        ``config`` names, ``buckets`` does not set it.

        ``LOCATE`` names a member by its first address in the member table,
        which may differ from how ``bootstrap`` spelled it. Spellings of one
        address fold on their own, and a hostname is looked up once. A member
        the client has never talked to is added to :attr:`hosts` and takes its
        turn from then on.

        ```python
        with client.session_for(buckets={"app.prod": ["users", "orders"]}) as s:
            s.namespace.use("app.prod")
            with s.transaction():
                s.bucket.insert("users", {"name": "Alice"})
        ```
        """
        check_targets(buckets)
        found: Found = []
        # Opened on the first bucket that is not kept, so a call that has them
        # all sends nothing. No config on it, because each LOCATE names its own
        # namespace.
        finder: Session | None = None
        try:
            for namespace, names in buckets.items():
                for name in names:
                    members = self._placement.get(namespace, name)
                    if members is None:
                        if finder is None:
                            finder = Session(self._members.next().pool)
                        located = finder.bucket.locate(name, namespace=namespace)
                        members = primaries(located)
                        self._placement.keep(namespace, name, members)
                    found.append((namespace, name, members))
        finally:
            if finder is not None:
                finder.close()
        # sorted, so the turn taking runs over a stable order.
        member = self._members.next_of(sorted(common_members(found)))
        return Session(
            member.pool,
            self._config if config is None else config,
            on_reject=lambda: self._placement.forget(buckets),
        )

    def execute_command(self, *args: Any, **options: Any) -> Any:
        """Send a raw command to the member whose turn it is.

        The connection goes back to the pool right after, so the command must not
        depend on session state.
        """
        # Sessions share these pools. A slot a dead session left behind has to
        # come back here too, or the client starves on a small pool.
        for member in self._members.all:
            if member.queue:
                release_pending(member.pool, member.queue)
        redis_client = self._members.next().client
        return redis_client.execute_command(*args, **options)

    def close(self) -> None:
        for member in self._members.all:
            member.client.close()

    def __enter__(self) -> Kronotop:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def from_url(
    url: str, *, config: KronotopConfig | None = None, **kwargs: Any
) -> Kronotop:
    """Build a client from a ``kronotop://`` URL. See :meth:`Kronotop.from_url`.

    ```python
    import kronotop

    client = kronotop.from_url("kronotop://localhost")
    ```
    """
    return Kronotop.from_url(url, config=config, **kwargs)
