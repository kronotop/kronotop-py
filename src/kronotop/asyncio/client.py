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

"""Asynchronous Kronotop client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import redis.asyncio
from redis.asyncio.connection import parse_url

from kronotop._awaitable import AwaitableContext
from kronotop._members import Members
from kronotop._options import build_clients, check_kwargs, split_url
from kronotop._pending import arelease_pending
from kronotop._routing import (
    Found,
    Placement,
    check_targets,
    common_members,
    primaries,
)
from kronotop.asyncio.session import Session
from kronotop.config import KronotopConfig

__all__ = ["Kronotop", "from_url"]


class Kronotop:
    """The async twin of :class:`kronotop.Kronotop`.

    Same surface, same ``config`` and ``bootstrap``, every command awaited.
    :meth:`session` is not awaited, because it sends nothing. Commands live on
    :class:`kronotop.asyncio.Session`.

    ```python
    client = Kronotop()
    async with client.session() as s:
        await s.ping()  # True
    ```
    """

    def __init__(
        self,
        *,
        config: KronotopConfig | None = None,
        bootstrap: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Why check_kwargs comes first: see :meth:`kronotop.Kronotop.__init__`.
        check_kwargs(kwargs)
        self._config = config
        #: The cluster members. See :mod:`kronotop._members`.
        self._members = Members(
            redis.asyncio.Redis, build_clients(redis.asyncio.Redis, bootstrap, kwargs)
        )
        #: What ``session_for`` learned from ``LOCATE``. See :mod:`kronotop._routing`.
        self._placement = Placement()

    @classmethod
    def from_url(
        cls, url: str, *, config: KronotopConfig | None = None, **kwargs: Any
    ) -> Kronotop:
        """Build a client from a ``kronotop://`` URL.

        See :meth:`kronotop.Kronotop.from_url`.
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
    def _pools(self) -> dict[str, redis.asyncio.ConnectionPool]:
        """``host:port`` -> the pool that member's connections come from."""
        return {member.name: member.pool for member in self._members.all}

    def session(
        self,
        config: KronotopConfig | None = None,
        *,
        host: str | None = None,
        namespace: str | None = None,
    ) -> Session:
        """Open a session. Not a coroutine. See :meth:`kronotop.Kronotop.session`."""
        config = self._config if config is None else config
        member = self._members.next() if host is None else self._members.get(host)
        return Session(member.pool, config, namespace=namespace)

    def session_for(
        self,
        *,
        buckets: Mapping[str, Sequence[str]],
        config: KronotopConfig | None = None,
    ) -> AwaitableContext[Session]:
        """Open a session on a member that holds every bucket named.

        See :meth:`kronotop.Kronotop.session_for`. This one sends commands
        before it has the session, so it returns a wrapper to await or enter
        with ``async with``. A bad ``buckets`` argument raises here, before the
        await.

        ```python
        async with client.session_for(buckets={"app.prod": ["users"]}) as s:
            await s.namespace.use("app.prod")
        ```
        """
        check_targets(buckets)
        return AwaitableContext(self._session_for(buckets, config))

    async def _session_for(
        self,
        buckets: Mapping[str, Sequence[str]],
        config: KronotopConfig | None,
    ) -> Session:
        """Read where the buckets live, then open the session. See above."""
        found: Found = []
        # See the sync twin for why the finder is opened late.
        finder: Session | None = None
        try:
            for namespace, names in buckets.items():
                for name in names:
                    members = self._placement.get(namespace, name)
                    if members is None:
                        if finder is None:
                            finder = Session(self._members.next().pool)
                        located = await finder.bucket.locate(name, namespace=namespace)
                        members = primaries(located)
                        self._placement.keep(namespace, name, members)
                    found.append((namespace, name, members))
        finally:
            if finder is not None:
                await finder.close()
        # sorted, so the turn taking runs over a stable order.
        member = await self._members.anext_of(sorted(common_members(found)))
        return Session(
            member.pool,
            self._config if config is None else config,
            on_reject=lambda: self._placement.forget(buckets),
        )

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        """Send a raw command. See :meth:`kronotop.Kronotop.execute_command`.

        A coroutine, because the pending release has to be awaited.
        """
        for member in self._members.all:
            if member.queue:
                await arelease_pending(member.pool, member.queue)
        redis_client = self._members.next().client
        return await redis_client.execute_command(*args, **options)

    async def aclose(self) -> None:
        for member in self._members.all:
            await member.client.aclose()

    async def __aenter__(self) -> Kronotop:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def from_url(
    url: str, *, config: KronotopConfig | None = None, **kwargs: Any
) -> Kronotop:
    """Build an asyncio client from a ``kronotop://`` URL.

    Not a coroutine. See :func:`kronotop.from_url`.

    ```python
    from kronotop import asyncio as kronotop_asyncio

    client = kronotop_asyncio.from_url("kronotop://localhost")
    ```
    """
    return Kronotop.from_url(url, config=config, **kwargs)
