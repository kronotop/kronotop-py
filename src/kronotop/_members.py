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

"""The cluster members a client talks to, and the turn taking between them.

``bootstrap`` is where the list starts. Members are added and removed while
the client runs, from any thread. Both clients keep their members here, so the
rules exist once.

The list is never changed in place. A writer builds a new tuple under the lock
and assigns it. A reader takes the tuple once, so it never sees half a change.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from kronotop._callbacks import register_callbacks
from kronotop._options import (
    aresolve,
    canonical_host,
    clone_pool,
    parse_host,
    resolve,
)
from kronotop._pending import queue_for
from kronotop.exceptions import ConfigurationError

__all__ = ["Member", "Members", "member_key"]


class Member(NamedTuple):
    """One cluster member, with everything a command needs to reach it."""

    #: ``host:port``, the name the member is known by.
    name: str
    #: The redis-py client that talks to it.
    client: Any
    #: The pool that client's connections come from.
    pool: Any
    #: Connections a session finalizer left behind. See :mod:`kronotop._pending`.
    queue: deque[Any]
    #: The lookup key. ``localhost:5484`` and ``127.0.0.1:5484`` share one, see
    #: :func:`~kronotop._options.canonical_host`.
    key: tuple[str, int]


def member_key(host: str) -> tuple[str, int]:
    """The lookup key of one ``host`` or ``host:port`` entry."""
    _, hostname, port = parse_host(host)
    return canonical_host(hostname), port


def _member(name: str, client: Any) -> Member:
    """Build a :class:`Member` around a redis-py client."""
    register_callbacks(client)
    pool = client.connection_pool
    return Member(name, client, pool, queue_for(pool), member_key(name))


class Members:
    """The members of one client, in the order commands go to them.

    ``factory`` is ``redis.Redis`` or its asyncio twin. It is called with a pool,
    so the same code serves both clients.
    """

    def __init__(self, factory: Callable[..., Any], clients: dict[str, Any]) -> None:
        self._factory = factory
        #: Guards the two attributes below. :meth:`add` also holds it while it
        #: builds the pool and the client, so two threads cannot open two pools
        #: for one member. Neither step opens a socket.
        self._lock = threading.Lock()
        self._members = tuple(_member(name, client) for name, client in clients.items())
        self._turn = 0
        #: Canonical host -> the addresses it stands for. Filled by :meth:`add`,
        #: one lookup per host. Never cleared, because a member that changed
        #: address comes back from the cluster under a new name. See
        #: :meth:`_keep` for the lookups that do not land here.
        self._addresses: dict[str, frozenset[str]] = {}

    @property
    def all(self) -> tuple[Member, ...]:
        """Every member, as one snapshot."""
        return self._members

    def get(self, host: str) -> Member:
        """The member named ``host``.

        ``host`` goes through :func:`~kronotop._options.parse_host`, so a name
        without a port finds the member :meth:`add` stored with one, and
        another spelling of the same address finds it too. See
        :func:`~kronotop._options.canonical_host`.

        No hostname lookup here, unlike in :meth:`add`, because ``session()``
        calls this and does no I/O.
        """
        key = member_key(host)
        members = self._members
        for member in members:
            if member.key == key:
                return member
        known = ", ".join(member.name for member in members)
        raise ConfigurationError(f"host {host!r} is not one of: {known}")

    def next(self) -> Member:
        """The member whose turn it is. Each call moves to the next one.

        There is no default member. A command with no host named goes to each
        member in turn.
        """
        with self._lock:
            members = self._members
            turn = self._turn
            self._turn = turn + 1
        return members[turn % len(members)]

    def _turn_of(self, names: Sequence[str]) -> str:
        """The name whose turn it is. The counter :meth:`next` uses."""
        with self._lock:
            turn = self._turn
            self._turn = turn + 1
        return names[turn % len(names)]

    def _by_key(self, key: tuple[str, int]) -> Member | None:
        """The member with this key. No lookup, so this answers on every call."""
        for member in self._members:
            if member.key == key:
                return member
        return None

    def _unresolved(self, key: tuple[str, int]) -> list[str]:
        """The members to look up before ``key`` can fold onto one.

        Empty when the new host got no address of its own, because there is
        nothing to match against. That holds a name server that is down to one
        failed lookup per call. Only members on the same port can match.
        """
        host, port = key
        if host not in self._addresses:
            return []
        return [
            member.key[0]
            for member in self._members
            if member.key[1] == port and member.key[0] not in self._addresses
        ]

    def _sharing_address(self, key: tuple[str, int]) -> Member | None:
        """The member ``key`` resolves onto, from what was looked up already."""
        host, port = key
        mine = self._addresses.get(host, frozenset())
        if not mine:
            return None
        for member in self._members:
            if member.key[1] != port:
                continue
            if self._addresses.get(member.key[0], frozenset()) & mine:
                return member
        return None

    def next_of(self, names: Sequence[str]) -> Member:
        """The member whose turn it is among ``names``.

        ``next`` over a shortlist, on the same turn counter, so the sessions of
        a client that uses both still spread over its members.

        ``names`` comes from a ``LOCATE`` reply and can hold a member this
        client has never talked to, so the pick goes through :meth:`add`. The
        turn is taken first, because ``add`` takes the lock as well.
        """
        return self.add(self._turn_of(names))

    async def anext_of(self, names: Sequence[str]) -> Member:
        """The async twin of :meth:`next_of`."""
        return await self.aadd(self._turn_of(names))

    def add(self, host: str) -> Member:
        """Add ``host``, or return the member it names under any spelling.

        A second pool for one member would double the connections. The key
        folds the spellings of one address, so ``127.0.0.1:5484`` finds
        ``localhost:5484``. A hostname is looked up, so ``10.0.0.5:5484`` finds
        ``db1.internal:5484``. The match runs again under the lock, so two
        threads adding one member build one pool. The lookup stays outside the
        lock, because it blocks.

        A host that answers is looked up once. See :meth:`_keep`.
        """
        key = member_key(host)
        member = self._by_key(key)
        if member is not None:
            return member
        if key[0] not in self._addresses:
            self._keep(key[0], resolve(key[0]))
        for name in self._unresolved(key):
            self._keep(name, resolve(name))
        return self._add(host, key)

    async def aadd(self, host: str) -> Member:
        """The async twin of :meth:`add`.

        ``socket.getaddrinfo`` blocks, so the lookups go through
        :func:`~kronotop._options.aresolve`.
        """
        key = member_key(host)
        member = self._by_key(key)
        if member is not None:
            return member
        if key[0] not in self._addresses:
            self._keep(key[0], await aresolve(key[0]))
        for name in self._unresolved(key):
            self._keep(name, await aresolve(name))
        return self._add(host, key)

    def _keep(self, name: str, found: frozenset[str]) -> None:
        """Remember what a lookup found. An empty answer is not kept.

        A name server can be down for a moment. Keeping that answer would stop
        ``name`` from ever folding onto another member, so the next
        :meth:`add` asks again.
        """
        if found:
            self._addresses[name] = found

    def _add(self, host: str, key: tuple[str, int]) -> Member:
        """Build the member, unless the addresses put ``host`` on one already.

        The last step of :meth:`add` and :meth:`aadd`, once every lookup is in
        ``_addresses``. Nothing here does I/O, so it all runs under the lock.
        """
        name, hostname, port = parse_host(host)
        with self._lock:
            member = self._by_key(key) or self._sharing_address(key)
            if member is not None:
                return member
            pool = clone_pool(self._members[0].pool, hostname, port)
            client = self._factory(connection_pool=pool)
            # redis-py leaves a pool it was handed open on close, because the
            # caller owns it. This one was built here and nothing else holds it.
            client.auto_close_connection_pool = True
            member = _member(name, client)
            self._members += (member,)
            return member

    def remove(self, host: str) -> None:
        """Drop ``host`` from the list. The name is read as :meth:`get` reads it.

        The pool stays open, because a session pinned to one of its connections
        is still running. The sessions hold the last reference, so the pool
        goes when they do. Nothing drains its pending queue after this, so a
        connection a dead session left there stays open until then.

        The last member cannot be removed. A client with no member has nowhere
        to send a command.
        """
        key = member_key(host)
        with self._lock:
            members = self._members
            kept = tuple(member for member in members if member.key != key)
            if len(kept) == len(members):
                known = ", ".join(member.name for member in members)
                raise ConfigurationError(f"host {host!r} is not one of: {known}")
            if not kept:
                raise ConfigurationError(f"host {host!r} is the last member")
            self._members = kept
