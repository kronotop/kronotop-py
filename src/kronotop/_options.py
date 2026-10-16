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

"""The redis-py options Kronotop cannot honour, and the ones it re-defaults.

The client forwards its keyword arguments to redis-py untouched. A few describe
Redis features Kronotop does not implement and would fail at connect time or
later, so they are rejected here. Both clients import this module, so the
rules exist once.

``host``, ``port`` and ``connection_pool`` are rejected for another reason. A
client talks to several members, and ``bootstrap`` names them. The three name
one member, so they would say the same thing twice.

The ``bootstrap`` list is parsed here too, and a URL becomes one entry of it.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from redis.maint_notifications import MaintNotificationsConfig

from kronotop.exceptions import ConfigurationError

__all__ = [
    "DEFAULT_ADMIN_PORT",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_PORT",
    "FORBIDDEN",
    "SCHEMES",
    "apply_defaults",
    "aresolve",
    "build_clients",
    "canonical_host",
    "check_kwargs",
    "clone_pool",
    "format_host",
    "parse_bootstrap",
    "parse_host",
    "resolve",
    "split_url",
    "to_redis_url",
]

#: Kronotop listens on 5484 for client traffic.
DEFAULT_PORT = 5484

#: The port ``KR.ADMIN`` answers on. See :mod:`kronotop.admin`.
DEFAULT_ADMIN_PORT = 3320

#: The member a client starts from when the caller names none.
DEFAULT_BOOTSTRAP: Final[tuple[str, ...]] = (f"localhost:{DEFAULT_PORT}",)

#: The message for a member named outside the client's own argument.
#: ``check_kwargs`` fills in that argument's name.
_NAME_THE_MEMBER: Final = "not supported, name the member in {}"

#: redis-py keyword argument -> the reason it is rejected.
FORBIDDEN: Final[dict[str, str]] = {
    "cache_config": "client side caching is not supported",
    "cache": "client side caching is not supported",
    "connection_pool": _NAME_THE_MEMBER,
    "host": _NAME_THE_MEMBER,
    "port": _NAME_THE_MEMBER,
    "single_connection_client": "not supported, use client.session()",
    "unix_socket_path": "unix sockets are not supported",
}

#: Kronotop URL scheme -> the scheme redis-py parses it as. ``kronotops`` is the
#: TLS form, and matches ``network.external.tls`` on the server.
SCHEMES: Final[dict[str, str]] = {"kronotop": "redis", "kronotops": "rediss"}


def check_kwargs(kwargs: Mapping[str, Any], member_argument: str = "bootstrap") -> None:
    """Reject redis-py options Kronotop cannot honour.

    Checked by value, because every default here is falsy and a caller
    forwarding a prepared keyword dict passes those defaults without asking for
    the feature. ``kronotop://host/0`` parses to ``db=0``, and redis-py sends
    ``SELECT`` only for a non-zero db.

    ``member_argument`` is the argument that names a member on the calling
    client, ``bootstrap`` on ``Kronotop`` and ``host`` on ``Admin``.
    """
    for name, reason in FORBIDDEN.items():
        if kwargs.get(name):
            raise ConfigurationError(f"{name}: {reason.format(member_argument)}")
    db = kwargs.get("db")
    if db:
        raise ConfigurationError(f"db={db!r} is not supported, use a namespace instead")


def apply_defaults(kwargs: dict[str, Any]) -> None:
    """Turn maintenance notifications off, unless the caller asked for them.

    redis-py enables them under RESP3 and reconnects a live connection when the
    server pushes one, which starts a new server side session. Kronotop never
    sends them, so this is a safety net. A fresh config object per call,
    because it is mutable and the pool keeps a reference.
    """
    kwargs.setdefault(
        "maint_notifications_config", MaintNotificationsConfig(enabled=False)
    )


def to_redis_url(url: str) -> str:
    """Turn a ``kronotop://`` URL into the form redis-py parses.

    Everything after the scheme is left alone, so the rest of the URL behaves
    as it does in redis-py.
    """
    scheme, separator, rest = url.partition("://")
    target = SCHEMES.get(scheme.lower())
    if target is None or not separator:
        raise ConfigurationError(
            f"invalid URL {url!r}: expected kronotop://host:{DEFAULT_PORT}, or "
            f"kronotops:// for TLS"
        )
    return f"{target}://{rest}"


def split_url(url: str, parse_url: Callable[[str], Any]) -> tuple[str, dict[str, Any]]:
    """Split a ``kronotop://`` URL into one ``bootstrap`` entry and keyword arguments.

    ``parse_url`` is redis-py's own, so the query string is read the way it is
    everywhere else. Each client passes its own, because they return different
    connection classes.

    That class is the only sign of TLS the URL leaves, and no client takes it
    as a keyword argument, so it becomes ``ssl=True``. Only ``rediss://`` sets
    it, because :func:`to_redis_url` refuses every other scheme.
    """
    # A copy, because the asyncio ``parse_url`` returns a TypedDict.
    parsed: dict[str, Any] = dict(parse_url(to_redis_url(url)))
    host = parsed.pop("host")
    port = parsed.pop("port", DEFAULT_PORT)
    # After the two pops, because ``FORBIDDEN`` rejects both names. The URL path
    # is the other way a db can arrive.
    check_kwargs(parsed)
    if parsed.pop("connection_class", None) is not None:
        parsed["ssl"] = True
    return format_host(host, port), parsed


def _port(text: str, entry: str, label: str) -> int:
    """The port of one entry. The message names the entry it came from."""
    try:
        return int(text)
    except ValueError:
        raise ConfigurationError(f"{label}: {entry!r} has no valid port") from None


def _is_address(text: str) -> bool:
    """Whether the whole of ``text`` is one IP address."""
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


def format_host(host: str, port: int) -> str:
    """Join a host and a port into the name a member is known by.

    An IPv6 address is bracketed, because ``::1`` and ``5484`` would otherwise
    join into ``::1:5484``, an address of its own.
    """
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def parse_host(
    entry: str, label: str = "host", default_port: int = DEFAULT_PORT
) -> tuple[str, str, int]:
    """Split one ``host`` or ``host:port`` entry into name, host and port.

    The name comes from :func:`format_host` and is the key a member is stored
    under. The host is bare, the form ``getaddrinfo`` takes. An entry without a
    port gets ``default_port``, and ``label`` names the argument the entry came
    from for the error message.

    An IPv6 address takes brackets when a port follows it, ``[::1]:5484``, as
    RFC 5952 asks. Written bare it is the whole entry, so ``::1`` is one
    address on the default port. Otherwise the last colon separates the port.
    """
    if entry.startswith("["):
        host, closed, rest = entry[1:].partition("]")
        if not closed:
            raise ConfigurationError(f"{label}: {entry!r} has no closing ]")
        if not rest:
            port = default_port
        elif rest.startswith(":"):
            port = _port(rest[1:], entry, label)
        else:
            raise ConfigurationError(f"{label}: {entry!r} has no valid port")
    elif _is_address(entry):
        host, port = entry, default_port
    else:
        host, separator, port_text = entry.rpartition(":")
        if not separator:
            host, port = entry, default_port
        else:
            port = _port(port_text, entry, label)
    if not host:
        raise ConfigurationError(f"{label}: {entry!r} has no host")
    # A colon left in the host means neither the brackets nor the branch above
    # recognised an address, so it is no address at all.
    if ":" in host and not _is_address(host):
        raise ConfigurationError(f"{label}: {entry!r} has no valid host")
    return format_host(host, port), host, port


#: The addresses that stand for the machine a client runs on. The rest of
#: ``127.0.0.0/8`` is left out, because another member can listen on one of
#: those.
_LOOPBACK: Final = frozenset(
    {ipaddress.IPv4Address("127.0.0.1"), ipaddress.IPv6Address("::1")}
)


def canonical_host(host: str) -> str:
    """One spelling for one host, so two names for one member compare equal.

    A member is known by the name ``bootstrap`` gave it, and a server reports
    the address it has. Without this, a client that took ``localhost:5484``
    would open a second pool for ``127.0.0.1:5484``.

    This folds what needs no name server: the brackets around an IPv6 address,
    its case and zeros, an IPv4-mapped address, and the loopback address.
    ``::1``, ``0:0:0:0:0:0:0:1``, ``127.0.0.1`` and ``localhost`` all land on
    ``localhost``, see :data:`_LOOPBACK`.

    A hostname comes back lowercased. Folding ``db1.internal`` onto
    ``10.0.0.5`` is :func:`resolve`.
    """
    text = host.strip().lower()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return text
    # Before the loopback check, so ``::ffff:127.0.0.1`` lands with the rest.
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if address in _LOOPBACK:
        return "localhost"
    return str(address)


def _canonical_addresses(found: Sequence[Any]) -> frozenset[str]:
    """The addresses of a ``getaddrinfo`` reply, each in canonical form."""
    return frozenset(canonical_host(entry[4][0]) for entry in found)


def resolve(host: str) -> frozenset[str]:
    """Every address ``host`` stands for, from the resolver.

    ``db1.internal`` and ``10.0.0.5`` are the same member only if the resolver
    says so, and :func:`canonical_host` cannot know that. A name that does not
    resolve gives an empty set, which matches nothing, so the caller falls back
    to comparing names.

    The call blocks. Only :meth:`kronotop._members.Members.add` makes it, once
    per host, and keeps the answer.
    """
    try:
        return _canonical_addresses(
            socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        )
    except OSError:
        return frozenset()


async def aresolve(host: str) -> frozenset[str]:
    """The async twin of :func:`resolve`.

    ``socket.getaddrinfo`` would block the event loop, so this uses the loop's
    own, which runs the call in a thread.
    """
    loop = asyncio.get_running_loop()
    try:
        return _canonical_addresses(
            await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        )
    except OSError:
        return frozenset()


def parse_bootstrap(bootstrap: Sequence[str] | None) -> dict[str, tuple[str, int]]:
    """Turn a ``bootstrap`` list into ``"host:port" -> (host, port)``.

    ``None`` gives :data:`DEFAULT_BOOTSTRAP`. An empty list is a caller who
    meant to name one and did not.

    One member named twice is kept once, under the first name, because a second
    pool would double the connections and take two turns out of every round.
    :func:`canonical_host` folds the spellings of one address, so
    ``localhost:5484`` and ``127.0.0.1:5484`` are one entry. A hostname is not
    looked up, because the client does no I/O while it is built.

    The order is kept. It is the order commands are sent in.
    """
    if bootstrap is None:
        bootstrap = DEFAULT_BOOTSTRAP
    if not bootstrap:
        raise ConfigurationError("bootstrap: empty, name at least one member")
    parsed: dict[str, tuple[str, int]] = {}
    seen: set[tuple[str, int]] = set()
    for entry in bootstrap:
        name, host, port = parse_host(entry, "bootstrap")
        key = (canonical_host(host), port)
        if key in seen:
            continue
        seen.add(key)
        parsed[name] = (host, port)
    return parsed


def clone_pool(pool: Any, host: str, port: int) -> Any:
    """A second pool of the same kind, pointing at another member.

    A member added later has to reach the server the same way the others do.
    The connection class carries TLS and ``connection_kwargs`` carries the
    credentials, timeouts and encoding, so both come from a pool that works.

    ``timeout`` and ``queue_class`` belong to ``BlockingConnectionPool`` alone
    and are copied when the source has them. The asyncio pool does not keep its
    ``queue_class``, so a custom one is lost there. The maintenance handler
    belongs to the pool it was built for, so it is dropped.

    :func:`apply_defaults` runs again, because redis-py keeps the maintenance
    config out of ``connection_kwargs`` while it is off, and a clone would come
    back with redis-py's own ``"auto"``.
    """
    kwargs = dict(pool.connection_kwargs, host=host, port=port)
    kwargs.pop("maint_notifications_pool_handler", None)
    apply_defaults(kwargs)
    extra = {
        name: getattr(pool, name)
        for name in ("timeout", "queue_class")
        if getattr(pool, name, None) is not None
    }
    return type(pool)(
        connection_class=pool.connection_class,
        max_connections=pool.max_connections,
        **extra,
        **kwargs,
    )


def build_clients(
    factory: Callable[..., Any],
    bootstrap: Sequence[str] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """One redis-py client per member, keyed by ``host:port``.

    ``factory`` is ``redis.Redis`` or its asyncio twin. Response callbacks are
    the caller's job.
    """
    clients: dict[str, Any] = {}
    for name, (host, port) in parse_bootstrap(bootstrap).items():
        target = dict(kwargs, host=host, port=port)
        # Per member, because the config object is mutable and each pool keeps a
        # reference to it.
        apply_defaults(target)
        clients[name] = factory(**target)
    return clients
