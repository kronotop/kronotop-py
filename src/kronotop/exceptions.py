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

"""Exception hierarchy for kronotop-py.

Every exception derives from :class:`KronotopError` and a matching ``redis``
exception, so code that catches ``redis.RedisError`` keeps working.
"""

from __future__ import annotations

from redis.exceptions import ConnectionError as _RedisConnectionError
from redis.exceptions import RedisError, ResponseError

__all__ = [
    "BarrierNotSatisfiedError",
    "BucketAlreadyExistsError",
    "BucketBeingRemovedError",
    "ConfigurationError",
    "DuplicateKeyError",
    "IndexTypeMismatchError",
    "KronotopCommandError",
    "KronotopConnectionError",
    "KronotopError",
    "NamespaceAlreadyExistsError",
    "NamespaceBeingRemovedError",
    "NoCommitVersionError",
    "NoCommonMemberError",
    "NoSuchBucketError",
    "NoSuchIndexError",
    "NoSuchNamespaceError",
    "NotCommittedError",
    "RejectError",
    "TransactionError",
    "TransactionTooOldError",
    "VectorIndexNotReadyError",
    "wrap_response_error",
]


class KronotopError(RedisError):
    """Base class for all Kronotop-specific errors."""


class KronotopConnectionError(KronotopError, _RedisConnectionError):
    """Raised when the connection to a Kronotop server fails."""


class KronotopCommandError(KronotopError, ResponseError):
    """Raised when a Kronotop server replies with an error to a command."""


class ConfigurationError(KronotopError, ValueError):
    """Raised for a client side configuration mistake, before any command goes out.

    A bad :class:`kronotop.KronotopConfig` field, or a client keyword argument
    Kronotop cannot honour. Also a ``ValueError``, because that is what a
    caller passing a bad value expects to catch.
    """


class NoCommonMemberError(KronotopError):
    """Raised when no member holds every bucket named in ``session_for``.

    A session is one connection to one member, and a bucket command runs on a
    member that owns one of the bucket's shards. Buckets with no such member
    in common cannot share a session or a transaction.

    No command failed. The client read where each bucket lives and stopped
    before it opened the session. Create the buckets on the same shards, or
    use one session per member.
    """


class TransactionError(KronotopCommandError):
    """Raised for server errors carrying the ``TRANSACTION`` prefix.

    Kronotop uses that prefix when a transaction is already open, or when a
    command needs an open transaction and there is none.
    """


class NoCommitVersionError(KronotopCommandError):
    """Raised for server errors carrying the ``NO_COMMIT_VERSION`` prefix.

    A FoundationDB error. ``commit("versionstamp")`` needs at least one
    mutation, and this transaction wrote nothing. ``commit("committed-version")``
    returns ``-1`` in that case.
    """


class NotCommittedError(KronotopCommandError):
    """Raised for server errors carrying the ``NOT_COMMITTED`` prefix.

    FoundationDB error 1020. The transaction conflicted with another one that
    wrote the same keys, so the commit was rejected and nothing was applied.
    Retry the whole transaction. The old one is dead, so retrying the commit
    alone does nothing.
    """


class TransactionTooOldError(KronotopCommandError):
    """Raised for server errors carrying the ``TRANSACTION_TOO_OLD`` prefix.

    FoundationDB error 1007. A transaction has about five seconds from the
    moment it takes its read version. That clock starts at the first read or at
    :meth:`~kronotop.commands.transactions.TransactionCommands.get_read_version`,
    not at ``BEGIN``. A transaction that only writes takes its read version at
    commit, so it does not hit this.

    Retry the transaction, or split the work into smaller ones.
    """


class TransactionCancelledError(KronotopCommandError):
    """Raised for server errors carrying the ``TRANSACTION_CANCELLED`` prefix.

    FoundationDB error 1025. Another session cancelled the transaction with
    ``TRANSACTION KILL``. Reads and the commit raise this, a write alone does
    not. The transaction stays open until ``rollback``.
    """


class NoSuchNamespaceError(KronotopCommandError):
    """Raised for server errors carrying the ``NOSUCHNAMESPACE`` prefix.

    ``NAMESPACE USE`` on a namespace that does not exist, or a command that
    names one with ``namespace``. Create it first with ``NAMESPACE CREATE``.
    """


class NamespaceAlreadyExistsError(KronotopCommandError):
    """Raised for server errors carrying the ``NAMESPACEALREADYEXISTS`` prefix.

    ``NAMESPACE CREATE`` on a name that is taken, or ``NAMESPACE MOVE`` onto
    one.
    """


class NamespaceBeingRemovedError(KronotopCommandError):
    """Raised for server errors carrying the ``NAMESPACEBEINGREMOVED`` prefix.

    ``NAMESPACE REMOVE`` marked the namespace and ``NAMESPACE PURGE`` has not
    run yet. In between, ``use``, ``create``, ``exists``, ``move``, a second
    ``remove`` and a command that names it with ``namespace`` all raise this.
    """


class NoSuchBucketError(KronotopCommandError):
    """Raised for server errors carrying the ``NOSUCHBUCKET`` prefix.

    The bucket is not in the namespace the command ran in, the current one or
    the one given with ``namespace``. ``BUCKET.CREATE`` makes it.
    """


class NoSuchIndexError(KronotopCommandError):
    """Raised for server errors carrying the ``NOSUCHINDEX`` prefix.

    ``BUCKET.INDEX DESCRIBE``, ``DROP`` or ``ANALYZE`` named an index the bucket
    does not have. ``BUCKET.INDEX LIST`` gives the names.
    """


class BucketAlreadyExistsError(KronotopCommandError):
    """Raised for server errors carrying the ``BUCKETALREADYEXISTS`` prefix.

    ``BUCKET.CREATE`` on a name that is taken. A bucket that ``BUCKET.REMOVE``
    marked still holds its name, so it raises this too, until ``BUCKET.PURGE``
    runs. Pass ``if_not_exists=True`` to get ``OK`` instead.
    """


class BucketBeingRemovedError(KronotopCommandError):
    """Raised for server errors carrying the ``BUCKETBEINGREMOVED`` prefix.

    ``BUCKET.REMOVE`` marked the bucket and ``BUCKET.PURGE`` has not run yet.
    In between, every command on the bucket raises this.
    """


class DuplicateKeyError(KronotopCommandError):
    """Raised for server errors carrying the ``DUPLICATEKEY`` prefix.

    ``BUCKET.INSERT`` with an ``_id`` that the bucket already holds. Only a
    document that carries its own ``_id`` can hit this. Nothing was written.
    """


class IndexTypeMismatchError(KronotopCommandError):
    """Raised for server errors carrying the ``INDEXTYPE_MISMATCH`` prefix.

    An indexed field holds a value of a type the index was not built for.
    Kronotop only checks this when ``strict_types`` is on for the index.
    """


class VectorIndexNotReadyError(KronotopCommandError):
    """Raised for server errors carrying the ``VECTORINDEXNOTREADY`` prefix.

    A vector index on the bucket is still bootstrapping. Wait a moment and send
    the command again.
    """


class BarrierNotSatisfiedError(KronotopCommandError):
    """Raised for server errors carrying the ``BARRIERNOTSATISFIED`` prefix.

    ``NAMESPACE PURGE`` and ``BUCKET.PURGE`` need every live member to have
    seen the removal, and Kronotop waits about five seconds for that. One
    member had not. Send the command again, one retry is usually enough,
    because the failed attempt publishes the removal event once more.
    """


def _split_address(text: str) -> tuple[str, int] | None:
    """Split one ``host:port`` address into its two parts, or ``None``.

    An IPv6 host arrives bracketed, ``[::1]:5484``, as RFC 5952 asks. The
    brackets come off.
    """
    host, separator, port = text.rpartition(":")
    if not separator:
        return None
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return host, int(port)
    except ValueError:
        return None


def _parse_reject(message: str) -> tuple[int | None, tuple[str, ...]]:
    """Split ``REJECT <shard_id> <address>, <address>`` into shard and addresses.

    The addresses come back as the server wrote them, in the form
    :func:`~kronotop._options.format_host` uses, so each one is a name
    ``bootstrap`` takes.

    The shard is ``None`` and the addresses are empty when the message has
    another shape. One unreadable address fails the whole message, because a
    half read list would point at the wrong member. Nothing raises here, or
    the real error is lost.
    """
    parts = message.split(" ", 2)
    if len(parts) != 3:
        return None, ()
    addresses = tuple(entry.strip() for entry in parts[2].split(","))
    if any(_split_address(address) is None for address in addresses):
        return None, ()
    try:
        return int(parts[1]), addresses
    except ValueError:
        return None, ()


class RejectError(KronotopCommandError):
    """Raised for server errors carrying the ``REJECT`` prefix.

    A bucket command reached a member that owns none of the bucket's shards.
    The message names the shard and every client address of the owner:
    ``REJECT <shard_id> <host>:<port>, <host>:<port>``.

    The client does not follow the redirect, because a session is pinned to
    one connection and the open transaction lives on it. Send the command on a
    session to :attr:`addresses`, or open the session with ``session_for``,
    which picks the member from the buckets.

    The transaction stays open, and the commands before this one still commit.
    """

    #: Shard the bucket's data lives on. ``None`` if the message was not parsed.
    shard_id: int | None
    #: Every ``host:port`` of the owner, in the order the server sent them.
    #: Pass one to ``bootstrap``. Empty if the message was not parsed.
    addresses: tuple[str, ...]
    #: Host of the first address, which is the preferred one.
    host: str | None
    #: Port of that address.
    port: int | None

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.shard_id, self.addresses = _parse_reject(message)
        first = _split_address(self.addresses[0]) if self.addresses else None
        self.host, self.port = first if first is not None else (None, None)


#: Server error prefix, the first word of the reply, -> exception class. Add a
#: prefix here when code needs to catch it on its own.
_ERROR_CLASSES: dict[str, type[ResponseError]] = {
    "TRANSACTION": TransactionError,
    "NO_COMMIT_VERSION": NoCommitVersionError,
    "NOT_COMMITTED": NotCommittedError,
    "TRANSACTION_TOO_OLD": TransactionTooOldError,
    "TRANSACTION_CANCELLED": TransactionCancelledError,
    "NOSUCHNAMESPACE": NoSuchNamespaceError,
    "NAMESPACEALREADYEXISTS": NamespaceAlreadyExistsError,
    "NAMESPACEBEINGREMOVED": NamespaceBeingRemovedError,
    "NOSUCHBUCKET": NoSuchBucketError,
    "NOSUCHINDEX": NoSuchIndexError,
    "BUCKETALREADYEXISTS": BucketAlreadyExistsError,
    "BUCKETBEINGREMOVED": BucketBeingRemovedError,
    "BARRIERNOTSATISFIED": BarrierNotSatisfiedError,
    "DUPLICATEKEY": DuplicateKeyError,
    "INDEXTYPE_MISMATCH": IndexTypeMismatchError,
    "VECTORINDEXNOTREADY": VectorIndexNotReadyError,
    "REJECT": RejectError,
}


def wrap_response_error(exc: ResponseError) -> ResponseError:
    """Return the Kronotop-specific exception for ``exc``, or ``exc`` itself.

    An unmapped prefix stays a plain ``ResponseError``, with the prefix kept in
    the message.
    """
    # redis-py strips only the prefixes it knows, so ours is still the first
    # word. ``BaseParser.EXCEPTION_CLASSES`` is shared by every redis-py client
    # in the process, so the mapping lives here instead.
    prefix = str(exc).split(" ", 1)[0]
    error_class = _ERROR_CLASSES.get(prefix)
    if error_class is None:
        return exc
    return error_class(str(exc))
