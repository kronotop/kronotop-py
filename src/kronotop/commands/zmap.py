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

"""ZMap commands, reached through ``session.zmap``.

ZMap is a RESP view of FoundationDB's ordered key-value API. Keys and values
are raw bytes in lexicographic order. Every entry belongs to the session's
current namespace, and namespace and transaction both live in the session, so
this family hangs off :class:`kronotop.session.Session`.

Every command takes ``namespace``, which runs that one command in the
namespace given and leaves the session's namespace alone. The namespace has to
exist, or the command raises :class:`~kronotop.exceptions.NoSuchNamespaceError`.
"""

from __future__ import annotations

from collections.abc import Awaitable
from decimal import Decimal
from typing import Any, Literal, overload

from redis.typing import EncodableT, KeyT

from kronotop.commands.base import (
    AsyncClient,
    CommandGroup,
    CommandsBase,
    SyncClient,
    namespace_args,
)

__all__ = ["AsyncZMap", "KeySelector", "MutationType", "ZMap", "ZMapCommands"]

#: How a range command resolves a boundary against the keys that are there.
KeySelector = Literal[
    "first_greater_or_equal",
    "first_greater_than",
    "last_less_than",
    "last_less_or_equal",
]

#: The atomic mutations FoundationDB offers through ``ZMUTATE``.
MutationType = Literal[
    "add",
    "bit_and",
    "bit_or",
    "bit_xor",
    "append_if_fits",
    "max",
    "min",
    "set_versionstamped_value",
    "byte_min",
    "byte_max",
    "compare_and_clear",
]

#: The unbounded boundary on the wire.
_ASTERISK = "*"


def _bound(key: KeyT | None) -> KeyT:
    """``None`` is the Python spelling of the server's ``*`` boundary."""
    return _ASTERISK if key is None else key


class ZMapCommands(CommandsBase):
    """The eighteen ZMap commands of the server."""

    @overload
    def set(
        self: SyncClient,
        key: KeyT,
        value: EncodableT,
        *,
        namespace: str | None = None,
    ) -> bool: ...

    @overload
    def set(
        self: AsyncClient,
        key: KeyT,
        value: EncodableT,
        *,
        namespace: str | None = None,
    ) -> Awaitable[bool]: ...

    def set(self, key: KeyT, value: EncodableT, *, namespace: str | None = None) -> Any:
        """``ZSET``. Set ``key`` to ``value``, overwriting without an error."""
        return self.execute_command("ZSET", key, value, *namespace_args(namespace))

    @overload
    def get(
        self: SyncClient, key: KeyT, *, namespace: str | None = None
    ) -> bytes | str | None: ...

    @overload
    def get(
        self: AsyncClient, key: KeyT, *, namespace: str | None = None
    ) -> Awaitable[bytes | str | None]: ...

    def get(self, key: KeyT, *, namespace: str | None = None) -> Any:
        """``ZGET``. The value of ``key``, or ``None`` when it is not there.

        The value is ``bytes`` unless the client was built with
        ``decode_responses=True``.
        """
        return self.execute_command("ZGET", key, *namespace_args(namespace))

    @overload
    def delete(
        self: SyncClient, key: KeyT, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def delete(
        self: AsyncClient, key: KeyT, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def delete(self, key: KeyT, *, namespace: str | None = None) -> Any:
        """``ZDEL``. Delete ``key``.

        A key that is not there also returns ``True``. Named ``delete``
        because ``del`` is a Python keyword.
        """
        return self.execute_command("ZDEL", key, *namespace_args(namespace))

    @overload
    def get_range(
        self: SyncClient,
        begin: KeyT | None,
        end: KeyT | None,
        *,
        limit: int | None = None,
        reverse: bool = False,
        begin_key_selector: KeySelector | None = None,
        end_key_selector: KeySelector | None = None,
        namespace: str | None = None,
    ) -> list[tuple[bytes | str, bytes | str]]: ...

    @overload
    def get_range(
        self: AsyncClient,
        begin: KeyT | None,
        end: KeyT | None,
        *,
        limit: int | None = None,
        reverse: bool = False,
        begin_key_selector: KeySelector | None = None,
        end_key_selector: KeySelector | None = None,
        namespace: str | None = None,
    ) -> Awaitable[list[tuple[bytes | str, bytes | str]]]: ...

    def get_range(
        self,
        begin: KeyT | None,
        end: KeyT | None,
        *,
        limit: int | None = None,
        reverse: bool = False,
        begin_key_selector: KeySelector | None = None,
        end_key_selector: KeySelector | None = None,
        namespace: str | None = None,
    ) -> Any:
        """``ZGETRANGE``. Read an ordered range as ``(key, value)`` tuples.

        Both halves of a tuple are ``bytes`` unless the client was built with
        ``decode_responses=True``. A range with no keys gives an empty list.

        Both ends are inclusive by default, while ``delete_range`` is
        half-open, so the same pair of keys covers one key more here. ``None``
        as a boundary means unbounded, so ``get_range(None, None)`` reads the
        whole namespace. It travels as ``*``, so a key named ``*`` cannot be a
        boundary.

        Without ``limit`` the server returns at most 100 pairs. ``reverse``
        walks in descending key order, so ``limit`` then keeps the last pairs.

        The selectors move the boundaries. ``begin_key_selector`` defaults to
        ``"first_greater_or_equal"`` and ``end_key_selector`` to
        ``"first_greater_than"``, which makes both ends inclusive.
        ``"first_greater_than"`` on the begin side drops the begin key,
        ``"first_greater_or_equal"`` on the end side drops the end key.
        """
        args: list[EncodableT] = [_bound(begin), _bound(end)]
        if limit is not None:
            args += ["LIMIT", limit]
        if reverse:
            args.append("REVERSE")
        if begin_key_selector is not None:
            args += ["BEGIN-KEY-SELECTOR", begin_key_selector]
        if end_key_selector is not None:
            args += ["END-KEY-SELECTOR", end_key_selector]
        args += namespace_args(namespace)
        return self.execute_command("ZGETRANGE", *args)

    @overload
    def delete_range(
        self: SyncClient,
        begin: KeyT | None,
        end: KeyT | None,
        *,
        namespace: str | None = None,
    ) -> bool: ...

    @overload
    def delete_range(
        self: AsyncClient,
        begin: KeyT | None,
        end: KeyT | None,
        *,
        namespace: str | None = None,
    ) -> Awaitable[bool]: ...

    def delete_range(
        self, begin: KeyT | None, end: KeyT | None, *, namespace: str | None = None
    ) -> Any:
        """``ZDELRANGE``. Delete every key in ``[begin, end)``.

        Half-open, like FoundationDB's range clear, so a key equal to ``end``
        survives and ``get_range`` with the same two keys reads one key more.
        ``None`` as a boundary means unbounded, so ``delete_range(None, None)``
        clears the namespace.

        A range with no keys also returns ``True``.
        """
        return self.execute_command(
            "ZDELRANGE", _bound(begin), _bound(end), *namespace_args(namespace)
        )

    @overload
    def get_key(
        self: SyncClient,
        key: KeyT,
        *,
        key_selector: KeySelector | None = None,
        namespace: str | None = None,
    ) -> bytes | str | None: ...

    @overload
    def get_key(
        self: AsyncClient,
        key: KeyT,
        *,
        key_selector: KeySelector | None = None,
        namespace: str | None = None,
    ) -> Awaitable[bytes | str | None]: ...

    def get_key(
        self,
        key: KeyT,
        *,
        key_selector: KeySelector | None = None,
        namespace: str | None = None,
    ) -> Any:
        """``ZGETKEY``. Resolve ``key`` against the keys that are there.

        Gives back a key name, so it works as a cursor: ask for the key after
        or before this one without knowing the names. The name is ``bytes``
        unless the client was built with ``decode_responses=True``.
        ``key_selector`` defaults to ``"first_greater_or_equal"``, so a
        reference key that exists resolves to itself.

        A selector that walks off either end of the namespace raises
        ``redis.ResponseError`` with ``Cannot unpack key that is not contained
        in subspace.``, so ``"last_less_than"`` on the first key is an error.
        """
        args: list[EncodableT] = [key]
        if key_selector is not None:
            args += ["KEY-SELECTOR", key_selector]
        args += namespace_args(namespace)
        return self.execute_command("ZGETKEY", *args)

    @overload
    def get_range_size(
        self: SyncClient,
        begin: KeyT | None,
        end: KeyT | None,
        *,
        namespace: str | None = None,
    ) -> int: ...

    @overload
    def get_range_size(
        self: AsyncClient,
        begin: KeyT | None,
        end: KeyT | None,
        *,
        namespace: str | None = None,
    ) -> Awaitable[int]: ...

    def get_range_size(
        self, begin: KeyT | None, end: KeyT | None, *, namespace: str | None = None
    ) -> Any:
        """``ZGETRANGESIZE``. Estimate ``[begin, end)`` in bytes.

        A rough estimate for capacity planning. FoundationDB samples the
        keyspace, and the sample stays empty until the data grows, so a small
        range answers ``0``. Half-open, like ``delete_range``, and ``None`` as
        a boundary means unbounded.
        """
        return self.execute_command(
            "ZGETRANGESIZE", _bound(begin), _bound(end), *namespace_args(namespace)
        )

    @overload
    def set_i64(
        self: SyncClient, key: KeyT, value: int, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def set_i64(
        self: AsyncClient, key: KeyT, value: int, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def set_i64(self, key: KeyT, value: int, *, namespace: str | None = None) -> Any:
        """``ZSET.I64``. Set ``key`` to a signed 64-bit integer.

        Stored as 8 bytes, little-endian two's complement, so ``get`` on the
        same key reads ``struct.pack("<q", value)``. Overwrites without an
        error, like ``set``.
        """
        return self.execute_command("ZSET.I64", key, value, *namespace_args(namespace))

    @overload
    def inc_i64(
        self: SyncClient, key: KeyT, value: int, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def inc_i64(
        self: AsyncClient, key: KeyT, value: int, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def inc_i64(self, key: KeyT, value: int, *, namespace: str | None = None) -> Any:
        """``ZINC.I64``. Add ``value`` to ``key``. Negative to subtract.

        FoundationDB's ``ADD`` mutation, so it writes without reading first
        and two sessions incrementing the same key never conflict. A key that
        is not there starts at zero. Overflow wraps, so one more than
        ``2**63 - 1`` is ``-(2**63)``.

        The reply is ``OK``. Read the new value with ``get_i64``.
        """
        return self.execute_command("ZINC.I64", key, value, *namespace_args(namespace))

    @overload
    def get_i64(
        self: SyncClient, key: KeyT, *, namespace: str | None = None
    ) -> int | None: ...

    @overload
    def get_i64(
        self: AsyncClient, key: KeyT, *, namespace: str | None = None
    ) -> Awaitable[int | None]: ...

    def get_i64(self, key: KeyT, *, namespace: str | None = None) -> Any:
        """``ZGET.I64``. Read ``key`` as a signed 64-bit integer.

        ``None`` when the key is not there. A value that is not exactly 8
        bytes, such as one written with ``set``, raises ``redis.ResponseError``.
        """
        return self.execute_command("ZGET.I64", key, *namespace_args(namespace))

    @overload
    def set_f64(
        self: SyncClient, key: KeyT, value: float, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def set_f64(
        self: AsyncClient, key: KeyT, value: float, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def set_f64(self, key: KeyT, value: float, *, namespace: str | None = None) -> Any:
        """``ZSET.F64``. Set ``key`` to a double.

        Stored as 8 bytes, little-endian IEEE 754, so ``get`` on the same key
        reads ``struct.pack("<d", value)``. Overwrites without an error, like
        ``set``. ``-0.0`` is stored as ``0.0``, and NaN and infinity are
        refused with ``redis.ResponseError``.
        """
        return self.execute_command("ZSET.F64", key, value, *namespace_args(namespace))

    @overload
    def inc_f64(
        self: SyncClient, key: KeyT, value: float, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def inc_f64(
        self: AsyncClient, key: KeyT, value: float, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def inc_f64(self, key: KeyT, value: float, *, namespace: str | None = None) -> Any:
        """``ZINC.F64``. Add ``value`` to ``key``. Negative to subtract.

        FoundationDB has no atomic add for a double, so the server reads the
        key before it writes. Two sessions incrementing the same key can
        conflict, unlike with ``inc_i64``. A key that is not there starts at
        zero, and a result that is not finite raises.

        The reply is ``OK``. Read the new value with ``get_f64``.
        """
        return self.execute_command("ZINC.F64", key, value, *namespace_args(namespace))

    @overload
    def get_f64(
        self: SyncClient, key: KeyT, *, namespace: str | None = None
    ) -> float | None: ...

    @overload
    def get_f64(
        self: AsyncClient, key: KeyT, *, namespace: str | None = None
    ) -> Awaitable[float | None]: ...

    def get_f64(self, key: KeyT, *, namespace: str | None = None) -> Any:
        """``ZGET.F64``. Read ``key`` as a double.

        ``None`` when the key is not there. A value that is not exactly 8
        bytes, such as one written with ``set``, raises ``redis.ResponseError``.

        A ``float`` on either protocol. RESP2 has no double type, so there the
        server sends 8 big-endian bytes and the client unpacks them. A client
        built with both ``protocol=2`` and ``decode_responses=True`` cannot
        read the value, because the raw bytes are not valid UTF-8 and the
        failed decode leaves the connection out of step.
        """
        return self.execute_command("ZGET.F64", key, *namespace_args(namespace))

    @overload
    def set_d128(
        self: SyncClient,
        key: KeyT,
        value: Decimal | int | str,
        *,
        namespace: str | None = None,
    ) -> bool: ...

    @overload
    def set_d128(
        self: AsyncClient,
        key: KeyT,
        value: Decimal | int | str,
        *,
        namespace: str | None = None,
    ) -> Awaitable[bool]: ...

    def set_d128(
        self, key: KeyT, value: Decimal | int | str, *, namespace: str | None = None
    ) -> Any:
        """``ZSET.D128``. Set ``key`` to a 128-bit decimal.

        Takes a :class:`decimal.Decimal`, an ``int`` or a decimal string, and
        sends ``str(value)``, so scientific notation goes through. Stored as
        16 bytes, little-endian IEEE 754 BID, with 34 significant digits and
        decimal arithmetic, which ``set_f64`` does not give.

        A string that is not a decimal, or a value outside the Decimal128
        range, raises ``redis.ResponseError``.
        """
        return self.execute_command(
            "ZSET.D128", key, str(value), *namespace_args(namespace)
        )

    @overload
    def inc_d128(
        self: SyncClient,
        key: KeyT,
        value: Decimal | int | str,
        *,
        namespace: str | None = None,
    ) -> bool: ...

    @overload
    def inc_d128(
        self: AsyncClient,
        key: KeyT,
        value: Decimal | int | str,
        *,
        namespace: str | None = None,
    ) -> Awaitable[bool]: ...

    def inc_d128(
        self, key: KeyT, value: Decimal | int | str, *, namespace: str | None = None
    ) -> Any:
        """``ZINC.D128``. Add ``value`` to ``key``. Negative to subtract.

        Like ``inc_f64``, the server reads the key before it writes, so two
        sessions incrementing the same key can conflict. A key that is not
        there starts at zero, and a result outside the Decimal128 range raises.

        The reply is ``OK``. Read the new value with ``get_d128``.
        """
        return self.execute_command(
            "ZINC.D128", key, str(value), *namespace_args(namespace)
        )

    @overload
    def get_d128(
        self: SyncClient, key: KeyT, *, namespace: str | None = None
    ) -> Decimal | None: ...

    @overload
    def get_d128(
        self: AsyncClient, key: KeyT, *, namespace: str | None = None
    ) -> Awaitable[Decimal | None]: ...

    def get_d128(self, key: KeyT, *, namespace: str | None = None) -> Any:
        """``ZGET.D128``. Read ``key`` as a :class:`decimal.Decimal`.

        The server answers plain decimal text, so a value written as
        ``1.5E+10`` reads back as ``Decimal("15000000000")``. Trailing zeros
        survive, so ``1.50`` in gives ``Decimal("1.50")`` out.

        ``None`` when the key is not there. A value that is not exactly 16
        bytes raises ``redis.ResponseError``.
        """
        return self.execute_command("ZGET.D128", key, *namespace_args(namespace))

    @overload
    def mutate(
        self: SyncClient,
        key: KeyT,
        param: bytes,
        mutation_type: MutationType,
        *,
        namespace: str | None = None,
    ) -> bool: ...

    @overload
    def mutate(
        self: AsyncClient,
        key: KeyT,
        param: bytes,
        mutation_type: MutationType,
        *,
        namespace: str | None = None,
    ) -> Awaitable[bool]: ...

    def mutate(
        self,
        key: KeyT,
        param: bytes,
        mutation_type: MutationType,
        *,
        namespace: str | None = None,
    ) -> Any:
        """``ZMUTATE``. Apply one of FoundationDB's atomic mutations to ``key``.

        The server writes without reading first, so two sessions mutating the
        same key never conflict. ``param`` is raw bytes, packed by the caller,
        and means something different in every mutation. For ``"add"`` it is
        a little-endian integer, so a counter needs ``struct.pack("<q", n)``,
        which is what ``inc_i64`` does.

        A key that is not there is created holding ``param``, whatever the
        mutation. Only ``"compare_and_clear"`` leaves it missing.

        ``"add"``, ``"bit_and"``, ``"bit_or"``, ``"bit_xor"``, ``"max"`` and
        ``"min"`` first match the value to the length of ``param``. A shorter
        ``param`` truncates the value with no error, so ``"add"`` of
        ``b"\x05"`` turns an 8 byte counter into a 1 byte one. A longer
        ``param`` zero-extends the value. ``"byte_max"``, ``"byte_min"``,
        ``"append_if_fits"`` and ``"compare_and_clear"`` neither pad nor cut.

        ``"max"`` and ``"min"`` compare the two as little-endian integers,
        ``"byte_max"`` and ``"byte_min"`` as byte strings, which gives a
        different answer for the same pair. ``"compare_and_clear"`` deletes
        the key when the value equals ``param``. ``"append_if_fits"`` appends
        while the result stays under the 100,000 byte value limit.

        ``"set_versionstamped_value"`` wants ``param`` to hold 10 empty bytes
        where the versionstamp goes, then a 4 byte little-endian offset that
        says where they start. The offset is consumed. The server only knows
        the versionstamp at commit time, so reading the key before ``COMMIT``
        raises ``ACCESSED_UNREADABLE``.

        The reply is ``OK``, which only means the mutation was accepted.
        ``"compare_and_clear"`` on a value that differs and
        ``"append_if_fits"`` past the limit both answer ``OK`` and change
        nothing.
        """
        return self.execute_command(
            "ZMUTATE", key, param, mutation_type, *namespace_args(namespace)
        )

    @overload
    def watch(self: SyncClient, key: KeyT, *, namespace: str | None = None) -> bool: ...

    @overload
    def watch(
        self: AsyncClient, key: KeyT, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def watch(self, key: KeyT, *, namespace: str | None = None) -> Any:
        """``ZWATCH``. Wait until the value at ``key`` changes.

        Blocks the session until the watch fires. The server has no timeout
        or cancel for it: the only way to stop a watch is to close its session
        from another thread or task. Use a second session to keep working.
        The server rejects a watch after ``BEGIN``.

        The signal carries no value, so read the key with ``get`` after each
        one. The usual shape is a loop of ``watch``, ``get``, ``watch``.

        The watch follows the value. Writing the same content back may not
        wake it, and deleting a key that is already gone does not. Two changes
        before the signal make one wake. Changes between one signal and the
        next ``watch`` are lost.

        It can end with an error instead of a signal, for example when
        FoundationDB's limit on open watches per database is full. Call
        ``watch`` again after an error. The read has no timeout, so a
        ``socket_timeout`` on the client does not end a watch.
        """
        return self.execute_blocking_command("ZWATCH", key, *namespace_args(namespace))


class ZMap(CommandGroup, ZMapCommands):
    """``session.zmap`` on the synchronous session."""

    _is_async_client: Literal[False] = False


class AsyncZMap(CommandGroup, ZMapCommands):
    """``session.zmap`` on the asyncio session."""

    _is_async_client: Literal[True] = True
