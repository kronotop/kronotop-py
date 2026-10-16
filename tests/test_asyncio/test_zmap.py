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

"""Async twin of ``tests/test_zmap.py``.

The structural tests are not here. They build no connection and await nothing,
so the sync twin runs them against both client kinds.
"""

from __future__ import annotations

import asyncio
import math
import struct
import time
from decimal import Decimal

import pytest
from redis.exceptions import ResponseError

from kronotop import NoSuchNamespaceError
from kronotop.asyncio import Kronotop


@pytest.fixture
async def range_session(session, namespace):
    """See the sync twin."""
    await session.namespace.use(namespace)
    return session


@pytest.fixture
async def pairs(range_session):
    """``key-0`` to ``key-5``, holding ``a`` to ``f``. Returns what to expect."""
    for index, value in enumerate("abcdef"):
        await range_session.zmap.set(f"key-{index}", value)
    return [
        (f"key-{index}".encode(), value.encode())
        for index, value in enumerate("abcdef")
    ]


@pytest.mark.integration
async def test_set_and_get(session, key):
    assert await session.zmap.set(key, "Hello") is True
    assert await session.zmap.get(key) == b"Hello"


@pytest.mark.integration
async def test_get_missing_key_returns_none(session):
    assert await session.zmap.get("kronotop-py:zmap:async:no-such-key") is None


@pytest.mark.integration
async def test_set_overwrites(session, key):
    await session.zmap.set(key, "Hello")
    assert await session.zmap.set(key, "World") is True
    assert await session.zmap.get(key) == b"World"


@pytest.mark.integration
async def test_delete(session, key):
    await session.zmap.set(key, "Hello")
    assert await session.zmap.delete(key) is True
    assert await session.zmap.get(key) is None


@pytest.mark.integration
async def test_delete_is_idempotent(session):
    assert await session.zmap.delete("kronotop-py:zmap:async:no-such-key") is True


@pytest.mark.integration
async def test_binary_key_and_value(session):
    key = b"kronotop-py:zmap:async:\x00binary"
    value = b"\x00\x01\xff"
    await session.zmap.set(key, value)
    assert await session.zmap.get(key) == value
    await session.zmap.delete(key)


@pytest.mark.integration
async def test_commands_join_the_open_transaction(session, key):
    async with session.transaction():
        await session.zmap.set(key, "staged")
        # The write is visible to the transaction that made it.
        assert await session.zmap.get(key) == b"staged"
    assert await session.zmap.get(key) == b"staged"


@pytest.mark.integration
async def test_rollback_drops_the_write(session, key):
    await session.zmap.set(key, "before")
    with pytest.raises(RuntimeError):
        async with session.transaction():
            await session.zmap.set(key, "after")
            raise RuntimeError("boom")
    assert await session.zmap.get(key) == b"before"


@pytest.mark.integration
async def test_uncommitted_write_is_invisible_to_another_session(kr, key):
    async with kr.session() as other:
        await other.begin()
        await other.zmap.set(key, "uncommitted")
        async with kr.session() as watcher:
            assert await watcher.zmap.get(key) is None
        await other.rollback()


@pytest.mark.integration
async def test_snapshot_read(session, key):
    await session.zmap.set(key, "Hello")
    await session.snapshot_read(True)
    try:
        async with session.transaction():
            assert await session.zmap.get(key) == b"Hello"
    finally:
        await session.snapshot_read(False)


@pytest.mark.integration
async def test_get_range_includes_both_ends(range_session, pairs):
    assert await range_session.zmap.get_range("key-0", "key-5") == pairs


@pytest.mark.integration
async def test_get_range_on_an_empty_range(range_session):
    assert await range_session.zmap.get_range("a", "b") == []


@pytest.mark.integration
async def test_get_range_limit(range_session, pairs):
    assert await range_session.zmap.get_range("key-0", "key-5", limit=3) == pairs[:3]


@pytest.mark.integration
async def test_get_range_reverse(range_session, pairs):
    result = await range_session.zmap.get_range("key-0", "key-5", limit=3, reverse=True)
    assert result == list(reversed(pairs))[:3]


@pytest.mark.integration
async def test_get_range_without_bounds(range_session, pairs):
    assert await range_session.zmap.get_range(None, None) == pairs
    assert await range_session.zmap.get_range("key-2", None) == pairs[2:]


@pytest.mark.integration
async def test_get_range_begin_key_selector_drops_the_begin_key(range_session, pairs):
    result = await range_session.zmap.get_range(
        "key-0", "key-5", begin_key_selector="first_greater_than"
    )
    assert result == pairs[1:]


@pytest.mark.integration
async def test_get_range_end_key_selector_drops_the_end_key(range_session, pairs):
    result = await range_session.zmap.get_range(
        "key-0", "key-5", end_key_selector="first_greater_or_equal"
    )
    assert result == pairs[:-1]


@pytest.mark.integration
async def test_get_range_takes_every_option_at_once(range_session, pairs):
    result = await range_session.zmap.get_range(
        "key-0",
        "key-5",
        limit=3,
        reverse=True,
        begin_key_selector="first_greater_or_equal",
        end_key_selector="first_greater_than",
    )
    assert result == list(reversed(pairs))[:3]


@pytest.mark.integration
async def test_get_range_sees_the_open_transaction(range_session, pairs):
    async with range_session.transaction():
        await range_session.zmap.set("key-6", "g")
        result = await range_session.zmap.get_range(None, None)
        assert result[-1] == (b"key-6", b"g")


@pytest.mark.integration
async def test_delete_range_keeps_the_end_key(range_session, pairs):
    assert await range_session.zmap.delete_range("key-0", "key-5") is True
    assert await range_session.zmap.get_range(None, None) == pairs[-1:]


@pytest.mark.integration
async def test_delete_range_on_an_empty_range(range_session):
    assert await range_session.zmap.delete_range("a", "b") is True


@pytest.mark.integration
async def test_delete_range_without_bounds_clears_the_namespace(range_session, pairs):
    assert await range_session.zmap.delete_range(None, None) is True
    assert await range_session.zmap.get_range(None, None) == []


@pytest.mark.integration
async def test_rollback_drops_the_range_delete(range_session, pairs):
    with pytest.raises(RuntimeError):
        async with range_session.transaction():
            await range_session.zmap.delete_range(None, None)
            raise RuntimeError("boom")
    assert await range_session.zmap.get_range(None, None) == pairs


@pytest.mark.integration
async def test_get_key_resolves_to_itself(range_session, pairs):
    assert await range_session.zmap.get_key("key-0") == b"key-0"


@pytest.mark.integration
async def test_get_key_on_a_missing_key_gives_the_next_one(range_session, pairs):
    # "key-00" sorts between "key-0" and "key-1".
    assert await range_session.zmap.get_key("key-00") == b"key-1"


@pytest.mark.integration
async def test_get_key_first_greater_than(range_session, pairs):
    result = await range_session.zmap.get_key(
        "key-0", key_selector="first_greater_than"
    )
    assert result == b"key-1"


@pytest.mark.integration
async def test_get_key_last_less_than(range_session, pairs):
    result = await range_session.zmap.get_key("key-1", key_selector="last_less_than")
    assert result == b"key-0"


@pytest.mark.integration
async def test_get_key_last_less_or_equal(range_session, pairs):
    result = await range_session.zmap.get_key(
        "key-1", key_selector="last_less_or_equal"
    )
    assert result == b"key-1"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("reference", "selector"),
    [("key-5", "first_greater_than"), ("key-0", "last_less_than")],
)
async def test_get_key_off_the_end_of_the_namespace(
    range_session, pairs, reference, selector
):
    """The key it lands on is outside the namespace, so the server cannot unpack it."""
    with pytest.raises(ResponseError, match="not contained in subspace"):
        await range_session.zmap.get_key(reference, key_selector=selector)


@pytest.mark.integration
async def test_get_key_sees_the_open_transaction(range_session, pairs):
    async with range_session.transaction():
        await range_session.zmap.set("key-6", "g")
        result = await range_session.zmap.get_key(
            "key-5", key_selector="first_greater_than"
        )
        assert result == b"key-6"


@pytest.mark.integration
async def test_get_range_size(range_session, pairs):
    # FoundationDB samples the keyspace and the sample is empty at this size, so
    # the answer is 0. Only the type and the sign are worth pinning.
    size = await range_session.zmap.get_range_size("key-0", "key-5")
    assert isinstance(size, int)
    assert size >= 0


@pytest.mark.integration
async def test_get_range_size_without_bounds(range_session, pairs):
    size = await range_session.zmap.get_range_size(None, None)
    assert isinstance(size, int)
    assert size >= 0


@pytest.mark.integration
async def test_set_i64_and_get_i64(session, key):
    assert await session.zmap.set_i64(key, 42) is True
    assert await session.zmap.get_i64(key) == 42


@pytest.mark.integration
async def test_set_i64_a_negative_value(session, key):
    await session.zmap.set_i64(key, -500)
    assert await session.zmap.get_i64(key) == -500


@pytest.mark.integration
@pytest.mark.parametrize("value", [2**63 - 1, -(2**63)])
async def test_set_i64_the_ends_of_the_range(session, key, value):
    await session.zmap.set_i64(key, value)
    assert await session.zmap.get_i64(key) == value


@pytest.mark.integration
async def test_set_i64_stores_eight_little_endian_bytes(session, key):
    """``get`` and ``get_i64`` are two views of one key."""
    await session.zmap.set_i64(key, 42)
    assert await session.zmap.get(key) == struct.pack("<q", 42)


@pytest.mark.integration
async def test_get_i64_missing_key_returns_none(session):
    assert await session.zmap.get_i64("kronotop-py:zmap:async:no-such-key") is None


@pytest.mark.integration
async def test_get_i64_on_a_value_that_is_not_eight_bytes(session, key):
    await session.zmap.set(key, "abc")
    with pytest.raises(ResponseError, match="expected 8-byte"):
        await session.zmap.get_i64(key)


@pytest.mark.integration
async def test_inc_i64_starts_at_zero(session, key):
    assert await session.zmap.inc_i64(key, 10) is True
    assert await session.zmap.get_i64(key) == 10


@pytest.mark.integration
async def test_inc_i64_accumulates(session, key):
    await session.zmap.inc_i64(key, 10)
    await session.zmap.inc_i64(key, 20)
    assert await session.zmap.get_i64(key) == 30


@pytest.mark.integration
async def test_inc_i64_with_a_negative_value(session, key):
    await session.zmap.set_i64(key, 100)
    await session.zmap.inc_i64(key, -30)
    assert await session.zmap.get_i64(key) == 70


@pytest.mark.integration
async def test_inc_i64_wraps_on_overflow(session, key):
    await session.zmap.set_i64(key, 2**63 - 1)
    await session.zmap.inc_i64(key, 1)
    assert await session.zmap.get_i64(key) == -(2**63)


@pytest.mark.integration
async def test_rollback_drops_the_increment(session, key):
    await session.zmap.set_i64(key, 5)
    with pytest.raises(RuntimeError):
        async with session.transaction():
            await session.zmap.inc_i64(key, 100)
            raise RuntimeError("boom")
    assert await session.zmap.get_i64(key) == 5


@pytest.mark.integration
async def test_set_f64_and_get_f64(session, key):
    assert await session.zmap.set_f64(key, 19.99) is True
    assert await session.zmap.get_f64(key) == 19.99


@pytest.mark.integration
async def test_set_f64_a_negative_value(session, key):
    await session.zmap.set_f64(key, -0.5)
    assert await session.zmap.get_f64(key) == -0.5


@pytest.mark.integration
async def test_set_f64_normalises_negative_zero(session, key):
    await session.zmap.set_f64(key, -0.0)
    # -0.0 == 0.0 is True, so only the sign bit tells them apart.
    assert math.copysign(1.0, await session.zmap.get_f64(key)) == 1.0


@pytest.mark.integration
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_set_f64_rejects_values_that_are_not_finite(session, key, value):
    """See the sync twin."""
    with pytest.raises(ResponseError):
        await session.zmap.set_f64(key, value)


@pytest.mark.integration
async def test_set_f64_rejects_the_servers_spelling_of_nan(session, key):
    """``NaN`` parses, so this reaches the finite guard the other test misses."""
    with pytest.raises(ResponseError, match="must be a finite"):
        await session.execute_command("ZSET.F64", key, "NaN")


@pytest.mark.integration
async def test_set_f64_stores_eight_little_endian_bytes(session, key):
    """``get`` and ``get_f64`` are two views of one key."""
    await session.zmap.set_f64(key, 19.99)
    assert await session.zmap.get(key) == struct.pack("<d", 19.99)


@pytest.mark.integration
async def test_get_f64_missing_key_returns_none(session):
    assert await session.zmap.get_f64("kronotop-py:zmap:async:no-such-key") is None


@pytest.mark.integration
async def test_get_f64_on_a_value_that_is_not_eight_bytes(session, key):
    await session.zmap.set(key, "abc")
    with pytest.raises(ResponseError, match="expected 8-byte"):
        await session.zmap.get_f64(key)


@pytest.mark.integration
async def test_inc_f64_starts_at_zero(session, key):
    assert await session.zmap.inc_f64(key, 10.5) is True
    assert await session.zmap.get_f64(key) == 10.5


@pytest.mark.integration
async def test_inc_f64_accumulates(session, key):
    await session.zmap.inc_f64(key, 1.5)
    await session.zmap.inc_f64(key, 3.0)
    assert await session.zmap.get_f64(key) == 4.5


@pytest.mark.integration
async def test_inc_f64_with_a_negative_value(session, key):
    await session.zmap.set_f64(key, 100.0)
    await session.zmap.inc_f64(key, -30.5)
    assert await session.zmap.get_f64(key) == 69.5


@pytest.mark.integration
async def test_inc_f64_rejects_an_overflow(session, key):
    """Unlike ``inc_i64``, which wraps. The result would be ``inf``."""
    await session.zmap.set_f64(key, 1.7976931348623157e308)
    with pytest.raises(ResponseError, match="not a finite"):
        await session.zmap.inc_f64(key, 1.7976931348623157e308)


@pytest.mark.integration
async def test_rollback_drops_the_float_increment(session, key):
    await session.zmap.set_f64(key, 5.5)
    with pytest.raises(RuntimeError):
        async with session.transaction():
            await session.zmap.inc_f64(key, 100.0)
            raise RuntimeError("boom")
    assert await session.zmap.get_f64(key) == 5.5


@pytest.mark.integration
async def test_set_d128_and_get_d128(session, key):
    assert await session.zmap.set_d128(key, Decimal("19.99")) is True
    assert await session.zmap.get_d128(key) == Decimal("19.99")


@pytest.mark.integration
@pytest.mark.parametrize("value", ["19.99", 7, Decimal("-0.5")])
async def test_set_d128_takes_a_string_an_int_and_a_decimal(session, key, value):
    await session.zmap.set_d128(key, value)
    assert await session.zmap.get_d128(key) == Decimal(str(value))


@pytest.mark.integration
async def test_set_d128_keeps_the_trailing_zero(session, key):
    """``Decimal("1.5") == Decimal("1.50")``, so compare the text."""
    await session.zmap.set_d128(key, Decimal("1.50"))
    assert str(await session.zmap.get_d128(key)) == "1.50"


@pytest.mark.integration
async def test_set_d128_keeps_thirty_four_digits(session, key):
    value = "3.141592653589793238462643383279502"
    await session.zmap.set_d128(key, value)
    assert str(await session.zmap.get_d128(key)) == value


@pytest.mark.integration
async def test_get_d128_always_answers_in_plain_form(session, key):
    """Scientific notation goes in, plain decimal comes back."""
    await session.zmap.set_d128(key, "1.5E+10")
    assert await session.zmap.get_d128(key) == Decimal("15000000000")


@pytest.mark.integration
async def test_set_d128_stores_sixteen_bytes(session, key):
    await session.zmap.set_d128(key, "19.99")
    assert len(await session.zmap.get(key)) == 16


@pytest.mark.integration
async def test_get_d128_missing_key_returns_none(session):
    assert await session.zmap.get_d128("kronotop-py:zmap:async:no-such-key") is None


@pytest.mark.integration
async def test_set_d128_rejects_a_string_that_is_not_a_decimal(session, key):
    with pytest.raises(ResponseError, match="invalid decimal"):
        await session.zmap.set_d128(key, "notanumber")


@pytest.mark.integration
async def test_set_d128_rejects_a_value_out_of_range(session, key):
    with pytest.raises(ResponseError, match="Exponent is out of range"):
        await session.zmap.set_d128(key, "1E+7000")


@pytest.mark.integration
async def test_get_d128_on_a_value_that_is_not_sixteen_bytes(session, key):
    await session.zmap.set(key, "abc")
    with pytest.raises(ResponseError, match="expected 16-byte"):
        await session.zmap.get_d128(key)


@pytest.mark.integration
async def test_inc_d128_starts_at_zero(session, key):
    assert await session.zmap.inc_d128(key, "10.5") is True
    assert await session.zmap.get_d128(key) == Decimal("10.5")


@pytest.mark.integration
async def test_inc_d128_is_exact(session, key):
    """0.1 plus 0.2 is 0.3 here, which ``inc_f64`` cannot say."""
    await session.zmap.inc_d128(key, Decimal("0.1"))
    await session.zmap.inc_d128(key, Decimal("0.2"))
    assert await session.zmap.get_d128(key) == Decimal("0.3")


@pytest.mark.integration
async def test_inc_d128_with_a_negative_value(session, key):
    await session.zmap.set_d128(key, "100.0")
    await session.zmap.inc_d128(key, "-30.5")
    assert await session.zmap.get_d128(key) == Decimal("69.5")


@pytest.mark.integration
async def test_rollback_drops_the_decimal_increment(session, key):
    await session.zmap.set_d128(key, "5.5")
    with pytest.raises(RuntimeError):
        async with session.transaction():
            await session.zmap.inc_d128(key, "100")
            raise RuntimeError("boom")
    assert await session.zmap.get_d128(key) == Decimal("5.5")


@pytest.mark.integration
async def test_mutate_add(session, key):
    await session.zmap.set_i64(key, 1)
    assert await session.zmap.mutate(key, struct.pack("<q", 5), "add") is True
    assert await session.zmap.get_i64(key) == 6


@pytest.mark.integration
async def test_mutate_add_on_a_missing_key_stores_the_operand(session, key):
    await session.zmap.mutate(key, struct.pack("<q", 5), "add")
    assert await session.zmap.get_i64(key) == 5


@pytest.mark.integration
async def test_mutate_truncates_to_the_length_of_the_operand(session, key):
    """A short operand shrinks the value, with no error."""
    await session.zmap.set_i64(key, 1)
    await session.zmap.mutate(key, b"\x05", "add")
    assert await session.zmap.get(key) == b"\x06"


@pytest.mark.integration
async def test_mutate_compare_and_clear_deletes_on_a_match(session, key):
    await session.zmap.set(key, "my-value")
    await session.zmap.mutate(key, b"my-value", "compare_and_clear")
    assert await session.zmap.get(key) is None


@pytest.mark.integration
async def test_mutate_compare_and_clear_keeps_a_value_that_differs(session, key):
    await session.zmap.set(key, "my-value")
    # ``OK`` all the same, so the reply says nothing about what happened.
    assert await session.zmap.mutate(key, b"other", "compare_and_clear") is True
    assert await session.zmap.get(key) == b"my-value"


@pytest.mark.integration
async def test_mutate_bit_or(session, key):
    await session.zmap.set(key, b"\x0f\x00")
    await session.zmap.mutate(key, b"\xf0\x0f", "bit_or")
    assert await session.zmap.get(key) == b"\xff\x0f"


@pytest.mark.integration
async def test_mutate_append_if_fits(session, key):
    await session.zmap.set(key, "abc")
    await session.zmap.mutate(key, b"def", "append_if_fits")
    assert await session.zmap.get(key) == b"abcdef"


@pytest.mark.integration
async def test_mutate_max_and_byte_max_disagree(session, key):
    """``max`` reads the pair as integers, ``byte_max`` as bytes."""
    await session.zmap.set(key, b"\x00\x01")
    await session.zmap.mutate(key, b"\x01\x00", "max")
    # Little-endian, so the stored value is 256 and the operand is 1.
    assert await session.zmap.get(key) == b"\x00\x01"

    await session.zmap.set(key, b"\x00\x01")
    await session.zmap.mutate(key, b"\x01\x00", "byte_max")
    assert await session.zmap.get(key) == b"\x01\x00"


@pytest.mark.integration
async def test_mutate_byte_max_keeps_the_length(session, key):
    """The byte-wise mutations neither pad nor cut, unlike ``add`` and ``max``."""
    await session.zmap.set(key, b"\x00\x01\x02\x03")
    await session.zmap.mutate(key, b"\x00\x01", "byte_max")
    assert await session.zmap.get(key) == b"\x00\x01\x02\x03"


@pytest.mark.integration
async def test_mutate_set_versionstamped_value(session, key):
    """The offset is consumed. The 10 bytes it points at get the versionstamp."""
    await session.zmap.mutate(
        key, b"pre" + b"\x00" * 10 + struct.pack("<i", 3), "set_versionstamped_value"
    )
    stored = await session.zmap.get(key)
    assert stored.startswith(b"pre")
    assert len(stored) == 13


@pytest.mark.integration
async def test_a_versionstamped_key_cannot_be_read_before_commit(session, key):
    """The server fills the versionstamp in at commit time."""
    with pytest.raises(ResponseError, match="ACCESSED_UNREADABLE"):
        async with session.transaction():
            await session.zmap.mutate(
                key,
                b"pre" + b"\x00" * 10 + struct.pack("<i", 3),
                "set_versionstamped_value",
            )
            await session.zmap.get(key)


@pytest.mark.integration
async def test_mutate_rejects_an_unknown_mutation_type(session, key):
    """Sent raw: ``MutationType`` keeps typed code away from this."""
    with pytest.raises(ResponseError):
        await session.execute_command("ZMUTATE", key, b"y", "NOPE")


@pytest.mark.integration
async def test_rollback_drops_the_mutation(session, key):
    await session.zmap.set_i64(key, 1)
    with pytest.raises(RuntimeError):
        async with session.transaction():
            await session.zmap.mutate(key, struct.pack("<q", 5), "add")
            raise RuntimeError("boom")
    assert await session.zmap.get_i64(key) == 1


async def watch_while_writing(
    watcher, writer, key, first_write_after=0.0, namespace=None
):
    """See the sync twin. A task instead of a thread."""
    task = asyncio.create_task(watcher.zmap.watch(key, namespace=namespace))
    await asyncio.sleep(first_write_after)
    deadline = time.monotonic() + 10
    for attempt in range(1000):
        if task.done() or time.monotonic() > deadline:
            break
        await writer.zmap.set(key, f"after-{attempt}", namespace=namespace)
        await asyncio.wait({task}, timeout=0.1)
    if not task.done():
        # A cancel ends the session. Only the failing path does it.
        task.cancel()
        raise AssertionError("the watch never fired")
    return await task


@pytest.mark.integration
async def test_watch_wakes_on_a_change(kr, session, key):
    """Two sessions: one blocks on the key, the other writes it."""
    await session.zmap.set(key, "before")
    async with kr.session() as watcher:
        assert await watch_while_writing(watcher, session, key) is True
        # The signal has no value in it, so the new one comes from a read.
        assert await watcher.zmap.get(key) is not None


@pytest.mark.integration
async def test_watch_is_rejected_inside_a_transaction(session, key):
    """The server rejects it. The block still rolls back."""
    with pytest.raises(ResponseError, match="not allowed within a transaction"):
        async with session.transaction():
            await session.zmap.watch(key)
    assert session.in_transaction is False
    assert await session.zmap.get(key) is None


@pytest.mark.integration
async def test_watch_ignores_the_socket_timeout(kronotop_url, session, key):
    """Without the blocking read this dies at 0.5 seconds and kills the session."""
    await session.zmap.set(key, "before")
    async with (
        Kronotop.from_url(kronotop_url, socket_timeout=0.5) as client,
        client.session() as watcher,
    ):
        reply = await watch_while_writing(watcher, session, key, first_write_after=1.0)
        assert reply is True


@pytest.mark.integration
@pytest.mark.parametrize(
    ("write", "read", "value", "expected"),
    [
        ("set", "get", "Hello", b"Hello"),
        ("set_i64", "get_i64", 7, 7),
        ("inc_i64", "get_i64", 7, 7),
        ("set_f64", "get_f64", 1.5, 1.5),
        ("inc_f64", "get_f64", 1.5, 1.5),
        ("set_d128", "get_d128", "1.50", Decimal("1.50")),
        ("inc_d128", "get_d128", "1.50", Decimal("1.50")),
    ],
)
async def test_a_command_runs_in_the_namespace_given(
    session, namespace, key, write, read, value, expected
):
    current = await session.namespace.current()
    written = await getattr(session.zmap, write)(key, value, namespace=namespace)
    assert written is True
    assert await getattr(session.zmap, read)(key, namespace=namespace) == expected
    # The session did not move, and its own namespace does not hold the key.
    assert await session.namespace.current() == current
    assert await session.zmap.get(key) is None


@pytest.mark.integration
async def test_delete_in_the_namespace_given(session, namespace, key):
    await session.zmap.set(key, "here")
    await session.zmap.set(key, "there", namespace=namespace)
    assert await session.zmap.delete(key, namespace=namespace) is True
    assert await session.zmap.get(key, namespace=namespace) is None
    assert await session.zmap.get(key) == b"here"


@pytest.mark.integration
async def test_mutate_in_the_namespace_given(session, namespace, key):
    param = struct.pack("<q", 5)
    assert await session.zmap.mutate(key, param, "add", namespace=namespace) is True
    assert await session.zmap.get_i64(key, namespace=namespace) == 5
    assert await session.zmap.get(key) is None


@pytest.mark.integration
async def test_the_range_commands_run_in_the_namespace_given(session, namespace):
    expected = []
    for index, value in enumerate("abc"):
        await session.zmap.set(f"key-{index}", value, namespace=namespace)
        expected.append((f"key-{index}".encode(), value.encode()))

    zmap = session.zmap
    assert await zmap.get_range(None, None, namespace=namespace) == expected
    found = await zmap.get_key(
        "key-0", key_selector="first_greater_than", namespace=namespace
    )
    assert found == b"key-1"
    assert await zmap.get_range_size(None, None, namespace=namespace) >= 0
    assert await zmap.delete_range(None, None, namespace=namespace) is True
    assert await zmap.get_range(None, None, namespace=namespace) == []


@pytest.mark.integration
async def test_watch_in_the_namespace_given(kr, session, namespace, key):
    await session.zmap.set(key, "before", namespace=namespace)
    async with kr.session() as watcher:
        reply = await watch_while_writing(watcher, session, key, namespace=namespace)
        assert reply is True


@pytest.mark.integration
async def test_a_namespace_that_is_not_there(session, namespaces, key):
    with pytest.raises(NoSuchNamespaceError):
        await session.zmap.get(key, namespace=await namespaces(create=False))
