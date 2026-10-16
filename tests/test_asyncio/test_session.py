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

"""Async twin of ``tests/test_session.py``."""

from __future__ import annotations

import asyncio
import gc
from typing import cast

import pytest
import redis
from redis.asyncio import Connection

from kronotop import (
    KronotopConnectionError,
    NoCommitVersionError,
    NotCommittedError,
    TransactionCancelledError,
    TransactionError,
)
from kronotop.asyncio import Kronotop, Session


async def test_session_does_not_connect():
    s = Kronotop().session()
    assert isinstance(s, Session)
    assert s._redis.connection is None
    assert s.in_transaction is False
    assert s.broken is False


class _DeadConnection:
    """A connection whose transport close fails, as it does after the loop ends."""

    retry = None

    def _close(self):
        raise RuntimeError("Event loop is closed")


def test_finalizer_survives_a_failed_init(only_pool):
    # noinspection PyTypeChecker
    s = Session.__new__(Session)
    s._pool = only_pool(Kronotop())
    s.__del__()


async def test_finalizer_queues_the_connection_when_the_loop_is_gone():
    """``Connection._close`` raises once the loop is closed.

    The queueing must not depend on it, or the pool slot is lost for good.
    """
    s = Kronotop().session()
    conn = _DeadConnection()
    s._redis.connection = cast(Connection, conn)
    with pytest.warns(ResourceWarning, match="unclosed session"):
        s.__del__()
    assert list(s._pending) == [conn]


@pytest.mark.parametrize(
    "command",
    [
        "begin",
        "commit",
        "rollback",
        "tick",
        "ping",
        "echo",
        "time",
        "info",
        "command",
    ],
)
def test_no_command_is_on_the_client(command):
    assert not hasattr(Kronotop(), command)


@pytest.mark.parametrize("command", ["set", "lpush", "xadd", "pipeline", "acl_list"])
async def test_redis_commands_are_not_exposed(command):
    assert not hasattr(Kronotop().session(), command)


@pytest.mark.integration
async def test_transaction_commits(session):
    async with session.transaction():
        assert session.in_transaction is True
    assert session.in_transaction is False


@pytest.mark.integration
async def test_transaction_rolls_back_on_error(session, key):
    await session.execute_command("ZSET", key, "before")
    with pytest.raises(RuntimeError):
        async with session.transaction():
            await session.execute_command("ZSET", key, "after")
            raise RuntimeError("boom")
    assert await session.execute_command("ZGET", key) == b"before"


@pytest.mark.integration
async def test_returning_committed_version(session, key):
    async with session.transaction(returning="committed-version") as tx:
        await session.execute_command("ZSET", key, "1")
    assert tx.committed_version > 0
    assert tx.versionstamp is None


@pytest.mark.integration
async def test_returning_versionstamp(session, key):
    async with session.transaction(returning="versionstamp") as tx:
        await session.execute_command("ZSET", key, "1")
    assert isinstance(tx.versionstamp, bytes)
    assert len(tx.versionstamp) == 10


@pytest.mark.integration
async def test_nested_transaction_is_refused(session):
    async with session.transaction():
        with pytest.raises(TransactionError, match="no savepoints"):
            async with session.transaction():
                pass


@pytest.mark.integration
async def test_commit_without_transaction(session):
    with pytest.raises(TransactionError, match="no transaction in progress"):
        await session.commit()


@pytest.mark.integration
async def test_inspection_inside_a_transaction(session):
    async with session.transaction():
        assert await session.get_read_version() > 0
        assert await session.get_approximate_size() >= 0


@pytest.mark.integration
async def test_sessions_are_isolated(kr):
    async with kr.session() as first, kr.session() as second:
        assert await first.begin() is True
        assert await second.begin() is True
        assert await first.rollback() is True
        assert await second.rollback() is True


@pytest.mark.integration
async def test_async_with_closes_the_session(kr):
    async with kr.session() as s:
        await s.ping()
    assert s.closed is True


@pytest.mark.integration
async def test_close_returns_the_connection_to_the_pool(kr, only_pool):
    pool = only_pool(kr)
    before = len(pool._in_use_connections)
    async with kr.session() as s:
        await s.ping()
        assert len(pool._in_use_connections) == before + 1
    assert len(pool._in_use_connections) == before


@pytest.mark.integration
async def test_snapshot_read_toggles(session):
    assert await session.snapshot_read(True) is True
    assert await session.snapshot_read(False) is True


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["fresh", "cached"])
async def test_tick(session, mode):
    assert await session.tick(mode) > 0


@pytest.mark.integration
async def test_list_transactions(kr, key):
    async with kr.session() as first, kr.session() as second:
        await first.begin()
        await first.zmap.set(key, "v")
        entries = [
            t for t in await second.list_transactions() if t["command"] == "ZSET"
        ]
        await first.rollback()
    assert len(entries) == 1
    assert entries[0]["model"] == "ZMap"
    assert entries[0]["state"] == "idle"
    assert isinstance(entries[0]["watching"], bool)


@pytest.mark.integration
async def test_kill_transaction(kr, key):
    """The owner fails on its next read and has to roll back."""
    async with kr.session() as first, kr.session() as second:
        await first.begin()
        await first.zmap.set(key, "v")
        [entry] = [
            t for t in await second.list_transactions() if t["command"] == "ZSET"
        ]
        assert await second.kill_transaction(entry["id"]) is True
        with pytest.raises(TransactionCancelledError):
            await first.zmap.get(key)
        await first.rollback()


@pytest.mark.integration
async def test_kill_unknown_transaction(session):
    with pytest.raises(TransactionError, match="no such transaction"):
        await session.kill_transaction(2**63 - 1)


@pytest.mark.integration
async def test_slow_transactions(kr, key):
    """Needs the default ``transaction.slow_log.threshold`` of 1000 ms."""
    async with kr.session() as first, kr.session() as second:
        await first.begin()
        await first.zmap.set(key, "v")
        [entry] = [
            t for t in await second.list_transactions() if t["command"] == "ZSET"
        ]
        await asyncio.sleep(1.1)
        await first.commit()
        [slow] = [t for t in await second.slow_transactions() if t["id"] == entry["id"]]
    assert slow["command"] == "ZSET"
    assert slow["committed"] is True


async def conflicting_writes(first, second, key):
    """Both sessions read ``key``, then write it. ``first`` commits and wins."""
    await first.execute_command("ZSET", key, "0")
    await first.begin()
    await second.begin()
    # Both take a read conflict range on the same key.
    await first.execute_command("ZGET", key)
    await second.execute_command("ZGET", key)
    await first.execute_command("ZSET", key, "first")
    await second.execute_command("ZSET", key, "second")
    assert await first.commit() is True


@pytest.mark.integration
async def test_failed_commit_ends_the_transaction(kr, key):
    async with kr.session() as first, kr.session() as second:
        await conflicting_writes(first, second, key)
        with pytest.raises(NotCommittedError):
            await second.commit()
        assert second.in_transaction is False

        async with second.transaction():
            await second.execute_command("ZSET", key, "retried")
        assert await second.execute_command("ZGET", key) == b"retried"


@pytest.mark.integration
async def test_failed_commit_returning_ends_the_transaction(session):
    await session.begin()
    with pytest.raises(NoCommitVersionError):
        await session.commit("versionstamp")
    assert session.in_transaction is False
    with pytest.raises(TransactionError, match="no transaction in progress"):
        await session.rollback()


@pytest.mark.integration
async def test_a_timeout_breaks_the_session(kr, timeout_once):
    async with kr.session() as s:
        await s.begin()
        conn = s._redis.connection
        conn.read_response = timeout_once(conn)
        with pytest.raises(redis.TimeoutError):
            await s.ping()

        assert conn.is_connected is False
        assert s.broken is True
        assert s.in_transaction is False
        with pytest.raises(KronotopConnectionError, match="connection was lost"):
            await s.ping()


@pytest.mark.integration
async def test_a_cancelled_command_breaks_the_session(kr):
    """redis-py's asyncio connection disconnects on any ``BaseException``.

    ``asyncio.CancelledError`` is one, and it is not an ``Exception``. The stub
    does what ``redis/asyncio/connection.py`` does on a cancelled read.
    """
    async with kr.session() as s:
        await s.begin()
        conn = s._redis.connection

        async def cancel_once(*_args, **_kwargs):
            await conn.disconnect(nowait=True)
            raise asyncio.CancelledError

        conn.read_response = cancel_once
        with pytest.raises(asyncio.CancelledError):
            await s.ping()

        assert s.broken is True
        assert s.in_transaction is False
        with pytest.raises(KronotopConnectionError, match="connection was lost"):
            await s.ping()


@pytest.mark.integration
async def test_a_dropped_socket_breaks_the_session(kr):
    async with kr.session() as s:
        await s.begin()
        await s._redis.connection.disconnect()
        with pytest.raises(KronotopConnectionError, match="connection was lost"):
            await s.ping()
        assert s.broken is True
        assert s.in_transaction is False


async def test_close_is_idempotent():
    s = Kronotop().session()
    await s.close()
    await s.close()
    assert s.closed is True


async def test_commands_after_close_raise():
    s = Kronotop().session()
    await s.close()
    with pytest.raises(Exception, match="session is closed"):
        await s.ping()


@pytest.mark.integration
async def test_close_rolls_back_an_open_transaction(kr, key):
    async with kr.session() as s:
        await s.begin()
        await s.execute_command("ZSET", key, "uncommitted")
    async with kr.session() as s:
        value = await s.execute_command("ZGET", key)
    assert value != b"uncommitted"


@pytest.mark.integration
async def test_retry_is_disabled_on_the_pinned_connection(kr):
    async with kr.session() as s:
        await s.ping()
        assert s._redis.connection.retry.get_retries() == 0


@pytest.mark.integration
async def test_read_version_without_transaction(session):
    with pytest.raises(TransactionError, match="no transaction in progress"):
        await session.get_read_version()


@pytest.mark.integration
async def test_returning_committed_version_without_mutations(session):
    async with session.transaction(returning="committed-version") as tx:
        pass
    assert tx.committed_version == -1


@pytest.mark.integration
async def test_versionstamp_without_mutations(session):
    await session.begin()
    with pytest.raises(NoCommitVersionError, match="read-only"):
        await session.commit("versionstamp")


@pytest.mark.integration
async def test_tick_never_goes_backwards(session):
    first = await session.tick("fresh")
    assert await session.tick("fresh") >= first


@pytest.mark.integration
async def test_conflicting_transactions(kr, key):
    async with kr.session() as first, kr.session() as second:
        await conflicting_writes(first, second, key)
        with pytest.raises(NotCommittedError, match="conflict"):
            await second.commit()


@pytest.mark.integration
async def test_rollback_failure_does_not_mask_the_original_error(kr, timeout_once):
    async with kr.session() as s:
        await s.ping()
        conn = s._redis.connection
        stub = timeout_once(conn)
        with pytest.raises(redis.TimeoutError):
            async with s.transaction():
                conn.read_response = stub
                await s.ping()


@pytest.mark.integration
async def test_unclosed_session_warns(kr):
    s = kr.session()
    await s.ping()
    with pytest.warns(ResourceWarning, match="unclosed session"):
        del s
        gc.collect()


@pytest.mark.integration
async def test_unclosed_session_gives_the_pool_slot_back(kr, kronotop_url):
    """More sessions than the pool holds, none of them closed.

    ``pool.release`` is a coroutine, so the finalizer queues the connection and
    the next session hands it back. Without that this raises
    ``MaxConnectionsError`` on the third pass. See the sync twin for ``kr``.
    """
    client = Kronotop.from_url(kronotop_url, max_connections=2)
    try:
        for _ in range(4):
            s = client.session()
            await s.ping()
            del s
            gc.collect()
    finally:
        await client.aclose()


@pytest.mark.integration
async def test_ping(session):
    assert await session.ping() is True


@pytest.mark.integration
async def test_ping_with_message(session):
    assert await session.ping("Hello, World!") == b"Hello, World!"


@pytest.mark.integration
async def test_echo(session):
    assert await session.echo("Hello, World!") == b"Hello, World!"


@pytest.mark.integration
async def test_time(session):
    seconds, microseconds = await session.time()
    assert seconds > 0
    assert 0 <= microseconds < 1_000_000


@pytest.mark.integration
async def test_info(session):
    reply = await session.info()
    assert reply["server_name"] == "kronotop"
    assert reply["server_mode"] == "standalone"
    assert reply["cluster_enabled"] == "0"
    assert isinstance(reply["kronotop_version"], str)


@pytest.mark.integration
async def test_info_with_a_section(session):
    reply = await session.info("kronotop")
    assert "cluster_name" in reply
    assert "cluster_enabled" not in reply


@pytest.mark.integration
async def test_info_with_an_unknown_section(session):
    assert await session.info("nosuch") == {}
