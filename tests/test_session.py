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

from __future__ import annotations

import gc
import logging
import time
from typing import cast

import pytest
import redis
from redis import Connection
from redis.exceptions import ResponseError

from kronotop import (
    BarrierNotSatisfiedError,
    Kronotop,
    KronotopConnectionError,
    NamespaceAlreadyExistsError,
    NamespaceBeingRemovedError,
    NoCommitVersionError,
    NotCommittedError,
    RejectError,
    Session,
    TransactionCancelledError,
    TransactionError,
    TransactionTooOldError,
)
from kronotop._callbacks import parse_transaction
from kronotop._cleanup import ignore_cleanup_error
from kronotop.exceptions import wrap_response_error


def test_session_does_not_connect():
    """``session()`` must stay free of I/O, so it works with no server up."""
    s = Kronotop().session()
    assert isinstance(s, Session)
    assert s._redis.connection is None
    assert s.in_transaction is False
    assert s.closed is False
    assert s.broken is False


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
    """The client hands the connection back, so a command belongs to a session."""
    assert not hasattr(Kronotop(), command)


@pytest.mark.parametrize("command", ["set", "lpush", "xadd", "pipeline", "acl_list"])
def test_redis_commands_are_not_exposed(command):
    assert not hasattr(Kronotop().session(), command)


def test_close_is_idempotent():
    s = Kronotop().session()
    s.close()
    s.close()
    assert s.closed is True


def test_commands_after_close_raise():
    s = Kronotop().session()
    s.close()
    with pytest.raises(Exception, match="session is closed"):
        s.ping()


class _StubConnection:
    """Enough of a redis-py connection for the finalizer to work on."""

    def __init__(self):
        self.retry = None
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


def test_finalizer_survives_a_failed_init(only_pool):
    """``__init__`` can raise before ``_closed`` is assigned.

    ``__del__`` is called by hand: CPython swallows what a finalizer raises.
    """
    # noinspection PyTypeChecker
    s = Session.__new__(Session)
    s._pool = only_pool(Kronotop())
    s.__del__()


def test_finalizer_does_not_release_to_the_pool():
    """A finalizer must not take the locks ``pool.release`` takes.

    It queues the connection for the next session instead. See ``kronotop._pending``.
    """
    s = Kronotop().session()
    conn = _StubConnection()
    s._redis.connection = cast(Connection, conn)
    with pytest.warns(ResourceWarning, match="unclosed session"):
        s.__del__()
    assert conn.disconnected is True
    assert list(s._pending) == [conn]


@pytest.mark.integration
def test_transaction_commits(session):
    with session.transaction():
        assert session.in_transaction is True
    assert session.in_transaction is False


@pytest.mark.integration
def test_transaction_rolls_back_on_error(session, key):
    session.execute_command("ZSET", key, "before")
    with pytest.raises(RuntimeError), session.transaction():
        session.execute_command("ZSET", key, "after")
        raise RuntimeError("boom")
    assert session.execute_command("ZGET", key) == b"before"
    assert session.in_transaction is False


@pytest.mark.integration
def test_returning_committed_version(session, key):
    # The write goes out raw, because the commit is the subject here.
    with session.transaction(returning="committed-version") as tx:
        session.execute_command("ZSET", key, "1")
    assert tx.committed_version > 0
    assert tx.versionstamp is None


@pytest.mark.integration
def test_returning_committed_version_without_mutations(session):
    with session.transaction(returning="committed-version") as tx:
        pass
    assert tx.committed_version == -1


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("TRANSACTION there is no transaction in progress.", TransactionError),
        ("NO_COMMIT_VERSION Transaction is read-only", NoCommitVersionError),
        (
            "NOT_COMMITTED Transaction not committed due to conflict with "
            "another transaction",
            NotCommittedError,
        ),
        (
            "TRANSACTION_TOO_OLD Transaction is too old to perform reads or be "
            "committed",
            TransactionTooOldError,
        ),
        (
            "TRANSACTION_CANCELLED Operation aborted because the transaction "
            "was cancelled",
            TransactionCancelledError,
        ),
        (
            "NAMESPACEALREADYEXISTS Namespace already exists: app.prod",
            NamespaceAlreadyExistsError,
        ),
        (
            "NAMESPACEBEINGREMOVED Namespace 'app.prod' is being removed",
            NamespaceBeingRemovedError,
        ),
        (
            "BARRIERNOTSATISFIED Barrier not satisfied: not all members observed "
            "version 4",
            BarrierNotSatisfiedError,
        ),
        ("REJECT 4 127.0.0.1:5484", RejectError),
        # Starts with TRANSACTION but is a different prefix. The lookup is on
        # the whole first word, so it must not match TransactionError.
        ("TRANSACTION_TOO_LARGE Transaction exceeds byte limit", ResponseError),
        ("ERR something else", ResponseError),
    ],
)
def test_error_prefixes_map_to_classes(message, expected):
    mapped = wrap_response_error(ResponseError(message))
    assert type(mapped) is expected
    assert str(mapped) == message


@pytest.mark.parametrize(
    ("message", "called"),
    [("REJECT 4 127.0.0.1:5484", True), ("ERR something else", False)],
)
def test_on_reject_runs_for_a_reject_error_only(only_pool, message, called):
    calls = []
    s = Session(only_pool(Kronotop()), on_reject=lambda: calls.append(message))
    with pytest.raises(ResponseError), s._tracking("BUCKET.INSERT", ()):
        raise ResponseError(message)
    assert calls == ([message] if called else [])


def test_reject_error_carries_the_target():
    mapped = wrap_response_error(ResponseError("REJECT 4 127.0.0.1:5484"))
    assert (mapped.shard_id, mapped.host, mapped.port) == (4, "127.0.0.1", 5484)
    assert mapped.addresses == ("127.0.0.1:5484",)


def test_reject_error_carries_every_address_of_the_member():
    """A member advertises more than one, and the first is the preferred one."""
    mapped = wrap_response_error(
        ResponseError("REJECT 4 10.0.0.5:5484, 192.168.1.5:5484")
    )
    assert mapped.addresses == ("10.0.0.5:5484", "192.168.1.5:5484")
    assert (mapped.host, mapped.port) == ("10.0.0.5", 5484)


def test_reject_error_keeps_an_ipv6_host():
    """The brackets mark the address, so they come off with the port."""
    mapped = wrap_response_error(ResponseError("REJECT 4 [::1]:5484"))
    assert (mapped.host, mapped.port) == ("::1", 5484)
    assert mapped.addresses == ("[::1]:5484",)


@pytest.mark.parametrize(
    "message",
    [
        "REJECT",
        "REJECT 4",
        "REJECT 4 node1",
        "REJECT four node1:5484",
        "REJECT 4 node1:http",
        "REJECT 4 node1:5484 extra",
        # One address it cannot read fails the whole message.
        "REJECT 4 node1:5484, node2",
        "REJECT 4 node1:5484, node2:http",
    ],
)
def test_reject_error_with_a_message_it_cannot_read(message):
    """Every field goes to ``None``. Building the exception must not raise."""
    mapped = wrap_response_error(ResponseError(message))
    assert type(mapped) is RejectError
    assert (mapped.shard_id, mapped.host, mapped.port) == (None, None, None)
    assert mapped.addresses == ()


def conflicting_writes(first, second, key):
    """Both sessions read ``key``, then write it. ``first`` commits and wins."""
    first.execute_command("ZSET", key, "0")
    first.begin()
    second.begin()
    # Both take a read conflict range on the same key.
    first.execute_command("ZGET", key)
    second.execute_command("ZGET", key)
    first.execute_command("ZSET", key, "first")
    second.execute_command("ZSET", key, "second")
    assert first.commit() is True


@pytest.mark.integration
def test_conflicting_transactions(kr, key):
    """Two transactions writing a key they both read: the second one loses."""
    with kr.session() as first, kr.session() as second:
        conflicting_writes(first, second, key)
        with pytest.raises(NotCommittedError, match="conflict"):
            second.commit()


@pytest.mark.integration
def test_versionstamp_without_mutations(session):
    """FoundationDB has no commit version for a read-only transaction."""
    session.begin()
    with pytest.raises(NoCommitVersionError, match="read-only"):
        session.commit("versionstamp")


@pytest.mark.integration
def test_returning_versionstamp(session, key):
    with session.transaction(returning="versionstamp") as tx:
        session.execute_command("ZSET", key, "1")
    assert isinstance(tx.versionstamp, bytes)
    assert len(tx.versionstamp) == 10
    assert tx.committed_version is None


@pytest.mark.integration
def test_nested_transaction_is_refused_without_a_round_trip(session):
    # The nesting is the subject of the test, so the blocks stay apart.
    with session.transaction():  # noqa: SIM117
        with pytest.raises(TransactionError, match="no savepoints"):
            with session.transaction():
                pass


@pytest.mark.integration
def test_commit_without_transaction(session):
    with pytest.raises(TransactionError, match="no transaction in progress"):
        session.commit()


@pytest.mark.integration
def test_read_version_without_transaction(session):
    with pytest.raises(TransactionError, match="no transaction in progress"):
        session.get_read_version()


@pytest.mark.integration
def test_inspection_inside_a_transaction(session):
    with session.transaction():
        assert session.get_read_version() > 0
        assert session.get_approximate_size() >= 0


@pytest.mark.integration
def test_sessions_are_isolated(kr, key):
    """Two sessions each get their own connection, so each gets its own transaction."""
    with kr.session() as first, kr.session() as second:
        assert first.begin() is True
        assert second.begin() is True
        first.execute_command("ZSET", key, "first")
        assert second.execute_command("ZGET", key) != b"first"
        assert first.rollback() is True
        assert second.rollback() is True


@pytest.mark.integration
def test_close_returns_the_connection_to_the_pool(kr, only_pool):
    pool = only_pool(kr)
    before = len(pool._in_use_connections)
    with kr.session() as s:
        s.ping()
        assert len(pool._in_use_connections) == before + 1
    assert len(pool._in_use_connections) == before


@pytest.mark.integration
def test_close_rolls_back_an_open_transaction(kr, key):
    with kr.session() as s:
        s.begin()
        s.execute_command("ZSET", key, "uncommitted")
    with kr.session() as s:
        assert s.execute_command("ZGET", key) != b"uncommitted"


@pytest.mark.integration
def test_snapshot_read_toggles(session):
    assert session.snapshot_read(True) is True
    assert session.snapshot_read(False) is True


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["fresh", "cached"])
def test_tick(session, mode):
    assert session.tick(mode) > 0


@pytest.mark.integration
def test_tick_never_goes_backwards(session):
    """The only promise TICK makes. The same value twice is allowed."""
    first = session.tick("fresh")
    assert session.tick("fresh") >= first


#: One ``TRANSACTION LIST`` entry as RESP2 hands it over: a background transaction.
TRANSACTION_RESP2 = [
    b"id", 42, b"duration_ms", 1203, b"idle_ms", 0, b"client_id", -1,
    b"state", b"active", b"namespace", None, b"bucket", None, b"model", None,
    b"command", None, b"query", None, b"watching", 0, b"reads", 3,
    b"range_reads", 1,
    b"writes", 0, b"deletes", 0, b"range_deletes", 0, b"mutations", 0,
    b"bytes_read", 512, b"bytes_written", 0,
]  # fmt: skip


def test_parse_transaction():
    assert parse_transaction([TRANSACTION_RESP2]) == [
        {
            "id": 42,
            "duration_ms": 1203,
            "idle_ms": 0,
            "client_id": -1,
            "state": "active",
            "namespace": None,
            "bucket": None,
            "model": None,
            "command": None,
            "query": None,
            "watching": False,
            "reads": 3,
            "range_reads": 1,
            "writes": 0,
            "deletes": 0,
            "range_deletes": 0,
            "mutations": 0,
            "bytes_read": 512,
            "bytes_written": 0,
        }
    ]
    assert parse_transaction([]) == []
    assert parse_transaction(b"OK") is True


#: One ``TRANSACTION SLOW`` entry as RESP2 hands it over: an internal one.
SLOW_TRANSACTION_RESP2 = [
    b"id", 42, b"timestamp", 1791964800000, b"duration_ms", 1850,
    b"commit_latency_ms", 12, b"client_id", -1, b"namespace", None,
    b"bucket", None, b"model", None, b"command", None, b"query", None,
    b"committed", 1, b"conflicted", 0, b"reads", 3, b"range_reads", 1,
    b"writes", 0, b"deletes", 0, b"range_deletes", 0, b"mutations", 0,
    b"bytes_read", 512, b"bytes_written", 0,
]  # fmt: skip


def test_parse_slow_transaction():
    assert parse_transaction([SLOW_TRANSACTION_RESP2]) == [
        {
            "id": 42,
            "timestamp": 1791964800000,
            "duration_ms": 1850,
            "commit_latency_ms": 12,
            "client_id": -1,
            "namespace": None,
            "bucket": None,
            "model": None,
            "command": None,
            "query": None,
            "committed": True,
            "conflicted": False,
            "reads": 3,
            "range_reads": 1,
            "writes": 0,
            "deletes": 0,
            "range_deletes": 0,
            "mutations": 0,
            "bytes_read": 512,
            "bytes_written": 0,
        }
    ]


@pytest.mark.integration
def test_list_transactions(kr, key):
    """An open ``BEGIN`` block is listed as ``idle`` with its last command."""
    with kr.session() as first, kr.session() as second:
        first.begin()
        first.zmap.set(key, "v")
        entries = [t for t in second.list_transactions() if t["command"] == "ZSET"]
        first.rollback()
    assert len(entries) == 1
    assert entries[0]["model"] == "ZMap"
    assert entries[0]["state"] == "idle"
    assert isinstance(entries[0]["watching"], bool)


@pytest.mark.integration
def test_kill_transaction(kr, key):
    """The owner fails on its next read and has to roll back."""
    with kr.session() as first, kr.session() as second:
        first.begin()
        first.zmap.set(key, "v")
        [entry] = [t for t in second.list_transactions() if t["command"] == "ZSET"]
        assert second.kill_transaction(entry["id"]) is True
        with pytest.raises(TransactionCancelledError):
            first.zmap.get(key)
        first.rollback()


@pytest.mark.integration
def test_kill_unknown_transaction(session):
    with pytest.raises(TransactionError, match="no such transaction"):
        session.kill_transaction(2**63 - 1)


@pytest.mark.integration
def test_slow_transactions(kr, key):
    """Needs the default ``transaction.slow_log.threshold`` of 1000 ms."""
    with kr.session() as first, kr.session() as second:
        first.begin()
        first.zmap.set(key, "v")
        [entry] = [t for t in second.list_transactions() if t["command"] == "ZSET"]
        time.sleep(1.1)
        first.commit()
        [slow] = [t for t in second.slow_transactions() if t["id"] == entry["id"]]
    assert slow["command"] == "ZSET"
    assert slow["committed"] is True


@pytest.mark.integration
def test_retry_is_disabled_on_the_pinned_connection(kr):
    """A retry would re-run the command in a brand new session."""
    with kr.session() as s:
        s.ping()
        assert s._redis.connection.retry.get_retries() == 0


@pytest.mark.integration
def test_failed_commit_ends_the_transaction(kr, key):
    """A commit that loses a conflict still ends the transaction on the server.

    The server cleans up in a ``finally`` block, so the retry ``NotCommittedError``
    asks for must not hit a client side nesting error.
    """
    with kr.session() as first, kr.session() as second:
        conflicting_writes(first, second, key)
        with pytest.raises(NotCommittedError):
            second.commit()
        assert second.in_transaction is False

        # The retry a caller does after catching NotCommittedError.
        with second.transaction():
            second.execute_command("ZSET", key, "retried")
        assert second.execute_command("ZGET", key) == b"retried"


@pytest.mark.integration
def test_failed_commit_returning_ends_the_transaction(session):
    """NO_COMMIT_VERSION runs the same server side cleanup as a conflict."""
    session.begin()
    with pytest.raises(NoCommitVersionError):
        session.commit("versionstamp")
    assert session.in_transaction is False
    with pytest.raises(TransactionError, match="no transaction in progress"):
        session.rollback()


@pytest.mark.integration
def test_a_timeout_breaks_the_session(kr, timeout_once):
    """redis-py closes the socket on a timeout, so the session is gone with it.

    ``redis.TimeoutError`` is not a ``ConnectionError``. It used to slip through,
    and the next command reconnected into a fresh session without saying so.
    """
    with kr.session() as s:
        s.begin()
        conn = s._redis.connection
        conn.read_response = timeout_once(conn)
        with pytest.raises(redis.TimeoutError):
            s.ping()

        assert conn.is_connected is False
        assert s.broken is True
        assert s.in_transaction is False
        with pytest.raises(KronotopConnectionError, match="connection was lost"):
            s.ping()


@pytest.mark.integration
def test_a_dropped_socket_breaks_the_session(kr):
    with kr.session() as s:
        s.begin()
        s._redis.connection.disconnect()
        with pytest.raises(KronotopConnectionError, match="connection was lost"):
            s.ping()
        assert s.broken is True
        assert s.in_transaction is False


def test_ignore_cleanup_error_logs_and_swallows(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="kronotop"),
        ignore_cleanup_error("close", "target"),
    ):
        raise RuntimeError("boom")
    assert "error ignored in close of 'target': boom" in caplog.text


@pytest.mark.integration
def test_rollback_failure_does_not_mask_the_original_error(kr, timeout_once):
    """The caller must see the original error.

    ``rollback`` runs on a dead socket, so ``_acquire`` raises. That error used
    to replace the timeout on its way out of the block.
    """
    with kr.session() as s:
        s.ping()
        conn = s._redis.connection
        stub = timeout_once(conn)
        with pytest.raises(redis.TimeoutError), s.transaction():
            conn.read_response = stub
            s.ping()


@pytest.mark.integration
def test_unclosed_session_does_not_contaminate_the_pool(kr):
    """A dropped session must not hand its state to the next one.

    redis-py would release the connection without ``SESSION.CLOSE``.
    ``gc.collect``, because ``session.zmap`` makes the session a cycle.
    """
    s = kr.session()
    s.ping()
    conn = s._redis.connection
    original_retry = s._saved_retry
    s.begin()
    s.execute_command("SESSION.ATTRIBUTE", "SET", "BATCH", "50")
    del s
    gc.collect()

    assert conn.is_connected is False
    # The pinned-connection override is gone. Comparing the object, not the
    # count: a pool built by ``from_url`` already defaults to zero retries.
    assert conn.retry is original_retry

    with kr.session() as fresh:
        attributes = fresh.execute_command("SESSION.ATTRIBUTE", "LIST")
        assert attributes[b"batch"] == 100
        with pytest.raises(TransactionError, match="no transaction in progress"):
            fresh.rollback()


@pytest.mark.integration
def test_unclosed_session_warns(kr):
    s = kr.session()
    s.ping()
    with pytest.warns(ResourceWarning, match="unclosed session"):
        del s
        gc.collect()


@pytest.mark.integration
def test_unclosed_session_gives_the_pool_slot_back(kr, kronotop_url):
    """More sessions than the pool holds, none of them closed.

    Takes ``kr`` only for its skip: this test builds its own client.
    """
    client = Kronotop.from_url(kronotop_url, max_connections=2)
    try:
        for _ in range(4):
            s = client.session()
            s.ping()
            del s
            gc.collect()
    finally:
        client.close()


@pytest.mark.integration
def test_ping(session):
    assert session.ping() is True


@pytest.mark.integration
def test_ping_with_message(session):
    assert session.ping("Hello, World!") == b"Hello, World!"


@pytest.mark.integration
def test_echo(session):
    assert session.echo("Hello, World!") == b"Hello, World!"


@pytest.mark.integration
def test_time(session):
    seconds, microseconds = session.time()
    assert seconds > 0
    assert 0 <= microseconds < 1_000_000


@pytest.mark.integration
def test_info(session):
    reply = session.info()
    assert reply["server_name"] == "kronotop"
    assert reply["server_mode"] == "standalone"
    assert reply["cluster_enabled"] == "0"
    # Never a number, so a version keeps every digit it was sent with.
    assert isinstance(reply["kronotop_version"], str)


@pytest.mark.integration
def test_info_with_a_section(session):
    reply = session.info("kronotop")
    assert "cluster_name" in reply
    assert "cluster_enabled" not in reply


@pytest.mark.integration
def test_info_with_an_unknown_section(session):
    assert session.info("nosuch") == {}
