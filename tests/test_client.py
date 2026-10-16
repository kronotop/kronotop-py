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

"""Structural tests: no server required."""

from __future__ import annotations

import socket
import threading

import pytest
import redis
from redis.cache import CacheConfig
from redis.connection import SSLConnection
from redis.maint_notifications import MaintNotificationsConfig

import kronotop
from kronotop import (
    DEFAULT_PORT,
    ConfigurationError,
    Kronotop,
    NoCommonMemberError,
    _members,
)
from kronotop._callbacks import parse_bucket_locate
from kronotop._options import parse_host, resolve
from kronotop._routing import Placement, common_members, primaries
from kronotop.commands.bucket import BucketLocation, ShardLocation
from kronotop.exceptions import _parse_reject


def test_default_port(only_pool):
    client = Kronotop()
    assert only_pool(client).connection_kwargs["port"] == DEFAULT_PORT


def test_from_url_returns_our_client(only_pool):
    client = Kronotop.from_url(f"kronotop://localhost:{DEFAULT_PORT}")
    assert isinstance(client, Kronotop)
    assert only_pool(client).connection_kwargs["port"] == DEFAULT_PORT


def test_module_level_from_url(only_pool):
    client = kronotop.from_url("kronotop://localhost", protocol=2)
    assert isinstance(client, Kronotop)
    kwargs = only_pool(client).connection_kwargs
    assert kwargs["port"] == DEFAULT_PORT
    assert kwargs["protocol"] == 2


@pytest.mark.parametrize("command", ["set", "lpush", "xadd", "pipeline", "acl_list"])
def test_redis_commands_are_not_exposed(command):
    """Kronotop does not implement Redis' data commands, so we do not expose them."""
    assert not hasattr(Kronotop(), command)


def test_bootstrap_keeps_its_order_and_default_port():
    client = Kronotop(bootstrap=["a", "b:5485"])
    assert client.hosts == (f"a:{DEFAULT_PORT}", "b:5485")


def test_one_pool_per_host():
    client = Kronotop(bootstrap=["a", "b"])
    a, b = client._pools.values()
    assert a is not b


@pytest.mark.parametrize("bootstrap", [None, ["a"]])
@pytest.mark.parametrize("name", ["host", "port", "connection_pool"])
def test_a_member_named_outside_bootstrap_is_refused(name, bootstrap):
    """``bootstrap`` is the only way to name a member, with or without a list."""
    value = redis.ConnectionPool(host="a") if name == "connection_pool" else "a"
    with pytest.raises(ConfigurationError, match="name the member in bootstrap"):
        Kronotop(bootstrap=bootstrap, **{name: value})


@pytest.mark.parametrize(
    ("bootstrap", "message"),
    [
        ([], "at least one member"),
        (["a:x"], "no valid port"),
        ([":5484"], "no host"),
        (["[::1:5484"], "has no closing"),
        (["[::1]x"], "no valid port"),
        (["a:b:5484"], "no valid host"),
        (["[not:an:address]"], "no valid host"),
    ],
)
def test_invalid_bootstrap(bootstrap, message):
    with pytest.raises(ConfigurationError, match=message):
        Kronotop(bootstrap=bootstrap)


def test_session_picks_the_host():
    client = Kronotop(bootstrap=["a", "b"])
    with client.session(host="b:5484") as session:
        assert session._pool is client._pools["b:5484"]


@pytest.mark.parametrize("call", ["session", "remove"])
def test_unknown_host(call):
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(ConfigurationError, match="is not one of"):
        if call == "session":
            client.session(host="b:5484")
        else:
            client._members.remove("b:5484")


def test_a_command_goes_to_each_member_in_turn():
    """There is no default member, so the same call moves on."""
    client = Kronotop(bootstrap=["a", "b"])
    a, b = client._pools.values()
    assert [client.session()._pool for _ in range(4)] == [a, b, a, b]


def test_add_member_carries_the_transport_settings():
    client = Kronotop(bootstrap=["a"], password="pw", ssl=True, socket_timeout=3)
    pool = client._members.add("b:5485").pool
    assert client.hosts == (f"a:{DEFAULT_PORT}", "b:5485")
    assert pool.connection_class is SSLConnection
    assert pool.connection_kwargs["password"] == "pw"
    assert pool.connection_kwargs["socket_timeout"] == 3
    assert pool.connection_kwargs["port"] == 5485


def test_add_member_of_a_url_built_client():
    """The URL settings reach the new member through the first member's pool."""
    client = Kronotop.from_url("kronotops://user:pw@a")
    pool = client._members.add("b").pool
    assert pool.connection_class is SSLConnection
    assert pool.connection_kwargs["password"] == "pw"


def test_add_member_keeps_maintenance_notifications_off():
    """redis-py keeps the config out of ``connection_kwargs`` while it is off."""
    client = Kronotop(bootstrap=["a"])
    pool = client._members.add("b").pool
    assert not pool.maint_notifications_enabled()
    assert "maint_notifications_pool_handler" not in pool.connection_kwargs


def test_add_member_can_keep_maintenance_notifications_on():
    client = Kronotop(
        bootstrap=["a"],
        maint_notifications_config=MaintNotificationsConfig(enabled=True),
    )
    assert (
        "maint_notifications_pool_handler"
        in client._members.add("b").pool.connection_kwargs
    )


def test_close_disconnects_a_pool_built_here():
    """redis-py leaves a pool it was handed open, and this one is ours."""
    client = Kronotop(bootstrap=["a"])
    client._members.add("b")
    assert all(
        member.client.auto_close_connection_pool for member in client._members.all
    )


def test_add_member_twice_keeps_one_pool():
    client = Kronotop(bootstrap=["a"])
    assert client._members.add("b").pool is client._members.add("b:5484").pool
    assert client.hosts == (f"a:{DEFAULT_PORT}", "b:5484")


@pytest.mark.parametrize("call", ["session", "get", "remove"])
def test_a_host_without_a_port_finds_the_member(call):
    """``add`` stores ``a:5484``, so every lookup has to read ``a`` that way."""
    client = Kronotop(bootstrap=["a", "b"])
    if call == "session":
        assert client.session(host="a")._pool is client._pools[f"a:{DEFAULT_PORT}"]
    elif call == "remove":
        client._members.remove("a")
        assert client.hosts == ("b:5484",)
    else:
        client._members.get("a")


def test_a_new_member_takes_its_turn():
    client = Kronotop(bootstrap=["a"])
    client._members.add("b")
    a, b = client._pools.values()
    assert [client.session()._pool for _ in range(3)] == [a, b, a]


def test_remove_member_stops_the_commands():
    client = Kronotop(bootstrap=["a", "b"])
    client._members.remove("b:5484")
    assert client.hosts == (f"a:{DEFAULT_PORT}",)
    assert client.session()._pool is client._pools[f"a:{DEFAULT_PORT}"]


def test_remove_member_leaves_the_pool_open():
    """A session on that member is still running."""
    client = Kronotop(bootstrap=["a", "b"])
    pool = client._pools["b:5484"]
    session = client.session(host="b:5484")
    client._members.remove("b:5484")
    assert session._pool is pool
    assert pool.connection_kwargs


def test_the_last_member_stays():
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(ConfigurationError, match="last member"):
        client._members.remove(f"a:{DEFAULT_PORT}")


def test_members_change_while_commands_pick_one():
    """Every reader takes the member list once, so it never sees half a change."""
    client = Kronotop(bootstrap=["a"])
    stop = threading.Event()
    failures = []

    def pick():
        while not stop.is_set():
            try:
                assert client.session()._pool is not None
            except Exception as exc:
                failures.append(exc)
                return

    def churn(port):
        # A port of this thread's own, so two writers never race for one name.
        try:
            for _ in range(200):
                client._members.add(f"b:{port}")
                client._members.remove(f"b:{port}")
        except Exception as exc:
            failures.append(exc)

    readers = [threading.Thread(target=pick) for _ in range(4)]
    for thread in readers:
        thread.start()
    writers = [threading.Thread(target=churn, args=(port,)) for port in (5485, 5486)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()
    assert not failures


def test_two_threads_add_the_same_member():
    """One name is one pool, however many threads ask for it at once."""
    client = Kronotop(bootstrap=["a"])
    start = threading.Barrier(8)
    results = []

    def add():
        start.wait()
        results.append(client._members.add("b:5485"))

    threads = [threading.Thread(target=add) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(member) for member in results}) == 1
    assert client.hosts.count("b:5485") == 1


def test_version_is_exported():
    assert kronotop.__version__


def _two_hosts(kr):
    """The same server named twice, so one member has two names."""
    (pool,) = kr._pools.values()
    kwargs = pool.connection_kwargs
    other = "127.0.0.1" if kwargs["host"] != "127.0.0.1" else "localhost"
    return [f"{kwargs['host']}:{kwargs['port']}", f"{other}:{kwargs['port']}"]


@pytest.mark.integration
def test_two_names_for_one_server_give_one_member(kr):
    """And the name that was dropped still finds the member that was kept."""
    first, second = _two_hosts(kr)
    with Kronotop(bootstrap=[first, second]) as client:
        assert client.hosts == (first,)
        assert client.execute_command("PING") is True
        with client.session(host=second) as session:
            assert session.ping() is True
        with client.session(host=first) as session:
            assert session.attributes


@pytest.mark.integration
def test_a_second_name_for_one_server_answers(kr):
    """``add`` hands back the member it has, and the client keeps working."""
    first, second = _two_hosts(kr)
    with Kronotop(bootstrap=[first]) as client:
        assert client._members.add(second).name == first
        assert client.hosts == (first,)
        assert client.execute_command("PING") is True


def test_from_url_falls_back_to_the_kronotop_port(only_pool):
    """redis-py would use 6379 for a URL that names no port."""
    client = Kronotop.from_url("kronotop://localhost")
    assert only_pool(client).connection_kwargs["port"] == DEFAULT_PORT


def test_a_port_in_the_url_wins(only_pool):
    client = Kronotop.from_url("kronotop://localhost:6379")
    assert only_pool(client).connection_kwargs["port"] == 6379


def test_a_tls_url_is_accepted(only_pool):
    """``kronotops://`` is the TLS scheme, matching network.external.tls."""
    client = Kronotop.from_url("kronotops://localhost")
    assert only_pool(client).connection_class is SSLConnection


@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:5484",
        "rediss://localhost:5484",
        "unix:///tmp/kronotop.sock",
        "localhost:5484",
    ],
)
def test_other_url_schemes_are_refused(url):
    """Kronotop is not Redis, and it listens on TCP only."""
    with pytest.raises(ConfigurationError):
        Kronotop.from_url(url)


def test_a_unix_socket_path_is_refused():
    with pytest.raises(ConfigurationError, match="unix sockets"):
        Kronotop(unix_socket_path="/tmp/kronotop.sock")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"db": 1},
        {"single_connection_client": True},
        {"cache_config": CacheConfig()},
    ],
)
def test_unsupported_options_are_refused(kwargs):
    with pytest.raises(ConfigurationError):
        Kronotop(**kwargs)


def test_a_forbidden_option_at_its_default_is_accepted(only_pool):
    """Every one of them is checked on its value, like ``db``."""
    client = Kronotop(single_connection_client=False, cache=None, unix_socket_path=None)
    assert only_pool(client).connection_kwargs["port"] == DEFAULT_PORT


def test_a_db_in_the_url_is_refused():
    with pytest.raises(ConfigurationError, match="namespace"):
        Kronotop.from_url("kronotop://localhost/1")


def test_db_zero_is_fine(only_pool):
    """redis-py sends SELECT only for a non-zero db, and a URL path gives 0."""
    assert only_pool(Kronotop.from_url("kronotop://localhost/0")) is not None


def test_maintenance_notifications_are_off_by_default(only_pool):
    """A reconnect on a pinned connection would start a new server side session."""
    client = Kronotop()
    assert "maint_notifications_pool_handler" not in (
        only_pool(client).connection_kwargs
    )


def test_maintenance_notifications_can_be_turned_back_on(only_pool):
    client = Kronotop(maint_notifications_config=MaintNotificationsConfig(enabled=True))
    assert "maint_notifications_pool_handler" in only_pool(client).connection_kwargs


def test_session_for_needs_a_bucket():
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(ConfigurationError, match="names no bucket"):
        client.session_for(buckets={})


def test_session_for_needs_a_name_in_every_namespace():
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(ConfigurationError, match="names no bucket"):
        client.session_for(buckets={"app": []})


@pytest.mark.parametrize("names", ["users", b"users"])
def test_session_for_refuses_one_name_where_a_list_belongs(names):
    """Iterating one would look its letters, or its bytes, up one at a time."""
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(ConfigurationError, match="needs a list of bucket names"):
        client.session_for(buckets={"app": names})


def test_session_for_takes_no_host():
    """The member comes from the buckets, so naming one would say it twice."""
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(TypeError):
        client.session_for(buckets={"app": ["users"]}, host="a:5484")


def test_placement_keeps_and_forgets():
    placement = Placement()
    assert placement.get("app", "users") is None
    placement.keep("app", "users", {"a:5484"})
    assert placement.get("app", "users") == {"a:5484"}
    placement.forget({"app": ["users", "orders"]})
    assert placement.get("app", "users") is None


def fake_session(locates):
    """Stands in for ``Session`` inside ``session_for``.

    Every LOCATE lands in ``locates`` and answers with the bootstrap member, so
    no lookup runs. The session ``session_for`` returns is one of these too.
    """

    class FakeBucket:
        def locate(self, name, *, namespace=None):
            locates.append((namespace, name))
            return BucketLocation([ShardLocation(0, "one", ())], {"one": ("a:5484",)})

    class FakeSession:
        def __init__(self, pool, config=None, *, on_reject=None):
            self.bucket = FakeBucket()
            self.on_reject = on_reject

        def close(self):
            pass

    return FakeSession


def test_session_for_locates_a_bucket_once(monkeypatch):
    """The second call sends nothing. A ``RejectError`` makes the third send again."""
    locates = []
    monkeypatch.setattr("kronotop.client.Session", fake_session(locates))
    client = Kronotop(bootstrap=["a"])
    buckets = {"app": ["users"]}

    client.session_for(buckets=buckets)
    opened = client.session_for(buckets=buckets)
    assert locates == [("app", "users")]

    opened.on_reject()
    client.session_for(buckets=buckets)
    assert locates == [("app", "users"), ("app", "users")]


def test_next_of_takes_its_turn_over_the_shortlist():
    """A name it does not know is added, because LOCATE can report one."""
    client = Kronotop(bootstrap=["a"])
    names = ["b:5484", "c:5484"]
    picked = [client._members.next_of(names).name for _ in range(3)]
    assert picked == ["b:5484", "c:5484", "b:5484"]
    assert client.hosts == ("a:5484", "b:5484", "c:5484")


def test_common_members_is_the_intersection():
    found = [("app", "users", {"a", "b"}), ("app", "orders", {"b", "c"})]
    assert common_members(found) == {"b"}


def test_common_members_names_where_each_bucket_is():
    found = [("app", "users", {"a"}), ("app", "orders", {"b"})]
    with pytest.raises(NoCommonMemberError, match="users in app on a"):
        common_members(found)


def test_common_members_reports_a_bucket_with_no_route():
    """LOCATE leaves out a shard the cluster cannot route to."""
    with pytest.raises(NoCommonMemberError, match="on no member"):
        common_members([("app", "users", set())])


def test_parse_locate_reads_both_tables():
    """Two shards, one standby. The ids come back as text, bytes or not."""
    reply = [
        [0, b"one", [b"two"], 1, b"one", []],
        [b"one", [b"10.0.0.1:5484", b"[fd00::1]:5484"], b"two", [b"10.0.0.2:5484"]],
    ]
    location = parse_bucket_locate(reply)
    assert location == BucketLocation(
        [
            ShardLocation(0, "one", ("two",)),
            ShardLocation(1, "one", ()),
        ],
        {
            "one": ("10.0.0.1:5484", "[fd00::1]:5484"),
            "two": ("10.0.0.2:5484",),
        },
    )


def test_primaries_are_looked_up_in_the_member_table():
    """A standby is left out, and the first address of a member is the one used."""
    location = BucketLocation(
        [ShardLocation(0, "one", ("two",)), ShardLocation(1, "three", ())],
        {
            "one": ("10.0.0.1:5484", "[fd00::1]:5484"),
            "two": ("10.0.0.2:5484",),
            "three": ("10.0.0.3:5484",),
        },
    )
    assert primaries(location) == {"10.0.0.1:5484", "10.0.0.3:5484"}


def test_parse_locate_of_a_bucket_with_no_route():
    """Both tables are empty when no shard of the bucket can be routed to.

    The bucket exists, or the reply would be ``NOSUCHBUCKET``.
    """
    assert parse_bucket_locate([[], []]) == BucketLocation([], {})


def test_primaries_of_a_bucket_with_no_route():
    """Empty, so ``common_members`` is the one that raises. See below."""
    assert primaries(BucketLocation([], {})) == set()


def test_session_for_reports_a_bucket_with_no_route():
    """The two steps above, end to end: an empty LOCATE stops the session."""
    location = parse_bucket_locate([[], []])
    with pytest.raises(NoCommonMemberError, match="users in app on no member"):
        common_members([("app", "users", primaries(location))])


@pytest.mark.parametrize(
    "reply",
    [
        [[0, b"one"], [b"one", [b"10.0.0.1:5484"]]],
        [[0, b"one", []], [b"one"]],
    ],
    ids=["route table", "member table"],
)
def test_parse_locate_rejects_a_half_row(reply):
    """A table that does not divide into rows is an error, not a short read.

    Dropping the half row would hide a shard, and the session would then open on
    a member that does not hold it.
    """
    with pytest.raises(ValueError):
        parse_bucket_locate(reply)


def test_primaries_needs_every_primary_in_the_member_table():
    """The server writes a row for each member it names, so this cannot happen.

    Here to keep it loud if it ever does: a missing address would otherwise
    drop the shard and send the session somewhere else.
    """
    location = BucketLocation([ShardLocation(0, "one", ())], {})
    with pytest.raises(KeyError):
        primaries(location)


@pytest.mark.parametrize(
    "other",
    ["127.0.0.1:5484", "[::1]:5484", "0:0:0:0:0:0:0:1:5484", "LOCALHOST:5484"],
)
def test_one_member_under_two_names(other):
    """LOCATE reports the server's address, bootstrap says what the user typed."""
    client = Kronotop(bootstrap=["localhost"])
    assert client._members.add(other) is client._members.get("localhost:5484")
    assert client.hosts == ("localhost:5484",)


def test_the_same_ipv6_address_written_two_ways(only_pool):
    client = Kronotop(bootstrap=["[2001:db8::1]:5484"])
    assert only_pool(client).connection_kwargs["host"] == "2001:db8::1"
    same = client._members.add("2001:0DB8:0000:0000:0000:0000:0000:0001:5484")
    assert same.name == "[2001:db8::1]:5484"
    assert client.hosts == ("[2001:db8::1]:5484",)


@pytest.mark.parametrize("entry", ["[::1]:5484", "[::1]", "::1"])
def test_an_ipv6_member_is_reachable(entry, only_pool):
    """The pool takes the bare address. getaddrinfo refuses a bracketed one."""
    client = Kronotop(bootstrap=[entry])
    assert client.hosts == ("[::1]:5484",)
    host = only_pool(client).connection_kwargs["host"]
    assert host == "::1"
    assert socket.getaddrinfo(host, DEFAULT_PORT, type=socket.SOCK_STREAM)


def test_an_ipv6_address_the_way_the_server_writes_it(only_pool):
    """LOCATE and REJECT bracket the address when a port follows it."""
    client = Kronotop(bootstrap=["[fd00::5]:5484"])
    assert client.hosts == ("[fd00::5]:5484",)
    kwargs = only_pool(client).connection_kwargs
    assert (kwargs["host"], kwargs["port"]) == ("fd00::5", 5484)


def test_a_bare_ipv6_address_is_the_whole_host():
    """RFC 5952 asks for brackets around a port, not around an address."""
    assert parse_host("fd00::5") == ("[fd00::5]:5484", "fd00::5", DEFAULT_PORT)


def test_a_rejected_address_reads_the_same_way_twice():
    """``RejectError`` and the member list have to name a member the same way.

    The address is kept as the server wrote it, so the name has to be the one
    ``parse_host`` builds. Otherwise ``bootstrap`` would take a second spelling.
    """
    shard, addresses = _parse_reject("REJECT 3 [fd00::5]:5484")
    assert (shard, addresses) == (3, ("[fd00::5]:5484",))
    assert parse_host(addresses[0])[0] == addresses[0]


def test_an_ipv6_url(only_pool):
    client = Kronotop.from_url("kronotop://[::1]:5484")
    assert client.hosts == ("[::1]:5484",)
    assert only_pool(client).connection_kwargs["host"] == "::1"


def test_another_loopback_address_is_another_member():
    """127.0.0.2 is an address of its own, and a member can listen on it."""
    client = Kronotop(bootstrap=["localhost:5484", "127.0.0.2:5484"])
    assert client.hosts == ("localhost:5484", "127.0.0.2:5484")


def test_an_ipv4_mapped_loopback_reads_as_the_ipv4_one():
    client = Kronotop(bootstrap=["127.0.0.2:5484"])
    assert client._members.add("[::ffff:127.0.0.2]:5484").name == "127.0.0.2:5484"


def test_an_ipv4_mapped_address_is_the_ipv4_one():
    client = Kronotop(bootstrap=["10.0.0.5:5484"])
    assert client._members.add("[::ffff:10.0.0.5]:5484").name == "10.0.0.5:5484"


def test_another_port_on_one_host_is_another_member():
    client = Kronotop(bootstrap=["localhost:5484"])
    client._members.add("127.0.0.1:5485")
    assert client.hosts == ("localhost:5484", "127.0.0.1:5485")


def test_a_hostname_and_its_address_are_one_member(monkeypatch):
    """The two fold only when the resolver says so. This test asks it."""
    monkeypatch.setattr(
        _members,
        "resolve",
        lambda host: frozenset({"10.0.0.5"}),
    )
    client = Kronotop(bootstrap=["db1.internal:5484"])
    assert client._members.add("10.0.0.5:5484").name == "db1.internal:5484"
    assert client.hosts == ("db1.internal:5484",)


def test_a_name_that_resolves_nowhere_stays_its_own_member(monkeypatch):
    """An empty answer matches nothing, so the names are compared as they are."""
    monkeypatch.setattr(_members, "resolve", lambda host: frozenset())
    client = Kronotop(bootstrap=["db1.internal:5484"])
    client._members.add("10.0.0.5:5484")
    assert client.hosts == ("db1.internal:5484", "10.0.0.5:5484")


def test_a_host_that_answered_is_looked_up_once(monkeypatch):
    asked = []

    def note(host):
        asked.append(host)
        return frozenset({host})

    monkeypatch.setattr(_members, "resolve", note)
    client = Kronotop(bootstrap=["db1.internal:5484"])
    client._members.add("10.0.0.5:5484")
    client._members.add("10.0.0.5:5484")
    client._members.add("10.0.0.6:5484")
    assert asked == ["10.0.0.5", "db1.internal", "10.0.0.6"]


def test_a_lookup_that_failed_is_asked_again(monkeypatch):
    """A name server down for a moment must not settle the question."""
    asked = []
    answers = {}

    def note(host):
        asked.append(host)
        return answers.get(host, frozenset())

    monkeypatch.setattr(_members, "resolve", note)
    client = Kronotop(bootstrap=["db1.internal:5484"])
    client._members.add("10.0.0.5:5484")
    assert asked == ["10.0.0.5"]

    answers["10.0.0.6"] = frozenset({"10.0.0.6"})
    client._members.add("10.0.0.6:5484")
    assert asked == ["10.0.0.5", "10.0.0.6", "db1.internal", "10.0.0.5"]


def test_a_host_with_no_address_leaves_the_members_alone(monkeypatch):
    """One failed lookup per call while a name server is down, not one each."""
    asked = []

    def note(host):
        asked.append(host)
        return frozenset()

    monkeypatch.setattr(_members, "resolve", note)
    client = Kronotop(bootstrap=[f"db{number}.internal" for number in range(4)])
    client._members.add("new.internal:5484")
    assert asked == ["new.internal"]


def test_a_failed_lookup_does_not_stop_a_later_fold(monkeypatch):
    """One failure covers every member on that port, so all of them are retried."""
    answers = {}
    monkeypatch.setattr(
        _members, "resolve", lambda host: answers.get(host, frozenset())
    )
    client = Kronotop(bootstrap=["db1.internal:5484"])
    client._members.add("other.internal:5484")
    assert client.hosts == ("db1.internal:5484", "other.internal:5484")

    answers["db1.internal"] = answers["10.0.0.5"] = frozenset({"10.0.0.5"})
    assert client._members.add("10.0.0.5:5484").name == "db1.internal:5484"
    assert client.hosts == ("db1.internal:5484", "other.internal:5484")


def test_a_lookup_only_covers_the_members_on_that_port(monkeypatch):
    asked = []

    def note(host):
        asked.append(host)
        return frozenset({host})

    monkeypatch.setattr(_members, "resolve", note)
    client = Kronotop(bootstrap=["db1.internal:5484", "db2.internal:5485"])
    client._members.add("10.0.0.5:5484")
    assert asked == ["10.0.0.5", "db1.internal"]


def test_resolve_folds_the_loopback_names():
    """A real lookup, of a name every machine answers without a name server."""
    assert resolve("localhost") == frozenset({"localhost"})


def test_resolve_of_a_name_that_cannot_exist():
    """``.invalid`` is reserved and must never resolve."""
    assert resolve("no-such-host.invalid") == frozenset()


def test_bootstrap_keeps_one_member_named_twice():
    """Two pools for one server would take two turns out of every round."""
    client = Kronotop(bootstrap=["localhost", "127.0.0.1:5484", "[::1]:5484", "b"])
    assert client.hosts == ("localhost:5484", "b:5484")


def test_bootstrap_keeps_the_first_name_of_a_member():
    client = Kronotop(bootstrap=["127.0.0.1:5484", "localhost"])
    assert client.hosts == ("127.0.0.1:5484",)


def test_bootstrap_keeps_a_member_on_another_port():
    client = Kronotop(bootstrap=["localhost:5484", "127.0.0.1:5485"])
    assert client.hosts == ("localhost:5484", "127.0.0.1:5485")
