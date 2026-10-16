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

"""Async twin of ``tests/test_client.py``."""

from __future__ import annotations

import pytest
import redis.asyncio
from redis.asyncio.connection import SSLConnection
from redis.cache import CacheConfig
from redis.maint_notifications import MaintNotificationsConfig

import kronotop.asyncio
from kronotop import DEFAULT_PORT, ConfigurationError, _members
from kronotop.asyncio import Kronotop
from kronotop.commands.bucket import BucketLocation, ShardLocation


def test_default_port(only_pool):
    client = Kronotop()
    assert only_pool(client).connection_kwargs["port"] == DEFAULT_PORT


def test_from_url_returns_our_client():
    client = Kronotop.from_url(f"kronotop://localhost:{DEFAULT_PORT}")
    assert isinstance(client, Kronotop)


def test_module_level_from_url(only_pool):
    client = kronotop.asyncio.from_url("kronotop://localhost", protocol=2)
    assert isinstance(client, Kronotop)
    kwargs = only_pool(client).connection_kwargs
    assert kwargs["port"] == DEFAULT_PORT
    assert kwargs["protocol"] == 2


@pytest.mark.parametrize("command", ["set", "lpush", "xadd", "pipeline", "acl_list"])
def test_redis_commands_are_not_exposed(command):
    assert not hasattr(Kronotop(), command)


def test_bootstrap_keeps_its_order_and_default_port():
    client = Kronotop(bootstrap=["a", "b:5485"])
    assert client.hosts == (f"a:{DEFAULT_PORT}", "b:5485")


@pytest.mark.parametrize("bootstrap", [None, ["a"]])
@pytest.mark.parametrize("name", ["host", "port", "connection_pool"])
def test_a_member_named_outside_bootstrap_is_refused(name, bootstrap):
    value = redis.asyncio.ConnectionPool(host="a") if name == "connection_pool" else "a"
    with pytest.raises(ConfigurationError, match="name the member in bootstrap"):
        Kronotop(bootstrap=bootstrap, **{name: value})


@pytest.mark.parametrize(
    ("bootstrap", "message"),
    [
        ([], "at least one member"),
        (["a:x"], "no valid port"),
        (["[::1:5484"], "has no closing"),
    ],
)
def test_invalid_bootstrap(bootstrap, message):
    with pytest.raises(ConfigurationError, match=message):
        Kronotop(bootstrap=bootstrap)


async def test_session_picks_the_host():
    client = Kronotop(bootstrap=["a", "b"])
    assert client.session(host="b:5484")._pool is client._pools["b:5484"]


async def test_unknown_host():
    with pytest.raises(ConfigurationError, match="is not one of"):
        Kronotop(bootstrap=["a"]).session(host="b:5484")


async def test_a_command_goes_to_each_member_in_turn():
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


def test_add_member_keeps_maintenance_notifications_off():
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


async def test_remove_member_leaves_the_pool_open():
    """A session on that member is still running."""
    client = Kronotop(bootstrap=["a", "b"])
    pool = client._pools["b:5484"]
    session = client.session(host="b:5484")
    client._members.remove("b:5484")
    assert session._pool is pool
    assert pool.connection_kwargs


def test_close_disconnects_a_pool_built_here():
    client = Kronotop(bootstrap=["a"])
    client._members.add("b")
    assert all(
        member.client.auto_close_connection_pool for member in client._members.all
    )


async def test_a_host_without_a_port_finds_the_member():
    client = Kronotop(bootstrap=["a", "b"])
    assert client.session(host="a")._pool is client._pools[f"a:{DEFAULT_PORT}"]
    client._members.remove("a")
    assert client.hosts == ("b:5484",)


def test_add_member_twice_keeps_one_pool():
    client = Kronotop(bootstrap=["a"])
    assert client._members.add("b").pool is client._members.add("b:5484").pool


async def test_a_new_member_takes_its_turn():
    client = Kronotop(bootstrap=["a"])
    client._members.add("b")
    a, b = client._pools.values()
    assert [client.session()._pool for _ in range(3)] == [a, b, a]


async def test_remove_member_stops_the_commands():
    client = Kronotop(bootstrap=["a", "b"])
    client._members.remove("b:5484")
    assert client.hosts == (f"a:{DEFAULT_PORT}",)
    assert client.session()._pool is client._pools[f"a:{DEFAULT_PORT}"]


def test_the_last_member_stays():
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(ConfigurationError, match="last member"):
        client._members.remove(f"a:{DEFAULT_PORT}")


def _two_hosts(kr):
    """The same server named twice, so one member has two names."""
    (pool,) = kr._pools.values()
    kwargs = pool.connection_kwargs
    other = "127.0.0.1" if kwargs["host"] != "127.0.0.1" else "localhost"
    return [f"{kwargs['host']}:{kwargs['port']}", f"{other}:{kwargs['port']}"]


@pytest.mark.integration
async def test_two_names_for_one_server_give_one_member(kr):
    """And the name that was dropped still finds the member that was kept."""
    first, second = _two_hosts(kr)
    async with Kronotop(bootstrap=[first, second]) as client:
        assert client.hosts == (first,)
        assert await client.execute_command("PING") is True
        async with client.session(host=second) as session:
            assert await session.ping() is True
        async with client.session(host=first) as session:
            assert await session.attributes


@pytest.mark.integration
async def test_a_second_name_for_one_server_answers(kr):
    """``add`` hands back the member it has, and the client keeps working."""
    first, second = _two_hosts(kr)
    async with Kronotop(bootstrap=[first]) as client:
        assert client._members.add(second).name == first
        assert client.hosts == (first,)
        assert await client.execute_command("PING") is True


def test_from_url_falls_back_to_the_kronotop_port(only_pool):
    client = Kronotop.from_url("kronotop://localhost")
    assert only_pool(client).connection_kwargs["port"] == DEFAULT_PORT


def test_a_port_in_the_url_wins(only_pool):
    client = Kronotop.from_url("kronotop://localhost:6379")
    assert only_pool(client).connection_kwargs["port"] == 6379


def test_a_tls_url_is_accepted(only_pool):
    client = Kronotop.from_url("kronotops://localhost")
    assert only_pool(client).connection_class is SSLConnection


def test_add_member_of_a_url_built_client(only_pool):
    """The URL settings reach the new member through the first member's pool."""
    client = Kronotop.from_url("kronotops://user:pw@a")
    pool = client._members.add("b").pool
    assert pool.connection_class is SSLConnection
    assert pool.connection_kwargs["password"] == "pw"


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
    # ``cache`` and ``cache_config`` are sync only, so the async twin sends the
    # two options ``redis.asyncio.Redis`` takes.
    client = Kronotop(single_connection_client=False, unix_socket_path=None)
    assert only_pool(client).connection_kwargs["port"] == DEFAULT_PORT


def test_a_db_in_the_url_is_refused():
    with pytest.raises(ConfigurationError, match="namespace"):
        Kronotop.from_url("kronotop://localhost/1")


def test_db_zero_is_fine(only_pool):
    assert only_pool(Kronotop.from_url("kronotop://localhost/0")) is not None


def test_maintenance_notifications_are_off_by_default(only_pool):
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
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(ConfigurationError, match="needs a list of bucket names"):
        client.session_for(buckets={"app": names})


def test_session_for_takes_no_host():
    client = Kronotop(bootstrap=["a"])
    with pytest.raises(TypeError):
        client.session_for(buckets={"app": ["users"]}, host="a:5484")


def fake_session(locates):
    """The async twin of the stand-in in ``tests/test_client.py``."""

    class FakeBucket:
        async def locate(self, name, *, namespace=None):
            locates.append((namespace, name))
            return BucketLocation([ShardLocation(0, "one", ())], {"one": ("a:5484",)})

    class FakeSession:
        def __init__(self, pool, config=None, *, on_reject=None):
            self.bucket = FakeBucket()
            self.on_reject = on_reject

        async def close(self):
            pass

    return FakeSession


async def test_session_for_locates_a_bucket_once(monkeypatch):
    locates = []
    monkeypatch.setattr("kronotop.asyncio.client.Session", fake_session(locates))
    client = Kronotop(bootstrap=["a"])
    buckets = {"app": ["users"]}

    await client.session_for(buckets=buckets)
    opened = await client.session_for(buckets=buckets)
    assert locates == [("app", "users")]

    opened.on_reject()
    await client.session_for(buckets=buckets)
    assert locates == [("app", "users"), ("app", "users")]


async def test_a_hostname_and_its_address_are_one_member(monkeypatch):
    """``aadd`` asks the loop's resolver, so a slow one does not stop the loop."""

    async def resolved(host):
        return frozenset({"10.0.0.5"})

    monkeypatch.setattr(_members, "aresolve", resolved)
    client = Kronotop(bootstrap=["db1.internal:5484"])
    assert (await client._members.aadd("10.0.0.5:5484")).name == "db1.internal:5484"
    assert client.hosts == ("db1.internal:5484",)


async def test_a_lookup_that_failed_is_asked_again(monkeypatch):
    """``aadd`` keeps an answer and drops a failure, the way ``add`` does."""
    asked = []
    answers = {}

    async def note(host):
        asked.append(host)
        return answers.get(host, frozenset())

    monkeypatch.setattr(_members, "aresolve", note)
    client = Kronotop(bootstrap=["db1.internal:5484"])
    await client._members.aadd("10.0.0.5:5484")
    assert asked == ["10.0.0.5"]

    answers["10.0.0.6"] = frozenset({"10.0.0.6"})
    await client._members.aadd("10.0.0.6:5484")
    assert asked == ["10.0.0.5", "10.0.0.6", "db1.internal", "10.0.0.5"]


def test_an_ipv6_member_is_reachable(only_pool):
    """The pool takes the bare address. getaddrinfo refuses a bracketed one."""
    client = Kronotop(bootstrap=["[::1]:5484"])
    assert client.hosts == ("[::1]:5484",)
    assert only_pool(client).connection_kwargs["host"] == "::1"


async def test_anext_of_takes_its_turn_over_the_shortlist():
    client = Kronotop(bootstrap=["a"])
    names = ["b:5484", "c:5484"]
    picked = [(await client._members.anext_of(names)).name for _ in range(3)]
    assert picked == ["b:5484", "c:5484", "b:5484"]
    assert client.hosts == ("a:5484", "b:5484", "c:5484")
