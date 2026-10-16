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

"""``kronotop.asyncio.Admin``.

The parser and the argument wiring are shared with the sync client and are
covered in ``tests/test_admin.py``.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop import DEFAULT_ADMIN_PORT, ConfigurationError
from kronotop.asyncio import Admin


def connection_kwargs(admin):
    return admin._redis.connection_pool.connection_kwargs


def test_default_host():
    kwargs = connection_kwargs(Admin())
    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == DEFAULT_ADMIN_PORT


def test_a_host_without_a_port_gets_the_admin_port():
    assert connection_kwargs(Admin("10.0.0.1"))["port"] == DEFAULT_ADMIN_PORT


@pytest.mark.parametrize("kwargs", [{"port": 1}, {"db": 1}])
def test_forbidden_kwargs(kwargs):
    with pytest.raises(ConfigurationError):
        Admin(**kwargs)


@pytest.mark.parametrize("name", ["set", "lpush", "session", "bootstrap"])
def test_redis_and_client_commands_are_not_exposed(name):
    assert not hasattr(Admin(), name)


async def test_what_a_method_sends(monkeypatch):
    calls = []
    admin = Admin()

    async def record(*args, **options):
        calls.append((args, options))

    monkeypatch.setattr(admin, "execute_command", record)
    await admin.set_shard_status("BUCKET", "*", "READWRITE")
    assert calls == [
        (
            ("KR.ADMIN", "SET-SHARD-STATUS", "BUCKET", "*", "READWRITE"),
            {"admin_subcommand": "SET-SHARD-STATUS"},
        )
    ]


@pytest.mark.integration
async def test_ping(admin):
    assert await admin.ping() is True


@pytest.mark.integration
async def test_describe_cluster(admin):
    cluster = await admin.describe_cluster()
    shard = cluster["bucket"][0]
    assert shard["linked_volumes"] == ["bucket-shard-0"]
    assert await admin.describe_shard("BUCKET", 0) == shard


@pytest.mark.integration
async def test_list_and_find_member(admin):
    me = await admin.describe_member()
    member_id = me["member_id"]
    members = await admin.list_members()
    assert members[member_id]["process_id"] == me["process_id"]
    found = await admin.find_member(member_id[:4])
    assert found["process_id"] == me["process_id"]


@pytest.mark.integration
async def test_list_tasks(admin):
    tasks = await admin.list_tasks()
    assert isinstance(tasks, dict)
    for task in tasks.values():
        assert set(task) == {"running", "finished", "started_at", "last_run"}
        assert isinstance(task["running"], bool)
        assert isinstance(task["last_run"], int)


@pytest.mark.integration
async def test_list_silent_members(admin):
    assert isinstance(await admin.list_silent_members(), list)


@pytest.mark.integration
async def test_idempotent_status_writes(admin):
    assert await admin.set_shard_status("BUCKET", "*", "READWRITE") is True
    me = (await admin.describe_member())["member_id"]
    assert await admin.set_member_status(me, "RUNNING") is True


@pytest.mark.integration
async def test_errors_are_response_errors(admin):
    with pytest.raises(ResponseError, match="already been initialized"):
        await admin.initialize_cluster()
    with pytest.raises(ResponseError, match="cluster name does not match"):
        await admin.drop_cluster("kronotop-py-no-such-cluster")
