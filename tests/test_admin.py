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

"""``kronotop.Admin``, the ``KR.ADMIN`` family.

The integration tests run against a live dev cluster, so the writes are the
ones that change nothing there: a status the member or the shard already
has, and calls the server refuses before it acts.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop import DEFAULT_ADMIN_PORT, Admin, ConfigurationError
from kronotop._callbacks import parse_kr_admin, parse_task_admin

MEMBER = "006cdc459c59e600c76494e8388857fc3cba2fa8"
STANDBY = "a3f18b2e74d9c5601f82e4a7b390d612c8f7e149"

#: ``DESCRIBE-SHARD BUCKET 0`` as RESP2 hands it over: every map is an array.
SHARD_RESP2 = [
    b"primary",
    MEMBER.encode(),
    b"standbys",
    [STANDBY.encode()],
    b"status",
    b"READWRITE",
    b"linked_volumes",
    [b"bucket-shard-0"],
]

#: The same on RESP3.
SHARD_RESP3 = {
    b"primary": MEMBER.encode(),
    b"standbys": [STANDBY.encode()],
    b"status": b"READWRITE",
    b"linked_volumes": [b"bucket-shard-0"],
}

SHARD = {
    "primary": MEMBER,
    "standbys": [STANDBY],
    "status": "READWRITE",
    "linked_volumes": ["bucket-shard-0"],
}

MEMBER_RESP2 = [
    b"status",
    b"RUNNING",
    b"process_id",
    b"A1B2C3D4E5F6G7H8I9J0",
    b"external_advertise",
    [b"10.0.0.1:5484"],
    b"internal_advertise",
    [b"10.0.0.1:3320"],
    b"latest_heartbeat",
    31404,
]

MEMBER_INFO = {
    "status": "RUNNING",
    "process_id": "A1B2C3D4E5F6G7H8I9J0",
    "external_advertise": ["10.0.0.1:5484"],
    "internal_advertise": ["10.0.0.1:3320"],
    "latest_heartbeat": 31404,
}


def connection_kwargs(admin):
    return admin._redis.connection_pool.connection_kwargs


def test_default_host():
    kwargs = connection_kwargs(Admin())
    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == DEFAULT_ADMIN_PORT


def test_a_host_without_a_port_gets_the_admin_port():
    assert connection_kwargs(Admin("10.0.0.1"))["port"] == DEFAULT_ADMIN_PORT


def test_a_port_in_the_host_wins():
    assert connection_kwargs(Admin("10.0.0.1:1"))["port"] == 1


def test_transport_settings_pass_through():
    kwargs = connection_kwargs(Admin(protocol=3, socket_timeout=2))
    assert kwargs["protocol"] == 3
    assert kwargs["socket_timeout"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"port": 1}, {"connection_pool": object()}, {"db": 1}],
)
def test_forbidden_kwargs(kwargs):
    with pytest.raises(ConfigurationError):
        Admin(**kwargs)


def test_the_port_message_names_host_not_bootstrap():
    with pytest.raises(ConfigurationError, match="name the member in host"):
        Admin(port=1)


@pytest.mark.parametrize("name", ["set", "lpush", "acl_list", "session", "bootstrap"])
def test_redis_and_client_commands_are_not_exposed(name):
    assert not hasattr(Admin(), name)


@pytest.fixture
def sent(monkeypatch):
    """Record what a method sends instead of sending it."""
    calls = []
    admin = Admin()
    monkeypatch.setattr(
        admin, "execute_command", lambda *args, **options: calls.append((args, options))
    )
    return admin, calls


@pytest.mark.parametrize(
    ("call", "args", "phase"),
    [
        (lambda a: a.initialize_cluster(), ("INITIALIZE-CLUSTER",), None),
        (lambda a: a.describe_cluster(), ("DESCRIBE-CLUSTER",), None),
        (
            lambda a: a.describe_shard("BUCKET", 0),
            ("DESCRIBE-SHARD", "BUCKET", 0),
            None,
        ),
        (lambda a: a.describe_member(), ("DESCRIBE-MEMBER",), None),
        (lambda a: a.find_member("006c"), ("FIND-MEMBER", "006c"), None),
        (lambda a: a.list_members(), ("LIST-MEMBERS",), None),
        (lambda a: a.list_silent_members(), ("LIST-SILENT-MEMBERS",), None),
        (
            lambda a: a.set_member_status("006c", "STOPPED"),
            ("SET-MEMBER-STATUS", "006c", "STOPPED"),
            None,
        ),
        (
            lambda a: a.set_shard_status("BUCKET", "*", "READWRITE"),
            ("SET-SHARD-STATUS", "BUCKET", "*", "READWRITE"),
            None,
        ),
        (
            lambda a: a.set_shard_status("bucket", 2, "readonly"),
            ("SET-SHARD-STATUS", "bucket", "2", "readonly"),
            None,
        ),
        (
            lambda a: a.set_primary("BUCKET", 0, MEMBER),
            ("ROUTE", "SET", "PRIMARY", "BUCKET", "0", MEMBER),
            None,
        ),
        (
            lambda a: a.set_standby("BUCKET", "*", STANDBY),
            ("ROUTE", "SET", "STANDBY", "BUCKET", "*", STANDBY),
            None,
        ),
        (
            lambda a: a.unset_standby("BUCKET", 0, STANDBY),
            ("ROUTE", "UNSET", "STANDBY", "BUCKET", "0", STANDBY),
            None,
        ),
        (lambda a: a.remove_member("006c"), ("REMOVE-MEMBER", "006c"), None),
        (
            lambda a: a.drop_cluster("my-cluster"),
            ("DROP-CLUSTER", "my-cluster"),
            "DROP-CLUSTER-TOKEN",
        ),
        (
            lambda a: a.drop_cluster("my-cluster", "tok"),
            ("DROP-CLUSTER", "my-cluster", "tok"),
            None,
        ),
    ],
)
def test_what_each_method_sends(sent, call, args, phase):
    admin, calls = sent
    call(admin)
    assert calls == [
        (("KR.ADMIN", *args), {"admin_subcommand": phase or args[0]}),
    ]


def test_ping_sends_a_bare_ping(sent):
    admin, calls = sent
    admin.ping()
    assert calls == [(("PING",), {})]


def test_list_tasks_sends_task_admin(sent):
    admin, calls = sent
    admin.list_tasks()
    assert calls == [(("TASK.ADMIN", "LIST"), {})]


@pytest.mark.parametrize("reply", [SHARD_RESP2, SHARD_RESP3])
@pytest.mark.parametrize("subcommand", ["DESCRIBE-SHARD"])
def test_parse_shard(reply, subcommand):
    assert parse_kr_admin(reply, admin_subcommand=subcommand) == SHARD


@pytest.mark.parametrize("subcommand", ["FIND-MEMBER", "DESCRIBE-MEMBER"])
def test_parse_member(subcommand):
    assert parse_kr_admin(MEMBER_RESP2, admin_subcommand=subcommand) == MEMBER_INFO


def test_parse_list_members():
    reply = [MEMBER.encode(), MEMBER_RESP2]
    assert parse_kr_admin(reply, admin_subcommand="LIST-MEMBERS") == {
        MEMBER: MEMBER_INFO
    }


def test_parse_list_members_on_resp3():
    fields = dict(zip(MEMBER_RESP2[::2], MEMBER_RESP2[1::2], strict=True))
    reply = {MEMBER.encode(): fields}
    assert parse_kr_admin(reply, admin_subcommand="LIST-MEMBERS") == {
        MEMBER: MEMBER_INFO
    }


def test_parse_describe_cluster():
    resp2 = [
        b"metadata_version",
        b"1.0.0",
        b"cluster_name",
        b"my-cluster",
        b"bucket",
        [0, SHARD_RESP2, 1, SHARD_RESP2],
    ]
    resp3 = {
        b"metadata_version": b"1.0.0",
        b"cluster_name": b"my-cluster",
        b"bucket": {0: SHARD_RESP3, 1: SHARD_RESP3},
    }
    expected = {
        "metadata_version": "1.0.0",
        "cluster_name": "my-cluster",
        "bucket": {0: SHARD, 1: SHARD},
    }
    for reply in (resp2, resp3):
        assert parse_kr_admin(reply, admin_subcommand="DESCRIBE-CLUSTER") == expected


def test_parse_silent_members():
    reply = [STANDBY.encode()]
    assert parse_kr_admin(reply, admin_subcommand="LIST-SILENT-MEMBERS") == [STANDBY]


def test_parse_drop_cluster_phases():
    assert parse_kr_admin(b"a1b2", admin_subcommand="DROP-CLUSTER-TOKEN") == "a1b2"
    assert parse_kr_admin(b"OK", admin_subcommand="DROP-CLUSTER") is True


TASK = {"running": False, "finished": False, "started_at": 1752582119, "last_run": 0}
TASK_RESP2 = [
    b"journal:cleanup-task",
    [b"running", 0, b"finished", 0, b"started_at", 1752582119, b"last_run", 0],
]
TASK_RESP3 = {
    b"journal:cleanup-task": {
        b"running": False,
        b"finished": False,
        b"started_at": 1752582119,
        b"last_run": 0,
    }
}


@pytest.mark.parametrize("reply", [TASK_RESP2, TASK_RESP3])
def test_parse_tasks(reply):
    assert parse_task_admin(reply) == {"journal:cleanup-task": TASK}


def test_parse_no_tasks():
    assert parse_task_admin([]) == {}


def test_parse_ok_and_raw_replies():
    assert parse_kr_admin(b"OK", admin_subcommand="SET-SHARD-STATUS") is True
    # A raw caller sets no option: ``OK`` still folds, a map passes through.
    assert parse_kr_admin(b"OK") is True
    assert parse_kr_admin(SHARD_RESP2) is SHARD_RESP2


@pytest.mark.integration
def test_ping(admin):
    assert admin.ping() is True


@pytest.mark.integration
def test_describe_cluster(admin):
    cluster = admin.describe_cluster()
    assert isinstance(cluster["metadata_version"], str)
    assert isinstance(cluster["cluster_name"], str)
    shard = cluster["bucket"][0]
    assert isinstance(shard["primary"], str)
    assert isinstance(shard["standbys"], list)
    assert shard["status"] in ("READWRITE", "READONLY", "INOPERABLE")
    assert shard["linked_volumes"] == ["bucket-shard-0"]


@pytest.mark.integration
def test_describe_shard_matches_the_cluster(admin):
    assert admin.describe_shard("BUCKET", 0) == admin.describe_cluster()["bucket"][0]


@pytest.mark.integration
def test_describe_shard_refuses_an_unknown_kind(admin):
    with pytest.raises(ResponseError, match="Unknown shard kind"):
        admin.describe_shard("FOOBAR", 0)


@pytest.mark.integration
def test_describe_member(admin):
    me = admin.describe_member()
    assert len(me["member_id"]) == 40
    assert int(me["member_id"], 16)
    assert me["status"] == "RUNNING"
    assert isinstance(me["latest_heartbeat"], int)
    assert me["external_advertise"] and me["internal_advertise"]


@pytest.mark.integration
def test_list_and_find_member(admin):
    me = admin.describe_member()
    member_id = me["member_id"]
    members = admin.list_members()
    assert member_id in members
    assert "member_id" not in members[member_id]
    assert members[member_id]["process_id"] == me["process_id"]
    assert admin.find_member(member_id)["process_id"] == me["process_id"]
    assert admin.find_member(member_id[:4])["process_id"] == me["process_id"]


@pytest.mark.integration
def test_find_member_refuses_a_bad_id(admin):
    with pytest.raises(ResponseError, match="Invalid memberId"):
        admin.find_member("abc")


@pytest.mark.integration
def test_list_tasks(admin):
    tasks = admin.list_tasks()
    assert isinstance(tasks, dict)
    for task in tasks.values():
        assert set(task) == set(TASK)
        assert isinstance(task["running"], bool)
        assert isinstance(task["finished"], bool)
        assert isinstance(task["started_at"], int)
        assert isinstance(task["last_run"], int)


@pytest.mark.integration
def test_list_silent_members(admin):
    assert isinstance(admin.list_silent_members(), list)


@pytest.mark.integration
def test_initialize_cluster_runs_once(admin):
    with pytest.raises(ResponseError, match="already been initialized"):
        admin.initialize_cluster()


@pytest.mark.integration
def test_idempotent_status_writes(admin):
    """The statuses a healthy dev cluster already has, so nothing changes."""
    assert admin.set_shard_status("BUCKET", "*", "READWRITE") is True
    me = admin.describe_member()["member_id"]
    assert admin.set_member_status(me, "RUNNING") is True


@pytest.mark.integration
def test_set_shard_status_refuses_an_unknown_status(admin):
    with pytest.raises(ResponseError, match="Unknown shard status"):
        admin.set_shard_status("BUCKET", 0, "DISABLED")


@pytest.mark.integration
def test_the_primary_cannot_be_a_standby(admin):
    primary = admin.describe_shard("BUCKET", 0)["primary"]
    with pytest.raises(ResponseError, match="primary cannot be assigned as a standby"):
        admin.set_standby("BUCKET", 0, primary)


@pytest.mark.integration
def test_a_running_member_cannot_be_removed(admin):
    me = admin.describe_member()["member_id"]
    with pytest.raises(ResponseError, match="RUNNING status cannot be removed"):
        admin.remove_member(me)


@pytest.mark.integration
def test_drop_cluster_checks_the_name(admin):
    """A wrong name fails before a token is issued, so the real name is never sent."""
    with pytest.raises(ResponseError, match="cluster name does not match"):
        admin.drop_cluster("kronotop-py-no-such-cluster")
