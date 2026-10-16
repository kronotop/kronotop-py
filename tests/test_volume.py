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

"""``admin.volume``: ``VOLUME.ADMIN`` and ``VOLUME.INSPECT``.

The parser tests live here only. ``tests/test_asyncio/test_volume.py``
covers the async twin's wiring and the integration path.
"""

from __future__ import annotations

import struct

import pytest
from redis.exceptions import ResponseError

from kronotop import Admin
from kronotop._callbacks import (
    parse_changelog_range,
    parse_segment_tail_pointer,
    parse_volume_admin,
    parse_volume_inspect,
    parse_volume_stats,
)
from kronotop.commands.volume import Volume

MEMBER = "006cdc459c59e600c76494e8388857fc3cba2fa8"
VOLUME = "bucket-shard-0"
KINDS = {"APPEND", "DELETE", "UPDATE", "INSERT"}

#: ``DESCRIBE`` as RESP2 hands it over: every map is an array, the double is
#: its 8 byte big-endian form.
DESCRIBE_RESP2 = [
    b"name",
    VOLUME.encode(),
    b"status",
    b"READWRITE",
    b"data_dir",
    b"/tmp/kronotop/data/bucket-shard-0",
    b"segment_size",
    268435456,
    b"segments",
    [
        0,
        [
            b"size",
            268435456,
            b"free_bytes",
            268435200,
            b"used_bytes",
            256,
            b"garbage_percentage",
            struct.pack(">d", 12.5),
            b"cardinality",
            1,
        ],
    ],
]

#: The same on RESP3.
DESCRIBE_RESP3 = {
    b"name": VOLUME.encode(),
    b"status": b"READWRITE",
    b"data_dir": b"/tmp/kronotop/data/bucket-shard-0",
    b"segment_size": 268435456,
    b"segments": {
        0: {
            b"size": 268435456,
            b"free_bytes": 268435200,
            b"used_bytes": 256,
            b"garbage_percentage": 12.5,
            b"cardinality": 1,
        }
    },
}

DESCRIBED = {
    "name": VOLUME,
    "status": "READWRITE",
    "data_dir": "/tmp/kronotop/data/bucket-shard-0",
    "segment_size": 268435456,
    "segments": {
        0: {
            "size": 268435456,
            "free_bytes": 268435200,
            "used_bytes": 256,
            "garbage_percentage": 12.5,
            "cardinality": 1,
        }
    },
}

SEGMENT_RESP2 = [
    b"segment_id",
    0,
    b"status",
    b"COMPLETED",
    b"started_at",
    1715100001000,
]
SEGMENT = {"segment_id": 0, "status": "COMPLETED", "started_at": 1715100001000}

METADATA_RESP2 = [
    b"started_at",
    1715100000000,
    b"completed_at",
    1715100060000,
    b"result",
    b"COMPLETED",
    b"segments_processed",
    1,
]
METADATA = {
    "started_at": 1715100000000,
    "completed_at": 1715100060000,
    "result": "COMPLETED",
    "segments_processed": 1,
}

LOCATE_RESP2 = [
    b"member_id",
    MEMBER.encode(),
    b"process_id",
    b"A1B2C3D4E5F6G7H8I9J0",
    b"external_advertise",
    [b"10.0.0.1:5484"],
    b"internal_advertise",
    [b"10.0.0.1:3320"],
]
LOCATED = {
    "member_id": MEMBER,
    "process_id": "A1B2C3D4E5F6G7H8I9J0",
    "external_advertise": ["10.0.0.1:5484"],
    "internal_advertise": ["10.0.0.1:3320"],
}

CURSOR_RESP2 = [
    b"active_segment_id",
    0,
    b"versionstamp",
    b"A1B2C3D4E5F6G7H8I9J0",
    b"next_position",
    10240,
    b"sequence_number",
    9,
]
CURSOR = {
    "active_segment_id": 0,
    "versionstamp": "A1B2C3D4E5F6G7H8I9J0",
    "next_position": 10240,
    "sequence_number": 9,
}

REPLICATION_RESP2 = [
    b"stage",
    b"CHANGE_DATA_CAPTURE",
    b"cursor",
    [b"segment_id", 2, b"position", 8192],
    b"status",
    b"RUNNING",
    b"error_message",
    b"",
    b"cdc_stage",
    [b"sequence_number", 15, b"position", 4096],
    b"segment_replication_stage",
    [b"tail_sequence_number", 10, b"tail_next_position", 6144],
]
REPLICATION_RESP3 = {
    b"stage": b"CHANGE_DATA_CAPTURE",
    b"cursor": {b"segment_id": 2, b"position": 8192},
    b"status": b"RUNNING",
    b"error_message": b"",
    b"cdc_stage": {b"sequence_number": 15, b"position": 4096},
    b"segment_replication_stage": {
        b"tail_sequence_number": 10,
        b"tail_next_position": 6144,
    },
}
REPLICATION = {
    "stage": "CHANGE_DATA_CAPTURE",
    "cursor": {"segment_id": 2, "position": 8192},
    "status": "RUNNING",
    "error_message": "",
    "cdc_stage": {"sequence_number": 15, "position": 4096},
    "segment_replication_stage": {
        "tail_sequence_number": 10,
        "tail_next_position": 6144,
    },
}


def test_the_group_is_bound():
    admin = Admin()
    assert isinstance(admin.volume, Volume)
    assert admin.volume._is_async_client is False


@pytest.fixture
def sent(monkeypatch):
    """Record what a method sends instead of sending it. The group forwards
    to the admin client, so the patch goes there."""
    calls = []
    admin = Admin()
    monkeypatch.setattr(
        admin, "execute_command", lambda *args, **options: calls.append((args, options))
    )
    return admin.volume, calls


@pytest.mark.parametrize(
    ("call", "args", "phase"),
    [
        (lambda v: v.list(), ("LIST",), None),
        (lambda v: v.describe(VOLUME), ("DESCRIBE", VOLUME), None),
        (
            lambda v: v.set_status(VOLUME, "READONLY"),
            ("SET-STATUS", VOLUME, "READONLY"),
            None,
        ),
        (lambda v: v.list_segments(VOLUME), ("LIST-SEGMENTS", VOLUME), None),
        (
            lambda v: v.cleanup_orphan_files(VOLUME),
            ("CLEANUP-ORPHAN-FILES", VOLUME),
            None,
        ),
        (
            lambda v: v.prune_changelog(VOLUME, 24),
            ("PRUNE-CHANGELOG", VOLUME, "24"),
            None,
        ),
        (
            lambda v: v.start_replication(VOLUME),
            ("REPLICATION", "START", VOLUME),
            None,
        ),
        (
            lambda v: v.stop_replication(VOLUME),
            ("REPLICATION", "STOP", VOLUME),
            None,
        ),
        (
            lambda v: v.vacuum_start(VOLUME, 30),
            ("VACUUM", "START", VOLUME, "30"),
            None,
        ),
        (
            lambda v: v.vacuum_start(VOLUME, 12.5),
            ("VACUUM", "START", VOLUME, "12.5"),
            None,
        ),
        (lambda v: v.vacuum_stop(VOLUME), ("VACUUM", "STOP", VOLUME), None),
        (lambda v: v.vacuum_drop(VOLUME), ("VACUUM", "DROP", VOLUME), None),
        (
            lambda v: v.vacuum_status(VOLUME),
            ("VACUUM", "STATUS", VOLUME),
            "VACUUM STATUS",
        ),
        (
            lambda v: v.mark_stale_prefixes("START"),
            ("MARK-STALE-PREFIXES", "START"),
            None,
        ),
        (
            lambda v: v.locate_stale_prefixes_task(),
            ("MARK-STALE-PREFIXES", "LOCATE"),
            "LOCATE",
        ),
    ],
)
def test_what_each_admin_method_sends(sent, call, args, phase):
    volume, calls = sent
    call(volume)
    assert calls == [
        (("VOLUME.ADMIN", *args), {"volume_subcommand": phase or args[0]}),
    ]


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (
            lambda v: v.segment_range(VOLUME, 3, (0, 16), (32, 8)),
            ("SEGMENT.RANGE", VOLUME, "3", "0", "16", "32", "8"),
        ),
        (
            lambda v: v.segment_tail_pointer(VOLUME, 3),
            ("SEGMENT.TAILPOINTER", VOLUME, "3"),
        ),
        (
            lambda v: v.changelog_range(VOLUME, 0, 10),
            ("CHANGELOG.RANGE", VOLUME, "*", "[0", "10]"),
        ),
        (
            lambda v: v.changelog_range(
                VOLUME,
                0,
                10,
                kind="LIFECYCLE",
                limit=5,
                reverse=True,
                start_inclusive=False,
                end_inclusive=False,
            ),
            (
                "CHANGELOG.RANGE",
                VOLUME,
                "LIFECYCLE",
                "(0",
                "10)",
                "LIMIT",
                "5",
                "REVERSE",
            ),
        ),
        (
            lambda v: v.changelog_watch(VOLUME, 7),
            ("CHANGELOG.WATCH", VOLUME, "7"),
        ),
    ],
)
def test_what_each_replication_method_sends(sent, call, args):
    volume, calls = sent
    call(volume)
    assert calls == [(args, {})]


def test_what_each_inspect_method_sends(sent):
    volume, calls = sent
    volume.cursor(VOLUME)
    volume.replication_status(VOLUME, "006c")
    assert calls == [
        (("VOLUME.INSPECT", "CURSOR", VOLUME), {"inspect_subcommand": "CURSOR"}),
        (
            ("VOLUME.INSPECT", "REPLICATION", VOLUME, "006c"),
            {"inspect_subcommand": "REPLICATION"},
        ),
    ]


@pytest.mark.parametrize(
    ("call", "args", "phase"),
    [
        (lambda v: v.stats(VOLUME), (), "OVERVIEW"),
        (lambda v: v.stats_opcounters(VOLUME), ("OPCOUNTERS",), "OPCOUNTERS"),
        (lambda v: v.stats_segments(VOLUME), ("SEGMENTS",), "SEGMENTS"),
        (
            lambda v: v.stats_replication(VOLUME, "006c"),
            ("REPLICATION", "006c"),
            "REPLICATION",
        ),
        (lambda v: v.stats_reset(VOLUME), ("RESET",), "RESET"),
    ],
)
def test_what_each_stats_method_sends(sent, call, args, phase):
    volume, calls = sent
    call(volume)
    assert calls == [
        (("VOLUME.STATS", VOLUME, *args), {"stats_subcommand": phase}),
    ]


@pytest.mark.parametrize("reply", [DESCRIBE_RESP2, DESCRIBE_RESP3])
def test_parse_describe(reply):
    assert parse_volume_admin(reply, volume_subcommand="DESCRIBE") == DESCRIBED


def test_parse_list_and_orphan_files():
    reply = [VOLUME.encode(), b"bucket-shard-1"]
    expected = [VOLUME, "bucket-shard-1"]
    assert parse_volume_admin(reply, volume_subcommand="LIST") == expected
    assert (
        parse_volume_admin(reply, volume_subcommand="CLEANUP-ORPHAN-FILES") == expected
    )


def test_parse_list_segments_keeps_the_integers():
    assert parse_volume_admin([0, 1], volume_subcommand="LIST-SEGMENTS") == [0, 1]


def test_parse_vacuum_status_of_a_finished_run():
    resp2 = [b"active", 0, b"metadata", METADATA_RESP2, b"segments", [SEGMENT_RESP2]]
    resp3 = {
        b"active": False,
        b"metadata": dict(zip(METADATA_RESP2[::2], METADATA_RESP2[1::2], strict=True)),
        b"segments": [dict(zip(SEGMENT_RESP2[::2], SEGMENT_RESP2[1::2], strict=True))],
    }
    expected = {"active": False, "metadata": METADATA, "segments": [SEGMENT]}
    for reply in (resp2, resp3):
        assert parse_volume_admin(reply, volume_subcommand="VACUUM STATUS") == expected


def test_parse_vacuum_status_of_a_running_one_has_no_metadata():
    reply = [b"active", 1, b"segments", [SEGMENT_RESP2]]
    assert parse_volume_admin(reply, volume_subcommand="VACUUM STATUS") == {
        "active": True,
        "segments": [SEGMENT],
    }


def test_parse_locate():
    assert parse_volume_admin(LOCATE_RESP2, volume_subcommand="LOCATE") == LOCATED


def test_parse_ok_and_raw_replies():
    assert parse_volume_admin(b"OK", volume_subcommand="SET-STATUS") is True
    assert parse_volume_admin(b"OK", volume_subcommand="VACUUM") is True
    # A raw caller sets no option: ``OK`` still folds, a map passes through.
    assert parse_volume_admin(b"OK") is True
    assert parse_volume_admin(DESCRIBE_RESP2) is DESCRIBE_RESP2


def test_parse_cursor():
    resp3 = dict(zip(CURSOR_RESP2[::2], CURSOR_RESP2[1::2], strict=True))
    for reply in (CURSOR_RESP2, resp3):
        assert parse_volume_inspect(reply, inspect_subcommand="CURSOR") == CURSOR


@pytest.mark.parametrize("reply", [REPLICATION_RESP2, REPLICATION_RESP3])
def test_parse_replication(reply):
    assert parse_volume_inspect(reply, inspect_subcommand="REPLICATION") == REPLICATION


def test_parse_segment_tail_pointer():
    assert parse_segment_tail_pointer([256, 12]) == {
        "next_position": 256,
        "sequence_number": 12,
    }


AFTER = {"sequence_number": 1, "segment_id": 0, "position": 0, "length": 64}
BEFORE = {"sequence_number": 2, "segment_id": 0, "position": 64, "length": 64}
ENTRIES = [
    {"versionstamp": "0000000000000001", "kind": "APPEND", "prefix": 7, "after": AFTER},
    {
        "versionstamp": "0000000000000002",
        "kind": "UPDATE",
        "prefix": 7,
        "before": BEFORE,
        "after": AFTER,
    },
]


def _flat(fields):
    """A map as RESP2 hands it over: keys as bytes, nested maps flattened."""
    out = []
    for key, value in fields.items():
        out.append(key.encode())
        if isinstance(value, dict):
            out.append(_flat(value))
        elif isinstance(value, str):
            out.append(value.encode())
        else:
            out.append(value)
    return out


def _resp3(fields):
    """The same map as RESP3 hands it over: keys and text as bytes."""
    out = {}
    for key, value in fields.items():
        if isinstance(value, dict):
            value = _resp3(value)
        elif isinstance(value, str):
            value = value.encode()
        out[key.encode()] = value
    return out


@pytest.mark.parametrize(
    "reply", [[_flat(e) for e in ENTRIES], [_resp3(e) for e in ENTRIES]]
)
def test_parse_changelog_range(reply):
    assert parse_changelog_range(reply) == ENTRIES


def test_parse_an_empty_changelog_range():
    assert parse_changelog_range([]) == []


def test_parse_inspect_raw_reply_passes_through():
    assert parse_volume_inspect(CURSOR_RESP2) is CURSOR_RESP2


OVERVIEW = {
    "status": "READWRITE",
    "segment_count": 1,
    "total_size_bytes": 1048576,
    "used_bytes": 100,
    "free_bytes": 1048456,
    "garbage_bytes": 20,
    "garbage_percentage": 0.25,
    "fill_ratio": 0.5,
    "total_cardinality": 10,
}
SEGMENT_STATS = {
    "segment_id": 0,
    "size_bytes": 1048576,
    "used_bytes": 100,
    "free_bytes": 1048456,
    "garbage_bytes": 20,
    "garbage_percentage": 0.25,
    "cardinality": 10,
}
OPCOUNTERS = {
    "appends": 50,
    "deletes": 5,
    "updates": 10,
    "gets": 30,
    "bytes_appended": 6000,
    "bytes_read": 3000,
    "segments_created": 1,
}
REPLICATION_STATS = {
    "stage": "CHANGE_DATA_CAPTURE",
    "status": "RUNNING",
    "error_message": "",
    "cursor_segment_id": 3,
    "cursor_position": 0,
    "cdc_sequence_number": 42,
    "cdc_position": 1024,
}


def _doubles_resp2(fields):
    """``_flat``, with each double as its 8 byte big-endian form."""
    packed = {
        key: struct.pack(">d", value) if isinstance(value, float) else value
        for key, value in fields.items()
    }
    return _flat(packed)


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("OVERVIEW", OVERVIEW),
        ("OPCOUNTERS", OPCOUNTERS),
        ("REPLICATION", REPLICATION_STATS),
    ],
)
def test_parse_stats_map(phase, expected):
    for reply in (_doubles_resp2(expected), _resp3(expected)):
        assert parse_volume_stats(reply, stats_subcommand=phase) == expected


def test_parse_stats_segments():
    for reply in ([_doubles_resp2(SEGMENT_STATS)], [_resp3(SEGMENT_STATS)]):
        assert parse_volume_stats(reply, stats_subcommand="SEGMENTS") == [SEGMENT_STATS]
    assert parse_volume_stats([], stats_subcommand="SEGMENTS") == []


def test_parse_stats_ok_and_raw_replies():
    assert parse_volume_stats(b"OK", stats_subcommand="RESET") is True
    assert parse_volume_stats(b"OK") is True
    raw = _flat(OPCOUNTERS)
    assert parse_volume_stats(raw) is raw


@pytest.mark.integration
def test_list(admin):
    assert VOLUME in admin.volume.list()


@pytest.mark.integration
def test_describe_matches_list_segments(admin):
    volume = admin.volume.describe(VOLUME)
    assert volume["name"] == VOLUME
    assert volume["status"] in ("READWRITE", "READONLY", "INOPERABLE")
    assert isinstance(volume["data_dir"], str)
    assert isinstance(volume["segment_size"], int)
    # The two subcommands walk the segments in different orders.
    assert sorted(volume["segments"]) == sorted(admin.volume.list_segments(VOLUME))
    for segment in volume["segments"].values():
        assert isinstance(segment["garbage_percentage"], float)
        assert isinstance(segment["cardinality"], int)


@pytest.mark.integration
def test_stats(admin):
    stats = admin.volume.stats(VOLUME)
    assert stats["status"] in ("READWRITE", "READONLY", "INOPERABLE")
    assert isinstance(stats["garbage_percentage"], float)
    assert isinstance(stats["fill_ratio"], float)
    assert (
        stats["total_size_bytes"] - stats["free_bytes"]
        == stats["used_bytes"] + stats["garbage_bytes"]
    )


@pytest.mark.integration
def test_stats_segments_match_list_segments(admin):
    segments = admin.volume.stats_segments(VOLUME)
    ids = [segment["segment_id"] for segment in segments]
    assert sorted(ids) == sorted(admin.volume.list_segments(VOLUME))
    for segment in segments:
        assert isinstance(segment["garbage_percentage"], float)
        assert (
            segment["size_bytes"] - segment["free_bytes"]
            == segment["used_bytes"] + segment["garbage_bytes"]
        )


@pytest.mark.integration
def test_stats_opcounters_and_reset(admin):
    counters = admin.volume.stats_opcounters(VOLUME)
    assert counters.keys() == OPCOUNTERS.keys()
    assert all(isinstance(value, int) for value in counters.values())
    assert admin.volume.stats_reset(VOLUME) is True


@pytest.mark.integration
def test_stats_errors(admin):
    with pytest.raises(ResponseError, match="Invalid memberId"):
        admin.volume.stats_replication(VOLUME, "abc")
    with pytest.raises(ResponseError, match="is not open"):
        admin.volume.stats("no-such-volume")


@pytest.mark.integration
def test_cursor(admin):
    cursor = admin.volume.cursor(VOLUME)
    assert isinstance(cursor["active_segment_id"], int)
    assert isinstance(cursor["versionstamp"], str)
    assert isinstance(cursor["next_position"], int)
    assert isinstance(cursor["sequence_number"], int)


@pytest.mark.integration
def test_set_status_is_idempotent(admin):
    assert admin.volume.set_status(VOLUME, "READWRITE") is True


@pytest.mark.integration
def test_an_unknown_volume_is_not_open(admin):
    with pytest.raises(ResponseError, match="is not open"):
        admin.volume.describe("no-such-volume")


@pytest.mark.integration
def test_an_unknown_status_is_refused(admin):
    with pytest.raises(ResponseError, match="Unknown volume status"):
        admin.volume.set_status(VOLUME, "DISABLED")


@pytest.mark.integration
def test_vacuum_status(admin):
    """Whether a run ever happened is the server's state, so both answers pass."""
    try:
        status = admin.volume.vacuum_status(VOLUME)
    except ResponseError as exc:
        assert "No active vacuum" in str(exc)
    else:
        assert isinstance(status["active"], bool)
        assert isinstance(status["segments"], list)


@pytest.mark.integration
def test_replication_needs_a_standby(admin):
    me = admin.describe_member()["member_id"]
    if me in admin.describe_shard("BUCKET", 0)["standbys"]:
        pytest.skip("this member is a standby of bucket-shard-0")
    with pytest.raises(ResponseError, match="not a standby"):
        admin.volume.start_replication(VOLUME)


@pytest.mark.integration
def test_replication_status_refuses_a_bad_id(admin):
    with pytest.raises(ResponseError, match="Invalid memberId"):
        admin.volume.replication_status(VOLUME, "abc")


@pytest.mark.integration
def test_segment_tail_pointer(admin):
    segment = admin.volume.cursor(VOLUME)["active_segment_id"]
    pointer = admin.volume.segment_tail_pointer(VOLUME, segment)
    assert isinstance(pointer["next_position"], int)
    assert isinstance(pointer["sequence_number"], int)
    assert pointer["next_position"] >= 0


@pytest.mark.integration
def test_segment_range_reads_the_head(admin):
    segment = admin.volume.cursor(VOLUME)["active_segment_id"]
    size = admin.volume.describe(VOLUME)["segment_size"]
    with pytest.raises(ResponseError, match="OUTOFBOUND"):
        admin.volume.segment_range(VOLUME, segment, (size, 1))
    tail = admin.volume.segment_tail_pointer(VOLUME, segment)["next_position"]
    if tail == 0:
        pytest.skip("the active segment is empty")
    n = min(tail, 16)
    (head,) = admin.volume.segment_range(VOLUME, segment, (0, n))
    assert isinstance(head, bytes)
    assert len(head) == n


@pytest.mark.integration
def test_segment_errors(admin):
    with pytest.raises(ResponseError, match="could not be found"):
        admin.volume.segment_range(VOLUME, 2**40, (0, 1))
    with pytest.raises(ResponseError, match="is not open"):
        admin.volume.segment_range("no-such-volume", 0, (0, 1))


@pytest.mark.integration
def test_changelog_range(admin):
    entries = admin.volume.changelog_range(VOLUME, 0, 2**62)
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry["versionstamp"], str)
        assert entry["kind"] in KINDS
        assert isinstance(entry["prefix"], int)
        assert "before" in entry or "after" in entry
    assert len(admin.volume.changelog_range(VOLUME, 0, 2**62, limit=1)) <= 1


@pytest.mark.integration
def test_changelog_range_refuses_a_bad_range(admin):
    with pytest.raises(ResponseError):
        admin.volume.changelog_range(VOLUME, 10, 0)
    with pytest.raises(ResponseError):
        admin.volume.changelog_range(VOLUME, 0, 10, kind="FOO")


@pytest.mark.integration
def test_changelog_watch_refuses_a_future_sequence(admin):
    with pytest.raises(ResponseError, match="cannot be greater"):
        admin.volume.changelog_watch(VOLUME, 2**62)
