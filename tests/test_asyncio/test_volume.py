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

"""``admin.volume`` on ``kronotop.asyncio.Admin``.

The parser and the argument wiring are shared with the sync client and are
covered in ``tests/test_volume.py``.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop.asyncio import Admin
from kronotop.commands.volume import AsyncVolume

VOLUME = "bucket-shard-0"


def test_the_group_is_bound():
    admin = Admin()
    assert isinstance(admin.volume, AsyncVolume)
    assert admin.volume._is_async_client is True


async def test_what_a_method_sends(monkeypatch):
    calls = []
    admin = Admin()

    async def record(*args, **options):
        calls.append((args, options))

    monkeypatch.setattr(admin, "execute_command", record)
    await admin.volume.vacuum_status(VOLUME)
    await admin.volume.cursor(VOLUME)
    await admin.volume.changelog_range(VOLUME, 0, 10)
    await admin.volume.stats(VOLUME)
    assert calls == [
        (
            ("VOLUME.ADMIN", "VACUUM", "STATUS", VOLUME),
            {"volume_subcommand": "VACUUM STATUS"},
        ),
        (("VOLUME.INSPECT", "CURSOR", VOLUME), {"inspect_subcommand": "CURSOR"}),
        (("CHANGELOG.RANGE", VOLUME, "*", "[0", "10]"), {}),
        (("VOLUME.STATS", VOLUME), {"stats_subcommand": "OVERVIEW"}),
    ]


@pytest.mark.integration
async def test_list(admin):
    assert VOLUME in await admin.volume.list()


@pytest.mark.integration
async def test_describe_matches_list_segments(admin):
    volume = await admin.volume.describe(VOLUME)
    assert volume["name"] == VOLUME
    segment_ids = await admin.volume.list_segments(VOLUME)
    assert sorted(volume["segments"]) == sorted(segment_ids)


@pytest.mark.integration
async def test_cursor(admin):
    cursor = await admin.volume.cursor(VOLUME)
    assert isinstance(cursor["active_segment_id"], int)
    assert isinstance(cursor["versionstamp"], str)


@pytest.mark.integration
async def test_stats(admin):
    stats = await admin.volume.stats(VOLUME)
    assert (
        stats["total_size_bytes"] - stats["free_bytes"]
        == stats["used_bytes"] + stats["garbage_bytes"]
    )


@pytest.mark.integration
async def test_stats_reset(admin):
    assert await admin.volume.stats_reset(VOLUME) is True


@pytest.mark.integration
async def test_set_status_is_idempotent(admin):
    assert await admin.volume.set_status(VOLUME, "READWRITE") is True


@pytest.mark.integration
async def test_segment_tail_pointer(admin):
    segment = (await admin.volume.cursor(VOLUME))["active_segment_id"]
    pointer = await admin.volume.segment_tail_pointer(VOLUME, segment)
    assert isinstance(pointer["next_position"], int)
    assert isinstance(pointer["sequence_number"], int)


@pytest.mark.integration
async def test_changelog_range(admin):
    entries = await admin.volume.changelog_range(VOLUME, 0, 2**62, limit=1)
    assert isinstance(entries, list)
    assert len(entries) <= 1


@pytest.mark.integration
async def test_errors_are_response_errors(admin):
    with pytest.raises(ResponseError, match="is not open"):
        await admin.volume.describe("no-such-volume")
    with pytest.raises(ResponseError, match="Unknown volume status"):
        await admin.volume.set_status(VOLUME, "DISABLED")
