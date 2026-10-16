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

"""``VOLUME.ADMIN``, ``VOLUME.INSPECT`` and ``VOLUME.STATS`` commands, bound as
``admin.volume``.

A volume is the local storage of one shard, named after it, such as
``bucket-shard-0``. The server answers these commands on the internal
listener only, port 3320, and only for the volumes open on the member asked.
``list`` gives their names.

The replication commands ``SEGMENT.RANGE``, ``SEGMENT.TAILPOINTER``,
``CHANGELOG.RANGE`` and ``CHANGELOG.WATCH`` are here too. They have no page
in the server documentation, so the server handlers and their tests are the
source of truth.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Literal, overload

from redis.typing import EncodableT

from kronotop.commands.base import AsyncClient, CommandGroup, CommandsBase, SyncClient

__all__ = [
    "AsyncVolume",
    "ChangeLogEntry",
    "ChangeLogKind",
    "SegmentInfo",
    "StalePrefixesOperation",
    "TailPointer",
    "VacuumStatus",
    "Volume",
    "VolumeCommands",
    "VolumeInfo",
    "VolumeStatus",
]

#: One segment: ``size``, ``free_bytes``, ``used_bytes`` and ``cardinality``
#: (``int``) and ``garbage_percentage`` (``float``).
SegmentInfo = dict[str, Any]

#: One volume: ``name``, ``status`` (a :data:`VolumeStatus`), ``data_dir``,
#: ``segment_size`` (``int``) and ``segments`` (segment id -> :data:`SegmentInfo`).
VolumeInfo = dict[str, Any]

#: A vacuum run: ``active`` (``bool``), ``segments`` (a list of maps with
#: ``segment_id``, ``status`` and ``started_at``) and ``metadata``
#: (``started_at``, ``completed_at``, ``result``, ``segments_processed``).
#: ``metadata`` is missing while the run is still going.
VacuumStatus = dict[str, Any]

#: ``READWRITE`` takes reads and writes, ``READONLY`` refuses writes,
#: ``INOPERABLE`` refuses everything. The server matches the value without
#: regard to case.
VolumeStatus = Literal["READWRITE", "READONLY", "INOPERABLE"]

StalePrefixesOperation = Literal["START", "STOP", "REMOVE"]

#: The parent operation kind a changelog entry belongs to.
ChangeLogKind = Literal["LIFECYCLE", "FINALIZATION"]

#: One changelog entry: ``versionstamp`` (``str``), ``kind`` (``APPEND``,
#: ``DELETE``, ``UPDATE`` or ``INSERT``), ``prefix`` (``int``), and
#: ``before`` and/or ``after``, each a map of the ``int`` fields
#: ``sequence_number``, ``segment_id``, ``position`` and ``length``.
ChangeLogEntry = dict[str, Any]

#: ``next_position`` and ``sequence_number`` of one segment.
TailPointer = dict[str, int]

#: Spelled out here because inside the class ``list`` is the method, and an
#: IDE reads the annotation that way.
_Names = list[str]
_SegmentIds = list[int]
_Chunks = list[bytes]
_Entries = list[ChangeLogEntry]
_SegmentStats = list[dict[str, Any]]


class VolumeCommands(CommandsBase):
    """``VOLUME.ADMIN``, ``VOLUME.INSPECT`` and ``VOLUME.STATS``.

    Every error the server raises here is a plain ``ERR``, so it arrives as a
    ``redis.ResponseError``. A volume the member has not opened answers
    ``Volume: '<name>' is not open``.
    """

    def _volume(self, subcommand: str, *args: Any, phase: str | None = None) -> Any:
        """Send one subcommand. ``phase`` overrides the callback key."""
        return self.execute_command(
            "VOLUME.ADMIN", subcommand, *args, volume_subcommand=phase or subcommand
        )

    @overload
    def list(self: SyncClient) -> _Names: ...

    @overload
    def list(self: AsyncClient) -> Awaitable[_Names]: ...

    def list(self) -> Any:
        """``VOLUME.ADMIN LIST``. The names of the volumes this member has open."""
        return self._volume("LIST")

    @overload
    def describe(self: SyncClient, name: str) -> VolumeInfo: ...

    @overload
    def describe(self: AsyncClient, name: str) -> Awaitable[VolumeInfo]: ...

    def describe(self, name: str) -> Any:
        """``VOLUME.ADMIN DESCRIBE``. The volume and its segment statistics.

        ``garbage_percentage`` is a double, which ``protocol=2`` sends as 8
        raw bytes. A client built with both ``protocol=2`` and
        ``decode_responses=True`` decodes those bytes as text, so the value
        comes back as a ``str`` of 8 characters or raises
        ``UnicodeDecodeError``.
        """
        return self._volume("DESCRIBE", name)

    @overload
    def set_status(self: SyncClient, name: str, status: VolumeStatus) -> bool: ...

    @overload
    def set_status(
        self: AsyncClient, name: str, status: VolumeStatus
    ) -> Awaitable[bool]: ...

    def set_status(self, name: str, status: VolumeStatus) -> Any:
        """``VOLUME.ADMIN SET-STATUS``. Open or close the volume.

        The shard has a status of its own, see
        :meth:`kronotop.Admin.set_shard_status`.
        """
        return self._volume("SET-STATUS", name, status)

    @overload
    def list_segments(self: SyncClient, name: str) -> _SegmentIds: ...

    @overload
    def list_segments(self: AsyncClient, name: str) -> Awaitable[_SegmentIds]: ...

    def list_segments(self, name: str) -> Any:
        """``VOLUME.ADMIN LIST-SEGMENTS``. The segment ids of the volume."""
        return self._volume("LIST-SEGMENTS", name)

    @overload
    def cleanup_orphan_files(self: SyncClient, name: str) -> _Names: ...

    @overload
    def cleanup_orphan_files(self: AsyncClient, name: str) -> Awaitable[_Names]: ...

    def cleanup_orphan_files(self, name: str) -> Any:
        """``VOLUME.ADMIN CLEANUP-ORPHAN-FILES``. Delete the segment files on
        disk that the metadata no longer tracks. Returns the deleted paths."""
        return self._volume("CLEANUP-ORPHAN-FILES", name)

    @overload
    def prune_changelog(self: SyncClient, name: str, retention_hours: int) -> bool: ...

    @overload
    def prune_changelog(
        self: AsyncClient, name: str, retention_hours: int
    ) -> Awaitable[bool]: ...

    def prune_changelog(self, name: str, retention_hours: int) -> Any:
        """``VOLUME.ADMIN PRUNE-CHANGELOG``. Drop the changelog entries older
        than ``retention_hours``. Segment files are not touched."""
        return self._volume("PRUNE-CHANGELOG", name, str(retention_hours))

    @overload
    def start_replication(self: SyncClient, name: str) -> bool: ...

    @overload
    def start_replication(self: AsyncClient, name: str) -> Awaitable[bool]: ...

    def start_replication(self, name: str) -> Any:
        """``VOLUME.ADMIN REPLICATION START``. Resume replication of the
        volume on this member, which must be a standby of the shard."""
        return self._volume("REPLICATION", "START", name)

    @overload
    def stop_replication(self: SyncClient, name: str) -> bool: ...

    @overload
    def stop_replication(self: AsyncClient, name: str) -> Awaitable[bool]: ...

    def stop_replication(self, name: str) -> Any:
        """``VOLUME.ADMIN REPLICATION STOP``. Stop replication of the volume
        on this member. It stays stopped until :meth:`start_replication`."""
        return self._volume("REPLICATION", "STOP", name)

    @overload
    def vacuum_start(self: SyncClient, name: str, garbage_threshold: float) -> bool: ...

    @overload
    def vacuum_start(
        self: AsyncClient, name: str, garbage_threshold: float
    ) -> Awaitable[bool]: ...

    def vacuum_start(self, name: str, garbage_threshold: float) -> Any:
        """``VOLUME.ADMIN VACUUM START``. Start a vacuum run in the background.

        A segment whose garbage percentage is over ``garbage_threshold``
        (between 0 and 100 exclusive) is evacuated and its file deleted. One
        run per volume, and the metadata of a finished run blocks the next one
        until :meth:`vacuum_drop`.
        """
        return self._volume("VACUUM", "START", name, str(garbage_threshold))

    @overload
    def vacuum_stop(self: SyncClient, name: str) -> bool: ...

    @overload
    def vacuum_stop(self: AsyncClient, name: str) -> Awaitable[bool]: ...

    def vacuum_stop(self, name: str) -> Any:
        """``VOLUME.ADMIN VACUUM STOP``. Stop the run. Blocks until every
        worker has finished its batch."""
        return self._volume("VACUUM", "STOP", name)

    @overload
    def vacuum_drop(self: SyncClient, name: str) -> bool: ...

    @overload
    def vacuum_drop(self: AsyncClient, name: str) -> Awaitable[bool]: ...

    def vacuum_drop(self, name: str) -> Any:
        """``VOLUME.ADMIN VACUUM DROP``. Clear the metadata of a finished or
        stopped run."""
        return self._volume("VACUUM", "DROP", name)

    @overload
    def vacuum_status(self: SyncClient, name: str) -> VacuumStatus: ...

    @overload
    def vacuum_status(self: AsyncClient, name: str) -> Awaitable[VacuumStatus]: ...

    def vacuum_status(self, name: str) -> Any:
        """``VOLUME.ADMIN VACUUM STATUS``. The progress of the run.

        An error when no run has left metadata, because none started or
        :meth:`vacuum_drop` ran.
        """
        return self._volume("VACUUM", "STATUS", name, phase="VACUUM STATUS")

    @overload
    def mark_stale_prefixes(
        self: SyncClient, operation: StalePrefixesOperation
    ) -> bool: ...

    @overload
    def mark_stale_prefixes(
        self: AsyncClient, operation: StalePrefixesOperation
    ) -> Awaitable[bool]: ...

    def mark_stale_prefixes(self, operation: StalePrefixesOperation) -> Any:
        """``VOLUME.ADMIN MARK-STALE-PREFIXES``. Control the cluster-wide task
        that clears prefixes whose data is gone.

        ``START`` runs it on this member, ``STOP`` stops it, ``REMOVE`` stops
        it and drops its metadata. The task keeps a progress marker, so a
        restart continues where it stopped.
        """
        return self._volume("MARK-STALE-PREFIXES", operation)

    @overload
    def locate_stale_prefixes_task(self: SyncClient) -> dict[str, Any]: ...

    @overload
    def locate_stale_prefixes_task(
        self: AsyncClient,
    ) -> Awaitable[dict[str, Any]]: ...

    def locate_stale_prefixes_task(self) -> Any:
        """``VOLUME.ADMIN MARK-STALE-PREFIXES LOCATE``. The member that owns
        the task, as ``member_id``, ``process_id``, ``external_advertise`` and
        ``internal_advertise``. The two address lists are empty when that
        member is no longer registered."""
        return self._volume("MARK-STALE-PREFIXES", "LOCATE", phase="LOCATE")

    @overload
    def cursor(self: SyncClient, name: str) -> dict[str, Any]: ...

    @overload
    def cursor(self: AsyncClient, name: str) -> Awaitable[dict[str, Any]]: ...

    def cursor(self, name: str) -> Any:
        """``VOLUME.INSPECT CURSOR``. The write cursor of the active segment, as
        ``active_segment_id``, ``versionstamp``, ``next_position`` and
        ``sequence_number``. An empty volume has ``""`` and ``-1`` for the
        last two."""
        return self.execute_command(
            "VOLUME.INSPECT", "CURSOR", name, inspect_subcommand="CURSOR"
        )

    @overload
    def replication_status(
        self: SyncClient, name: str, standby_id: str
    ) -> dict[str, Any]: ...

    @overload
    def replication_status(
        self: AsyncClient, name: str, standby_id: str
    ) -> Awaitable[dict[str, Any]]: ...

    def replication_status(self, name: str, standby_id: str) -> Any:
        """``VOLUME.INSPECT REPLICATION``. How far ``standby_id`` has
        replicated the volume, as ``stage``, ``status``, ``error_message``, and
        the maps ``cursor``, ``cdc_stage`` and ``segment_replication_stage``.

        ``standby_id`` is a member id or a 4-character prefix. Before the
        first run the strings are empty and the numbers zero.
        """
        return self.execute_command(
            "VOLUME.INSPECT",
            "REPLICATION",
            name,
            standby_id,
            inspect_subcommand="REPLICATION",
        )

    def _stats(self, name: str, *args: Any, phase: str) -> Any:
        """Send one ``VOLUME.STATS`` call. ``phase`` is the callback key."""
        return self.execute_command("VOLUME.STATS", name, *args, stats_subcommand=phase)

    @overload
    def stats(self: SyncClient, name: str) -> dict[str, Any]: ...

    @overload
    def stats(self: AsyncClient, name: str) -> Awaitable[dict[str, Any]]: ...

    def stats(self, name: str) -> Any:
        """``VOLUME.STATS``. An overview of the volume, as ``status`` (a
        :data:`VolumeStatus`), ``segment_count``, ``total_size_bytes``,
        ``used_bytes``, ``free_bytes``, ``garbage_bytes``,
        ``total_cardinality`` (``int``), ``garbage_percentage`` and
        ``fill_ratio`` (``float``). The byte and cardinality fields are sums
        over every segment, and the two doubles have the ``protocol=2`` limit
        of :meth:`describe`.
        """
        return self._stats(name, phase="OVERVIEW")

    @overload
    def stats_opcounters(self: SyncClient, name: str) -> dict[str, int]: ...

    @overload
    def stats_opcounters(self: AsyncClient, name: str) -> Awaitable[dict[str, int]]: ...

    def stats_opcounters(self, name: str) -> Any:
        """``VOLUME.STATS OPCOUNTERS``. The in-memory counters ``appends``,
        ``deletes``, ``updates``, ``gets``, ``bytes_appended``, ``bytes_read``
        and ``segments_created``. They count from the moment the volume
        opened, and a restart or :meth:`stats_reset` sets them to zero.
        """
        return self._stats(name, "OPCOUNTERS", phase="OPCOUNTERS")

    @overload
    def stats_segments(self: SyncClient, name: str) -> _SegmentStats: ...

    @overload
    def stats_segments(self: AsyncClient, name: str) -> Awaitable[_SegmentStats]: ...

    def stats_segments(self, name: str) -> Any:
        """``VOLUME.STATS SEGMENTS``. One map per segment, with ``segment_id``,
        ``size_bytes``, ``used_bytes``, ``free_bytes``, ``garbage_bytes``,
        ``cardinality`` (``int``) and ``garbage_percentage`` (``float``, see
        :meth:`stats`). Empty when the volume has no segment.
        """
        return self._stats(name, "SEGMENTS", phase="SEGMENTS")

    @overload
    def stats_replication(
        self: SyncClient, name: str, standby_id: str
    ) -> dict[str, Any]: ...

    @overload
    def stats_replication(
        self: AsyncClient, name: str, standby_id: str
    ) -> Awaitable[dict[str, Any]]: ...

    def stats_replication(self, name: str, standby_id: str) -> Any:
        """``VOLUME.STATS REPLICATION``. A part of :meth:`replication_status`
        as one flat map, with ``stage``, ``status``, ``error_message``,
        ``cursor_segment_id``, ``cursor_position``, ``cdc_sequence_number``
        and ``cdc_position``. ``segment_replication_stage`` is left out.
        ``standby_id`` is a member id or a 4-character prefix.
        """
        return self._stats(name, "REPLICATION", standby_id, phase="REPLICATION")

    @overload
    def stats_reset(self: SyncClient, name: str) -> bool: ...

    @overload
    def stats_reset(self: AsyncClient, name: str) -> Awaitable[bool]: ...

    def stats_reset(self, name: str) -> Any:
        """``VOLUME.STATS RESET``. Set the counters of
        :meth:`stats_opcounters` to zero. The segment statistics come from
        FoundationDB and do not change."""
        return self._stats(name, "RESET", phase="RESET")

    @overload
    def segment_range(
        self: SyncClient, name: str, segment_id: int, *ranges: tuple[int, int]
    ) -> _Chunks: ...

    @overload
    def segment_range(
        self: AsyncClient, name: str, segment_id: int, *ranges: tuple[int, int]
    ) -> Awaitable[_Chunks]: ...

    def segment_range(
        self, name: str, segment_id: int, *ranges: tuple[int, int]
    ) -> Any:
        """``SEGMENT.RANGE``. Raw bytes of the segment, one ``bytes`` per
        ``(position, length)`` pair, in the order asked.

        At least one pair. A pair past the end of the segment is refused with
        the ``OUTOFBOUND`` prefix. A client built with ``decode_responses=True``
        decodes the raw reply as text, so a chunk that is valid UTF-8 comes
        back as ``str`` and any other raises ``UnicodeDecodeError``.
        """
        args = [str(n) for pair in ranges for n in pair]
        return self.execute_command("SEGMENT.RANGE", name, str(segment_id), *args)

    @overload
    def segment_tail_pointer(
        self: SyncClient, name: str, segment_id: int
    ) -> TailPointer: ...

    @overload
    def segment_tail_pointer(
        self: AsyncClient, name: str, segment_id: int
    ) -> Awaitable[TailPointer]: ...

    def segment_tail_pointer(self, name: str, segment_id: int) -> Any:
        """``SEGMENT.TAILPOINTER``. ``next_position``, the first byte no
        write has reached, and ``sequence_number`` of the last write.

        ``next_position`` is ``0`` for an empty segment. ``sequence_number``
        is ``-1`` unless the segment is the active one, has data, and its last
        changelog entry is not pruned.
        """
        return self.execute_command("SEGMENT.TAILPOINTER", name, str(segment_id))

    @overload
    def changelog_range(
        self: SyncClient,
        name: str,
        start: int,
        end: int,
        *,
        kind: ChangeLogKind | None = None,
        limit: int | None = None,
        reverse: bool = False,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
    ) -> _Entries: ...

    @overload
    def changelog_range(
        self: AsyncClient,
        name: str,
        start: int,
        end: int,
        *,
        kind: ChangeLogKind | None = None,
        limit: int | None = None,
        reverse: bool = False,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
    ) -> Awaitable[_Entries]: ...

    def changelog_range(
        self,
        name: str,
        start: int,
        end: int,
        *,
        kind: ChangeLogKind | None = None,
        limit: int | None = None,
        reverse: bool = False,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
    ) -> Any:
        """``CHANGELOG.RANGE``. The changelog entries whose sequence number
        is between ``start`` and ``end``, both inclusive by default.

        ``kind`` keeps one parent operation kind, ``None`` keeps both. Without
        ``limit``, or with ``limit`` at ``0`` or below, the server returns at
        most 1000 entries. ``reverse`` walks from ``end`` down. An entry
        carries ``before``, ``after`` or both, see :data:`ChangeLogEntry`.
        """
        args: list[EncodableT] = [
            name,
            kind or "*",
            f"{'[' if start_inclusive else '('}{start}",
            f"{end}{']' if end_inclusive else ')'}",
        ]
        if limit is not None:
            args += ["LIMIT", str(limit)]
        if reverse:
            args.append("REVERSE")
        return self.execute_command("CHANGELOG.RANGE", *args)

    @overload
    def changelog_watch(self: SyncClient, name: str, sequence_number: int) -> int: ...

    @overload
    def changelog_watch(
        self: AsyncClient, name: str, sequence_number: int
    ) -> Awaitable[int]: ...

    def changelog_watch(self, name: str, sequence_number: int) -> Any:
        """``CHANGELOG.WATCH``. Block until the changelog's safe watermark
        is past ``sequence_number``, and return the watermark.

        A ``sequence_number`` past the volume's latest one is an error. The
        call holds the connection while it waits, so a ``socket_timeout`` on
        the client cuts it with ``TimeoutError``.
        """
        return self.execute_command("CHANGELOG.WATCH", name, str(sequence_number))


class Volume(CommandGroup, VolumeCommands):
    """``admin.volume`` on the synchronous admin client."""

    _is_async_client: Literal[False] = False


class AsyncVolume(CommandGroup, VolumeCommands):
    """``admin.volume`` on the asyncio admin client."""

    _is_async_client: Literal[True] = True
