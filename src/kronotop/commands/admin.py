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

"""``KR.ADMIN`` commands, the surface of :class:`kronotop.Admin`.

The server answers them on the internal listener only, port 3320, so they are
not on the session. Every subcommand is stateless, and the connection goes
back to the pool after each one.

A member id is 40 characters of hex. Where the server takes one, a 4-character
prefix works too. Every name, id and address comes back as ``str``, on either
protocol and whatever ``decode_responses`` says.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Literal, overload

from kronotop.commands.base import AsyncClient, CommandsBase, SyncClient

__all__ = [
    "AdminCommands",
    "MemberInfo",
    "MemberStatus",
    "ShardId",
    "ShardInfo",
    "ShardKind",
    "ShardStatus",
    "TaskInfo",
]

#: One shard: ``primary`` (a member id, or ``""`` when unassigned),
#: ``standbys`` (member ids), ``status`` (a :data:`ShardStatus`) and
#: ``linked_volumes`` (names such as ``bucket-shard-0``).
ShardInfo = dict[str, Any]

#: One member: ``status`` (a :data:`MemberStatus`), ``process_id``,
#: ``external_advertise`` and ``internal_advertise`` (``host:port`` lists,
#: preferred entry first) and ``latest_heartbeat`` (an ``int``, ``0`` before
#: the first heartbeat). ``describe_member`` adds ``member_id``.
MemberInfo = dict[str, Any]

#: One background task: ``running`` and ``finished`` (``bool``),
#: ``started_at`` and ``last_run`` (``int``, UNIX seconds).
TaskInfo = dict[str, Any]

#: ``STASH`` is experimental and off by default. The server matches the value
#: case insensitively.
ShardKind = Literal["BUCKET", "STASH"]

#: A zero-based shard index, or ``*`` for every shard of the kind in one
#: transaction, so one shard that fails a check rolls the whole call back.
ShardId = int | Literal["*"]

ShardStatus = Literal["READWRITE", "READONLY", "INOPERABLE"]

MemberStatus = Literal["RUNNING", "UNAVAILABLE", "STOPPED", "UNKNOWN"]


class AdminCommands(CommandsBase):
    """``KR.ADMIN``, ``TASK.ADMIN`` and ``PING``.

    Every error here is a plain ``ERR``, so it arrives as a
    ``redis.ResponseError``. The literal types are hints. The server validates
    the value and matches it case insensitively.
    """

    def _admin(self, subcommand: str, *args: Any, phase: str | None = None) -> Any:
        """Send one subcommand. ``phase`` overrides the callback key."""
        return self.execute_command(
            "KR.ADMIN", subcommand, *args, admin_subcommand=phase or subcommand
        )

    @overload
    def ping(self: SyncClient) -> bool: ...

    @overload
    def ping(self: AsyncClient) -> Awaitable[bool]: ...

    def ping(self) -> Any:
        """``PING``. Whether the member answers on its admin port."""
        return self.execute_command("PING")

    @overload
    def initialize_cluster(self: SyncClient) -> bool: ...

    @overload
    def initialize_cluster(self: AsyncClient) -> Awaitable[bool]: ...

    def initialize_cluster(self) -> Any:
        """``KR.ADMIN INITIALIZE-CLUSTER``. Create the cluster metadata.

        Runs once per cluster, and is the only subcommand that works before
        that. Every shard starts ``INOPERABLE`` with no route, so assign a
        primary with :meth:`set_primary` and open the shards with
        :meth:`set_shard_status`.
        """
        return self._admin("INITIALIZE-CLUSTER")

    @overload
    def describe_cluster(self: SyncClient) -> dict[str, Any]: ...

    @overload
    def describe_cluster(self: AsyncClient) -> Awaitable[dict[str, Any]]: ...

    def describe_cluster(self) -> Any:
        """``KR.ADMIN DESCRIBE-CLUSTER``. The whole topology.

        ``metadata_version`` and ``cluster_name`` are ``str``. ``bucket`` maps
        each shard id, an ``int``, to a :data:`ShardInfo`. ``stash`` is there
        too when the stash subsystem is enabled.
        """
        return self._admin("DESCRIBE-CLUSTER")

    @overload
    def describe_shard(
        self: SyncClient, kind: ShardKind, shard_id: int
    ) -> ShardInfo: ...

    @overload
    def describe_shard(
        self: AsyncClient, kind: ShardKind, shard_id: int
    ) -> Awaitable[ShardInfo]: ...

    def describe_shard(self, kind: ShardKind, shard_id: int) -> Any:
        """``KR.ADMIN DESCRIBE-SHARD``. One shard, as :meth:`describe_cluster`
        lists it."""
        return self._admin("DESCRIBE-SHARD", kind, shard_id)

    @overload
    def describe_member(self: SyncClient) -> MemberInfo: ...

    @overload
    def describe_member(self: AsyncClient) -> Awaitable[MemberInfo]: ...

    def describe_member(self) -> Any:
        """``KR.ADMIN DESCRIBE-MEMBER``. The member this client talks to.

        The one reply that carries ``member_id``, because the caller may not
        know it yet.
        """
        return self._admin("DESCRIBE-MEMBER")

    @overload
    def find_member(self: SyncClient, member_id: str) -> MemberInfo: ...

    @overload
    def find_member(self: AsyncClient, member_id: str) -> Awaitable[MemberInfo]: ...

    def find_member(self, member_id: str) -> Any:
        """``KR.ADMIN FIND-MEMBER``. One member, by id or 4-character prefix.

        A prefix that matches no member, or more than one, is an error.
        """
        return self._admin("FIND-MEMBER", member_id)

    @overload
    def list_members(self: SyncClient) -> dict[str, MemberInfo]: ...

    @overload
    def list_members(self: AsyncClient) -> Awaitable[dict[str, MemberInfo]]: ...

    def list_members(self) -> Any:
        """``KR.ADMIN LIST-MEMBERS``. Every registered member, keyed by id.

        Members of every status, ordered by ``process_id``.
        """
        return self._admin("LIST-MEMBERS")

    @overload
    def list_silent_members(self: SyncClient) -> list[str]: ...

    @overload
    def list_silent_members(self: AsyncClient) -> Awaitable[list[str]]: ...

    def list_silent_members(self) -> Any:
        """``KR.ADMIN LIST-SILENT-MEMBERS``. Ids of the members whose heartbeat stopped.

        This is the view of the member asked, so another member can report
        another list. The member asked is never in it, and a member whose
        heartbeat resumes drops off on its own.
        """
        return self._admin("LIST-SILENT-MEMBERS")

    @overload
    def set_member_status(
        self: SyncClient, member_id: str, status: MemberStatus
    ) -> bool: ...

    @overload
    def set_member_status(
        self: AsyncClient, member_id: str, status: MemberStatus
    ) -> Awaitable[bool]: ...

    def set_member_status(self, member_id: str, status: MemberStatus) -> Any:
        """``KR.ADMIN SET-MEMBER-STATUS``. Override a member's status.

        Set ``STOPPED`` before :meth:`remove_member`, because a ``RUNNING``
        member cannot be removed.
        """
        return self._admin("SET-MEMBER-STATUS", member_id, status)

    @overload
    def set_shard_status(
        self: SyncClient, kind: ShardKind, shard_id: ShardId, status: ShardStatus
    ) -> bool: ...

    @overload
    def set_shard_status(
        self: AsyncClient, kind: ShardKind, shard_id: ShardId, status: ShardStatus
    ) -> Awaitable[bool]: ...

    def set_shard_status(
        self, kind: ShardKind, shard_id: ShardId, status: ShardStatus
    ) -> Any:
        """``KR.ADMIN SET-SHARD-STATUS``. Open or close a shard.

        ``READWRITE`` takes reads and writes, ``READONLY`` refuses writes,
        ``INOPERABLE`` refuses everything. This is access control only, the
        routes and the volume are left alone.
        """
        return self._admin("SET-SHARD-STATUS", kind, str(shard_id), status)

    @overload
    def set_primary(
        self: SyncClient, kind: ShardKind, shard_id: ShardId, member_id: str
    ) -> bool: ...

    @overload
    def set_primary(
        self: AsyncClient, kind: ShardKind, shard_id: ShardId, member_id: str
    ) -> Awaitable[bool]: ...

    def set_primary(self, kind: ShardKind, shard_id: ShardId, member_id: str) -> Any:
        """``KR.ADMIN ROUTE SET PRIMARY``. Make ``member_id`` own the shard.

        A shard with no primary takes the member as it is. Replacing a primary
        needs the shard not ``READWRITE``, the volume ``READONLY``, and the
        new member a standby that has caught up. The promoted standby leaves
        the standby set. The server refuses ``UNSET PRIMARY``.
        """
        return self._admin("ROUTE", "SET", "PRIMARY", kind, str(shard_id), member_id)

    @overload
    def set_standby(
        self: SyncClient, kind: ShardKind, shard_id: ShardId, member_id: str
    ) -> bool: ...

    @overload
    def set_standby(
        self: AsyncClient, kind: ShardKind, shard_id: ShardId, member_id: str
    ) -> Awaitable[bool]: ...

    def set_standby(self, kind: ShardKind, shard_id: ShardId, member_id: str) -> Any:
        """``KR.ADMIN ROUTE SET STANDBY``. Add ``member_id`` to the shard's standbys.

        The shard needs a primary first. The member must not be the primary or
        a standby already.
        """
        return self._admin("ROUTE", "SET", "STANDBY", kind, str(shard_id), member_id)

    @overload
    def unset_standby(
        self: SyncClient, kind: ShardKind, shard_id: ShardId, member_id: str
    ) -> bool: ...

    @overload
    def unset_standby(
        self: AsyncClient, kind: ShardKind, shard_id: ShardId, member_id: str
    ) -> Awaitable[bool]: ...

    def unset_standby(self, kind: ShardKind, shard_id: ShardId, member_id: str) -> Any:
        """``KR.ADMIN ROUTE UNSET STANDBY``. Drop ``member_id`` from the standbys."""
        return self._admin("ROUTE", "UNSET", "STANDBY", kind, str(shard_id), member_id)

    @overload
    def remove_member(self: SyncClient, member_id: str) -> bool: ...

    @overload
    def remove_member(self: AsyncClient, member_id: str) -> Awaitable[bool]: ...

    def remove_member(self, member_id: str) -> Any:
        """``KR.ADMIN REMOVE-MEMBER``. Drop a member from the cluster.

        A ``RUNNING`` member is refused, so set it ``STOPPED`` with
        :meth:`set_member_status` first.
        """
        return self._admin("REMOVE-MEMBER", member_id)

    @overload
    def drop_cluster(self: SyncClient, name: str) -> str: ...

    @overload
    def drop_cluster(self: SyncClient, name: str, token: str) -> bool: ...

    @overload
    def drop_cluster(self: AsyncClient, name: str) -> Awaitable[str]: ...

    @overload
    def drop_cluster(self: AsyncClient, name: str, token: str) -> Awaitable[bool]: ...

    def drop_cluster(self, name: str, token: str | None = None) -> Any:
        """``KR.ADMIN DROP-CLUSTER``. Delete the cluster metadata, in two calls.

        Without ``token``, checks that ``name`` is the cluster this member runs
        and returns a token that lives 60 seconds. Asking again while it is
        pending returns the same token. With ``token``, removes the whole
        cluster directory from FoundationDB. The token is used up by that
        call, valid or not.

        Only the metadata goes: the volume files on disk stay, and the
        members that ran the cluster are left inconsistent, so stop them.
        """
        if token is None:
            return self._admin("DROP-CLUSTER", name, phase="DROP-CLUSTER-TOKEN")
        return self._admin("DROP-CLUSTER", name, token)

    @overload
    def list_tasks(self: SyncClient) -> dict[str, TaskInfo]: ...

    @overload
    def list_tasks(self: AsyncClient) -> Awaitable[dict[str, TaskInfo]]: ...

    def list_tasks(self) -> Any:
        """``TASK.ADMIN LIST``. This member's background tasks, keyed by name.

        ``last_run`` is ``0`` before the first run.
        """
        return self.execute_command("TASK.ADMIN", "LIST")
