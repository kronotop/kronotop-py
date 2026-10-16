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

"""Which member a session opens on, given the buckets it will touch.

``session_for`` reads where each bucket lives and opens the session on a
member that holds them all. The steps that send no command live here, shared
by both clients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from kronotop.exceptions import ConfigurationError, NoCommonMemberError

if TYPE_CHECKING:
    from kronotop.commands.bucket import BucketLocation

__all__ = ["Found", "Placement", "check_targets", "common_members", "primaries"]

#: ``(namespace, name, members)`` per target, in the order the caller named them.
Found = list[tuple[str, str, set[str]]]


class Placement:
    """What ``LOCATE`` answered, kept so the next ``session_for`` can skip it.

    Maps ``(namespace, name)`` to the primaries. A session opened from an entry
    forgets it on ``RejectError``, because that is how a moved bucket shows up.

    No lock. Each method is one dict operation, atomic in CPython, and a stale
    read costs one more ``LOCATE``.
    """

    def __init__(self) -> None:
        self._members: dict[tuple[str, str], set[str]] = {}

    def get(self, namespace: str, name: str) -> set[str] | None:
        return self._members.get((namespace, name))

    def keep(self, namespace: str, name: str, members: set[str]) -> None:
        self._members[(namespace, name)] = members

    def forget(self, buckets: Mapping[str, Sequence[str]]) -> None:
        """Drop every bucket in ``buckets``. One that is not kept is skipped."""
        for namespace, names in buckets.items():
            for name in names:
                self._members.pop((namespace, name), None)


def check_targets(buckets: Mapping[str, Sequence[str]]) -> None:
    """Reject a ``buckets`` argument with nothing to look up.

    Runs before any command, so a mistake costs no round trip. A bare name
    where a list belongs is caught too, because iterating ``"users"`` yields
    letters and ``b"users"`` yields ints.
    """
    if not buckets:
        raise ConfigurationError("buckets names no bucket")
    for namespace, names in buckets.items():
        if isinstance(names, str | bytes):
            raise ConfigurationError(
                f"namespace {namespace!r} needs a list of bucket names, not {names!r}"
            )
        if not names:
            raise ConfigurationError(f"namespace {namespace!r} names no bucket")


def primaries(location: BucketLocation) -> set[str]:
    """The members that hold one bucket, as ``host:port``.

    A bucket command runs on the primary of one of the bucket's shards. A
    standby holds the data too, but rejects the command.

    A shard names its primary by id, so the address comes from the member
    table. The first advertised address is the preferred one.
    """
    return {location.members[shard.primary][0] for shard in location.shards}


def common_members(found: Found) -> set[str]:
    """The members that hold every target.

    Raises ``NoCommonMemberError`` when there is none, because the targets then
    cannot share a session.
    """
    shared = set(found[0][2])
    for _, _, members in found[1:]:
        shared &= members
    if not shared:
        raise NoCommonMemberError(
            "no member holds all of them: "
            + ", ".join(
                f"{name} in {namespace} on "
                + (", ".join(sorted(members)) if members else "no member")
                for namespace, name, members in found
            )
        )
    return shared
