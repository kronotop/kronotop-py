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

"""Synchronous cluster administration client."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Literal

import redis
from redis.exceptions import ResponseError

from kronotop._callbacks import register_callbacks
from kronotop._options import (
    DEFAULT_ADMIN_PORT,
    apply_defaults,
    check_kwargs,
    parse_host,
)
from kronotop.commands.admin import AdminCommands
from kronotop.commands.volume import Volume
from kronotop.exceptions import wrap_response_error

__all__ = ["DEFAULT_ADMIN_PORT", "Admin"]


# Wraps ``redis.Redis`` instead of subclassing it, same reason as ``Kronotop``:
# a subclass would carry the whole Redis command set.
class Admin(AdminCommands):
    """Client for the ``KR.ADMIN`` and ``TASK.ADMIN`` commands of one member.
    The volume and replication commands are on ``volume``.

    The server answers them on its internal listener, port 3320, so this is a
    separate client with no session and no transaction. It names one member:
    ``describe_member``, ``list_silent_members``, ``list_tasks`` and
    ``volume.list`` give that member's own view.

    ``host`` is ``host`` or ``host:port``, default port ``3320``. The other
    keyword arguments are the ones :class:`kronotop.Kronotop` takes.

    The internal listener has ``HELLO`` but no ``AUTH``. With
    ``auth.requirepass`` on the server, pass ``protocol=3``, because
    ``protocol=2`` sends the password as ``AUTH``.

    ```python
    with Admin("10.0.0.1") as admin:
        admin.describe_cluster()
    ```
    """

    _is_async_client: Literal[False] = False

    def __init__(self, host: str = f"localhost:{DEFAULT_ADMIN_PORT}", **kwargs: Any):
        check_kwargs(kwargs, "host")
        _, hostname, port = parse_host(host, "host", DEFAULT_ADMIN_PORT)
        apply_defaults(kwargs)
        self._redis = redis.Redis(host=hostname, port=port, **kwargs)
        register_callbacks(self._redis)
        #: Volume commands, e.g. ``admin.volume.describe("bucket-shard-0")``.
        self.volume = Volume(self)

    def execute_command(self, *args: Any, **options: Any) -> Any:
        """Send a raw command. Every method funnels through here."""
        try:
            return self._redis.execute_command(*args, **options)  # type: ignore[no-untyped-call]
        except ResponseError as exc:
            raise wrap_response_error(exc) from None

    def close(self) -> None:
        self._redis.close()

    def __enter__(self) -> Admin:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
