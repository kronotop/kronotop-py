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

"""The aggregate classes both sessions are built from.

Every command is on the session. The client owns the pool and hands a connection
back after each command, so it only opens sessions. ``client.execute_command``
sends a raw command without one.

Each command family lives in a sibling module and imports from ``base.py``,
never from here.
"""

from __future__ import annotations

from kronotop.commands.connection import ConnectionCommands
from kronotop.commands.server import ServerCommands
from kronotop.commands.session import SessionAdminCommands
from kronotop.commands.transactions import TransactionCommands

__all__ = [
    "AsyncKronotopSessionCommands",
    "KronotopSessionCommands",
]


class KronotopSessionCommands(
    TransactionCommands, SessionAdminCommands, ConnectionCommands, ServerCommands
):
    """The commands the sync session exposes as plain methods.

    Transaction, session attribute, connection and server commands. A large
    family hangs off its own attribute instead, see
    :class:`kronotop.commands.base.CommandGroup`.
    """


class AsyncKronotopSessionCommands(KronotopSessionCommands):
    """Every command the asyncio session exposes.

    Empty, because the overloads in the families already give the async
    session its return types. Override here only when the behaviour differs.
    """
