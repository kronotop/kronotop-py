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

"""Kronotop command mixins.

Each family lives in a sibling module, aggregated in
:mod:`kronotop.commands.core`. A mixin only needs ``self.execute_command`` and
never touches the pool.

:class:`KronotopSessionCommands` is what a session exposes as plain methods. A
large family is a :class:`CommandGroup` on a session attribute.
"""

from __future__ import annotations

from kronotop.commands.base import (
    AsyncClient,
    CommandGroup,
    CommandsBase,
    SyncClient,
)
from kronotop.commands.bucket import (
    AsyncBucket,
    AsyncCursor,
    Bucket,
    BucketCommands,
    BucketLocation,
    Collation,
    Cursor,
    Operation,
    QueryBatch,
    ShardLocation,
    SortDirection,
    SortSpec,
)
from kronotop.commands.command import (
    AsyncCommand,
    Command,
    CommandCommands,
    CommandDoc,
    CommandInfo,
)
from kronotop.commands.connection import ConnectionCommands
from kronotop.commands.core import (
    AsyncKronotopSessionCommands,
    KronotopSessionCommands,
)
from kronotop.commands.namespace import (
    AsyncNamespace,
    Namespace,
    NamespaceCommands,
)
from kronotop.commands.server import ServerCommands
from kronotop.commands.session import (
    InputType,
    ObjectIdFormat,
    ReplyType,
    SessionAdminCommands,
    SessionAttributeName,
)
from kronotop.commands.transactions import Returning, TickMode, TransactionCommands
from kronotop.commands.zmap import AsyncZMap, ZMap, ZMapCommands

__all__ = [
    "AsyncBucket",
    "AsyncClient",
    "AsyncCommand",
    "AsyncCursor",
    "AsyncKronotopSessionCommands",
    "AsyncNamespace",
    "AsyncZMap",
    "Bucket",
    "BucketCommands",
    "BucketLocation",
    "Collation",
    "Command",
    "CommandCommands",
    "CommandDoc",
    "CommandGroup",
    "CommandInfo",
    "CommandsBase",
    "ConnectionCommands",
    "Cursor",
    "InputType",
    "KronotopSessionCommands",
    "Namespace",
    "NamespaceCommands",
    "ObjectIdFormat",
    "Operation",
    "QueryBatch",
    "ReplyType",
    "Returning",
    "ServerCommands",
    "SessionAdminCommands",
    "SessionAttributeName",
    "ShardLocation",
    "SortDirection",
    "SortSpec",
    "SyncClient",
    "TickMode",
    "TransactionCommands",
    "ZMap",
    "ZMapCommands",
]
