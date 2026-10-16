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

"""Python client for Kronotop.

The asyncio client lives in a separate namespace and is imported explicitly::

    from kronotop import Kronotop
    from kronotop.asyncio import Kronotop as AsyncKronotop
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from kronotop.admin import DEFAULT_ADMIN_PORT, Admin
from kronotop.client import DEFAULT_PORT, Kronotop, from_url
from kronotop.commands.bucket import (
    BucketLocation,
    Collation,
    CompoundIndex,
    CompoundIndexField,
    Cursor,
    ObjectIdBatch,
    QueryBatch,
    ShardLocation,
    SingleFieldIndex,
    VectorIndex,
    VectorSearchResult,
)
from kronotop.config import KronotopConfig
from kronotop.exceptions import (
    BarrierNotSatisfiedError,
    BucketAlreadyExistsError,
    BucketBeingRemovedError,
    ConfigurationError,
    DuplicateKeyError,
    IndexTypeMismatchError,
    KronotopCommandError,
    KronotopConnectionError,
    KronotopError,
    NamespaceAlreadyExistsError,
    NamespaceBeingRemovedError,
    NoCommitVersionError,
    NoCommonMemberError,
    NoSuchBucketError,
    NoSuchIndexError,
    NoSuchNamespaceError,
    NotCommittedError,
    RejectError,
    TransactionCancelledError,
    TransactionError,
    TransactionTooOldError,
    VectorIndexNotReadyError,
)
from kronotop.session import Session, Transaction

try:
    __version__ = version("kronotop-py")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "DEFAULT_ADMIN_PORT",
    "DEFAULT_PORT",
    "Admin",
    "BarrierNotSatisfiedError",
    "BucketAlreadyExistsError",
    "BucketBeingRemovedError",
    "BucketLocation",
    "Collation",
    "CompoundIndex",
    "CompoundIndexField",
    "ConfigurationError",
    "Cursor",
    "DuplicateKeyError",
    "IndexTypeMismatchError",
    "Kronotop",
    "KronotopCommandError",
    "KronotopConfig",
    "KronotopConnectionError",
    "KronotopError",
    "NamespaceAlreadyExistsError",
    "NamespaceBeingRemovedError",
    "NoCommitVersionError",
    "NoCommonMemberError",
    "NoSuchBucketError",
    "NoSuchIndexError",
    "NoSuchNamespaceError",
    "NotCommittedError",
    "ObjectIdBatch",
    "QueryBatch",
    "RejectError",
    "Session",
    "ShardLocation",
    "SingleFieldIndex",
    "Transaction",
    "TransactionCancelledError",
    "TransactionError",
    "TransactionTooOldError",
    "VectorIndex",
    "VectorIndexNotReadyError",
    "VectorSearchResult",
    "__version__",
    "from_url",
]
