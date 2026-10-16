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

"""Asyncio interface to Kronotop, in its own namespace::

    from kronotop.asyncio import Kronotop
"""

from __future__ import annotations

from kronotop._awaitable import AwaitableContext
from kronotop.asyncio.admin import Admin
from kronotop.asyncio.client import Kronotop, from_url
from kronotop.asyncio.session import Session, Transaction
from kronotop.commands.bucket import AsyncCursor

__all__ = [
    "Admin",
    "AsyncCursor",
    "AwaitableContext",
    "Kronotop",
    "Session",
    "Transaction",
    "from_url",
]
