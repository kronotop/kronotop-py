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

"""Command names that open or close a transaction.

Kept out of :mod:`kronotop.session`, so the async session imports no private
name from the sync module.
"""

from __future__ import annotations

#: Tracked on the client side, so ``transaction()`` rejects nesting without a
#: round trip.
OPENS_TRANSACTION = frozenset({"BEGIN"})
CLOSES_TRANSACTION = frozenset({"COMMIT", "ROLLBACK", "SESSION.CLOSE"})

#: These end the transaction even on an error reply, because the server cleans
#: up in a ``finally`` block. An error from ``SESSION.CLOSE`` means the command
#: never ran, so it is not here.
CLOSES_TRANSACTION_ON_ERROR = frozenset({"COMMIT", "ROLLBACK"})
