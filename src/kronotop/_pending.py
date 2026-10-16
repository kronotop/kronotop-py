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

"""Connections a session finalizer could not give back to the pool.

``Session.__del__`` cannot call ``ConnectionPool.release``. The async one is a
coroutine. The sync one takes the non-reentrant lock of ``queue.LifoQueue``,
and a finalizer runs on a GC pass, so it can start on any thread inside that
lock.

So the finalizer appends the connection here, with no lock. Normal code drains
the queue and calls the real ``release``, which is also where
``BlockingConnectionPool`` wakes a waiting caller.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    import redis
    import redis.asyncio

logger = logging.getLogger("kronotop")

#: One queue per pool, because two clients can share a pool. Weak keys, so a
#: dead pool takes its queue with it. The lock guards the lookup only.
_queues: WeakKeyDictionary[Any, deque[Any]] = WeakKeyDictionary()
_queues_lock = threading.Lock()


def queue_for(pool: Any) -> deque[Any]:
    """Return the pending queue of ``pool``, building it on first use.

    Normal code only. A finalizer appends to a queue it already holds.
    """
    with _queues_lock:
        queue = _queues.get(pool)
        if queue is None:
            queue = deque()
            _queues[pool] = queue
        return queue


def _next(queue: deque[Any]) -> Any:
    try:
        return queue.popleft()
    except IndexError:
        # Another thread drained the same pool first.
        return None


def release_pending(pool: redis.ConnectionPool, queue: deque[Any]) -> None:
    """Give every connection left by a finalizer back to ``pool``."""
    while queue:
        conn = _next(queue)
        if conn is None:
            return
        try:
            pool.release(conn)
        except Exception:
            # The pool may have been reset since the session died. One lost
            # connection must not break the command that came here to clean up.
            logger.warning("could not release %r back to the pool", conn, exc_info=True)


async def arelease_pending(
    pool: redis.asyncio.ConnectionPool, queue: deque[Any]
) -> None:
    """Async twin of :func:`release_pending`."""
    while queue:
        conn = _next(queue)
        if conn is None:
            return
        try:
            await pool.release(conn)
        except Exception:
            # See :func:`release_pending`.
            logger.warning("could not release %r back to the pool", conn, exc_info=True)
