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

"""Open sessions, kept between calls.

mem0 calls the store from several threads, one command per call. Opening a
session costs ``SESSION.ATTRIBUTE LIST`` and closing one costs
``SESSION.CLOSE``. A session from here is open already and goes back open, so
a call costs one command.

A session that goes back must carry no state. The caller names the namespace
on each command and closes every cursor before the block ends.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator

from kronotop import KronotopConnectionError, RejectError, Session

__all__ = ["SessionPool"]


class SessionPool:
    """Sessions that ``open_session`` opened, handed out one caller at a time.

    The pool has no upper bound. It holds as many sessions as the highest
    number of callers at one moment.
    """

    def __init__(self, open_session: Callable[[], Session]) -> None:
        self._open_session = open_session
        #: Guards the two attributes below.
        self._lock = threading.Lock()
        self._idle: list[Session] = []
        self._closed = False

    def _take(self) -> Session:
        with self._lock:
            if self._closed:
                raise KronotopConnectionError("session pool is closed")
            if self._idle:
                # The last one in is most likely still connected.
                return self._idle.pop()
        # Outside the lock, because this sends commands.
        return self._open_session()

    def _keep(self, session: Session) -> bool:
        """Put ``session`` back. ``False`` when it cannot be used again."""
        if session.broken or session.in_transaction:
            return False
        with self._lock:
            if self._closed:
                return False
            self._idle.append(session)
            return True

    @contextlib.contextmanager
    def borrow(self) -> Iterator[Session]:
        """Lend one session for the length of the block.

        The session goes back open. It is closed instead when it lost its
        connection, when a transaction is still open on it, when the pool was
        closed in the meantime, or when the block raised
        :class:`~kronotop.RejectError`.
        """
        session = self._take()
        rejected = False
        try:
            yield session
        except RejectError:
            # The shard moved, so this session is on the wrong member. The
            # next one is opened on the member that holds it now.
            rejected = True
            raise
        finally:
            if rejected or not self._keep(session):
                session.close()

    def close(self) -> None:
        """Close every idle session. A session that is out closes on return.

        Calling this twice is safe.
        """
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for session in idle:
            session.close()
