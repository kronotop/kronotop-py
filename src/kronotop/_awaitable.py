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

"""What an asyncio call returns when it has to do I/O before the object."""

from __future__ import annotations

from collections.abc import Coroutine, Generator
from types import TracebackType
from typing import Any, Generic, Protocol, TypeVar

__all__ = ["AwaitableContext"]


class _Closer(Protocol):
    """What the wrapper needs from the object: the exit half of ``async with``."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


EnteredT = TypeVar("EnteredT", bound=_Closer)


class AwaitableContext(Generic[EnteredT]):
    """Wraps a coroutine so it can be awaited or entered with ``async with``.

    ``session.bucket.query`` sends a command before it can build its cursor.
    Both forms read the same as on the sync client:

    ```python
    async with session.bucket.query("users", {}) as cursor: ...
    cursor = await session.bucket.query("users", {})
    ```

    A wrapper is used once. A second use raises ``RuntimeError`` because the
    coroutine is spent, so call the method again. Exit goes through the
    object's own ``__aexit__``.
    """

    def __init__(self, coro: Coroutine[Any, Any, EnteredT]) -> None:
        self._coro = coro
        self._entered: EnteredT | None = None

    def __await__(self) -> Generator[Any, None, EnteredT]:
        return self._coro.__await__()

    async def __aenter__(self) -> EnteredT:
        entered = await self._coro
        self._entered = entered
        return entered

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # __aenter__ ran, or the interpreter would not be here.
        assert self._entered is not None
        await self._entered.__aexit__(exc_type, exc, tb)
