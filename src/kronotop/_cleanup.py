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

"""Guard for a cleanup that runs while an error unwinds a block.

A failed ``with`` block still rolls back its transaction or closes its cursor
over the same socket, and that socket may be what failed. A cleanup that raised
would hide the error that caused it.

Own module because ``commands/bucket.py`` needs it and cannot import
``_session_base`` (circular through ``kronotop.commands``).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

logger = logging.getLogger("kronotop")


@contextlib.contextmanager
def ignore_cleanup_error(what: str, target: object) -> Iterator[None]:
    """Log and swallow what the cleanup raises. Only for a block that already failed.

    A plain ``contextmanager``, so the body can be ``await cleanup()`` too. It
    catches ``Exception`` and lets a cancellation propagate.
    """
    try:
        yield
    except Exception as exc:
        logger.warning("error ignored in %s of %r: %s", what, target, exc)
