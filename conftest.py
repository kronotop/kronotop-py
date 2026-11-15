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

"""Options that every test tree in the repository reads.

pytest loads this file whatever path it is given, so ``tests`` and
``packages/*/tests`` see the same options alone or together. An option added
in both trees would fail when they run in one call.
"""

from __future__ import annotations

import pytest

from kronotop import DEFAULT_PORT

DEFAULT_URL = f"kronotop://localhost:{DEFAULT_PORT}"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--kronotop-url",
        default=DEFAULT_URL,
        help=f"Kronotop server to run integration tests against [{DEFAULT_URL}]",
    )
