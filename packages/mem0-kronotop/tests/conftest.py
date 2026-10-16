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

"""Fixtures of the store tests. They skip when no server is reachable."""

import uuid

import pytest

from kronotop import BarrierNotSatisfiedError, Kronotop

DEFAULT_URL = "kronotop://localhost:5484"

#: Length of the vectors in the tests.
DIMENSIONS = 3


@pytest.fixture
def kronotop_url(request):
    # ``tests/conftest.py`` of the client adds the option, and adding it here
    # again fails when both trees run. Without that tree the default holds.
    return request.config.getoption("--kronotop-url", default=DEFAULT_URL)


@pytest.fixture
def store(kronotop_url):
    """A store in a namespace of this test's own, dropped on the way out."""
    from mem0_kronotop.store import KronotopVectorStore

    with Kronotop.from_url(kronotop_url) as client:
        try:
            client.execute_command("PING")
        except Exception as exc:
            pytest.skip(f"no Kronotop at {kronotop_url}: {exc}")

        namespace = f"test.mem0_kronotop_{uuid.uuid4().hex}"
        store = KronotopVectorStore(
            namespace, "memories", DIMENSIONS, bootstrap=client.hosts
        )
        try:
            yield store
        finally:
            store.delete_col()
            store.client.close()
            with client.session() as session:
                session.namespace.remove(namespace)
                try:
                    session.namespace.purge(namespace)
                except BarrierNotSatisfiedError:
                    session.namespace.purge(namespace)
