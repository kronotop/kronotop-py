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

"""Async fixtures, mirroring ``tests/conftest.py``."""

from __future__ import annotations

import uuid

import pytest
import redis

from kronotop import (
    BarrierNotSatisfiedError,
    BucketBeingRemovedError,
    NamespaceBeingRemovedError,
    NoSuchBucketError,
    NoSuchNamespaceError,
    NotCommittedError,
)


@pytest.fixture
async def kr(kronotop_url):
    """A connected async client, closed on teardown. Skips if no server is up."""
    from kronotop.asyncio import Kronotop

    client = Kronotop.from_url(kronotop_url)
    try:
        await client.execute_command("PING")
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"no Kronotop at {kronotop_url}: {exc}")
    yield client
    await client.aclose()


@pytest.fixture
async def session(kr):
    """A session on the ``kr`` client, closed on teardown."""
    async with kr.session() as s:
        yield s


@pytest.fixture
async def admin(kronotop_admin_host):
    """A connected async admin client, closed on teardown. Skips if 3320 is silent."""
    from kronotop.asyncio import Admin

    client = Admin(kronotop_admin_host)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"no Kronotop admin port at {kronotop_admin_host}: {exc}")
    yield client
    await client.aclose()


@pytest.fixture
def timeout_once():
    """See the sync twin. The stub is a coroutine here."""

    def factory(conn):
        real = conn.read_response

        async def stub(*_args, **_kwargs):
            conn.read_response = real
            raise redis.TimeoutError("Timeout reading from localhost:5484")

        return stub

    return factory


@pytest.fixture
async def key(request, session):
    """See the sync twin. The name carries ``async`` so the twins never share one."""
    name = f"kronotop-py:async:{request.node.name}"
    await session.zmap.delete(name)
    yield name
    await session.zmap.delete(name)


async def drop_namespace(session, name):
    """See the sync twin."""
    try:
        await session.namespace.remove(name)
    except NoSuchNamespaceError:
        return
    except NamespaceBeingRemovedError:
        pass
    try:
        await session.namespace.purge(name)
    except BarrierNotSatisfiedError:
        await session.namespace.purge(name)


@pytest.fixture
async def namespaces(session):
    """See the sync twin. The factory it hands out is a coroutine function."""
    made = []

    async def make(create=True):
        name = f"test.kronotop_py_async_{uuid.uuid4().hex}"
        if create:
            assert await session.namespace.create(name) is True
        made.append(name)
        return name

    try:
        yield make
    finally:
        await session.namespace.use("global")
        for name in made:
            await drop_namespace(session, name)


@pytest.fixture
async def namespace(namespaces):
    """One namespace of this test's own."""
    return await namespaces()


async def drop_bucket(session, name, namespace=None):
    """See the sync twin."""
    try:
        try:
            await session.bucket.remove(name, namespace=namespace)
        except NotCommittedError:
            await session.bucket.remove(name, namespace=namespace)
    except NoSuchBucketError:
        return
    except BucketBeingRemovedError:
        pass
    try:
        await session.bucket.purge(name, namespace=namespace)
    except BarrierNotSatisfiedError:
        await session.bucket.purge(name, namespace=namespace)


@pytest.fixture
async def buckets(session):
    """See the sync twin. The factory it hands out is a coroutine function."""
    made = []

    async def make(create=True, namespace=None):
        name = f"kronotop_py_async_{uuid.uuid4().hex}"
        if create:
            assert await session.bucket.create(name, namespace=namespace) is True
        made.append((name, namespace))
        return name

    try:
        yield make
    finally:
        for name, namespace in made:
            await drop_bucket(session, name, namespace)


@pytest.fixture
async def bucket(buckets):
    """One bucket of this test's own."""
    return await buckets()
