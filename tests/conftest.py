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

"""Shared fixtures. ``integration`` tests skip when no server is reachable."""

from __future__ import annotations

import inspect
import uuid

import pytest
import redis

from kronotop import (
    DEFAULT_ADMIN_PORT,
    DEFAULT_PORT,
    BarrierNotSatisfiedError,
    BucketBeingRemovedError,
    NamespaceBeingRemovedError,
    NoSuchBucketError,
    NoSuchNamespaceError,
    NotCommittedError,
    _members,
)

DEFAULT_URL = f"kronotop://localhost:{DEFAULT_PORT}"
DEFAULT_ADMIN_HOST = f"localhost:{DEFAULT_ADMIN_PORT}"


@pytest.fixture(autouse=True)
def no_lookups(monkeypatch):
    """Keep ``Members.add`` off the resolver.

    It looks a hostname up, and a test naming a member ``a`` would send that to
    a name server. A search domain could even answer, and then two made up
    names would be one member. A test that wants a lookup patches these again,
    which runs after this.

    The spellings of an address still fold, so an integration test still sees
    ``127.0.0.1`` and ``localhost`` as one member.
    """
    monkeypatch.setattr(_members, "resolve", lambda host: frozenset())

    async def aresolve(host):
        return frozenset()

    monkeypatch.setattr(_members, "aresolve", aresolve)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--kronotop-url",
        default=DEFAULT_URL,
        help=f"Kronotop server to run integration tests against [{DEFAULT_URL}]",
    )
    parser.addoption(
        "--kronotop-admin-host",
        default=DEFAULT_ADMIN_HOST,
        help=f"Admin port of a member, host:port [{DEFAULT_ADMIN_HOST}]",
    )
    parser.addoption(
        "--kronotop-password",
        default=None,
        help=(
            "Password of a Kronotop started with auth.requirepass. Tests that "
            "need authentication are skipped without it."
        ),
    )


@pytest.fixture(scope="session")
def kronotop_url(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--kronotop-url"))


@pytest.fixture(scope="session")
def kronotop_admin_host(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--kronotop-admin-host"))


@pytest.fixture(scope="session")
def kronotop_password(request: pytest.FixtureRequest) -> str | None:
    """The password to run the auth tests with, or ``None`` to skip them."""
    password = request.config.getoption("--kronotop-password")
    if password is None:
        pytest.skip("needs --kronotop-password and a server with auth.requirepass")
    return str(password)


async def open_session(client_class, **kwargs):
    """A session from either client kind, for a test that covers both.

    The asyncio ``session`` returns an awaitable, the sync one returns the
    session. Only a test that opens no connection can share one body this way.
    """
    session = client_class().session(**kwargs)
    return await session if inspect.isawaitable(session) else session


@pytest.fixture
def only_pool():
    """The pool of a client that has one member.

    There is no default member, so a one member client is the only case where
    "the pool" means something.
    """

    def pool_of(client):
        (pool,) = client._pools.values()
        return pool

    return pool_of


@pytest.fixture
def kr(request, kronotop_url):
    """A connected client, closed on teardown. Skips if no server is up."""
    from kronotop import Kronotop

    client = Kronotop.from_url(kronotop_url)
    try:
        client.execute_command("PING")
    except Exception as exc:
        client.close()
        pytest.skip(f"no Kronotop at {kronotop_url}: {exc}")
    request.addfinalizer(client.close)
    return client


@pytest.fixture
def session(kr):
    """A session on the ``kr`` client, closed on teardown."""
    with kr.session() as s:
        yield s


@pytest.fixture
def admin(request, kronotop_admin_host):
    """A connected admin client, closed on teardown. Skips if 3320 is silent."""
    from kronotop import Admin

    client = Admin(kronotop_admin_host)
    try:
        client.ping()
    except Exception as exc:
        client.close()
        pytest.skip(f"no Kronotop admin port at {kronotop_admin_host}: {exc}")
    request.addfinalizer(client.close)
    return client


@pytest.fixture
def timeout_once():
    """A factory of ``read_response`` stubs that raise once, then restore the real one.

    A fixture, not a helper: with ``--import-mode=importlib`` one test module
    cannot import from another.
    """

    def factory(conn):
        real = conn.read_response

        def stub(*_args, **_kwargs):
            conn.read_response = real
            raise redis.TimeoutError("Timeout reading from localhost:5484")

        return stub

    return factory


@pytest.fixture
def key(request, session):
    """A ZMap key named after the test, deleted before and after it runs.

    In the default namespace, which nothing clears. A test that writes a key
    without this leaves it on the server for good.
    """
    name = f"kronotop-py:{request.node.name}"
    session.zmap.delete(name)
    yield name
    session.zmap.delete(name)


def drop_namespace(session, name):
    """Remove ``name`` and purge it, so the server keeps nothing.

    ``PURGE`` needs every live member to have seen the removal, and answers
    ``BARRIERNOTSATISFIED`` until they have. The failed attempt republishes the
    event, so one retry is enough on a healthy cluster.

    A test may have removed, purged or moved the name itself. Each of those
    leaves the drop with less to do.
    """
    try:
        session.namespace.remove(name)
    except NoSuchNamespaceError:
        return
    except NamespaceBeingRemovedError:
        pass
    try:
        session.namespace.purge(name)
    except BarrierNotSatisfiedError:
        session.namespace.purge(name)


@pytest.fixture
def namespaces(session):
    """Names namespaces for one test and drops every one on the way out.

    Call it for a name of this test's own. ``NAMESPACE REMOVE`` only marks a
    name, so a name is good for one run: until ``NAMESPACE PURGE`` runs, it is
    neither there nor free. ``create=False`` names one without creating it, for
    a test that makes it some other way.
    """
    made = []

    def make(create=True):
        name = f"test.kronotop_py_{uuid.uuid4().hex}"
        if create:
            assert session.namespace.create(name) is True
        made.append(name)
        return name

    try:
        yield make
    finally:
        session.namespace.use("global")
        for name in made:
            drop_namespace(session, name)


@pytest.fixture
def namespace(namespaces):
    """One namespace of this test's own."""
    return namespaces()


def drop_bucket(session, name, namespace=None):
    """Remove ``name`` and purge it, so the server keeps nothing.

    The same two steps as ``drop_namespace``, and the same rule: a test may have
    removed or purged the bucket itself, which leaves this with less to do.

    The background index task writes the bucket metadata too, so ``REMOVE`` can
    lose a conflict. One retry is enough.
    """
    try:
        try:
            session.bucket.remove(name, namespace=namespace)
        except NotCommittedError:
            session.bucket.remove(name, namespace=namespace)
    except NoSuchBucketError:
        return
    except BucketBeingRemovedError:
        pass
    try:
        session.bucket.purge(name, namespace=namespace)
    except BarrierNotSatisfiedError:
        session.bucket.purge(name, namespace=namespace)


@pytest.fixture
def buckets(session):
    """Names buckets for one test and drops every one on the way out.

    A name is good for one run. ``BUCKET.REMOVE`` only marks the bucket, and the
    name stays taken until ``BUCKET.PURGE``. ``create=False`` names one without
    creating it, for a test that creates it itself.

    The buckets land in the session's current namespace, which nothing clears,
    so every name carries a uuid. ``namespace`` puts one in another namespace.
    A test that asks for that names the ``namespace`` fixture before this one,
    so the bucket is dropped before its namespace.
    """
    made = []

    def make(create=True, namespace=None):
        name = f"kronotop_py_{uuid.uuid4().hex}"
        if create:
            assert session.bucket.create(name, namespace=namespace) is True
        made.append((name, namespace))
        return name

    try:
        yield make
    finally:
        for name, namespace in made:
            drop_bucket(session, name, namespace)


@pytest.fixture
def bucket(buckets):
    """One bucket of this test's own."""
    return buckets()
