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

"""Async twin of ``tests/test_namespace.py``."""

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop import (
    NamespaceAlreadyExistsError,
    NamespaceBeingRemovedError,
    NoSuchNamespaceError,
)
from kronotop.asyncio import Kronotop
from kronotop.commands.namespace import AsyncNamespace

MISSING = "no.such.namespace"


async def test_the_group_is_on_the_session():
    session = Kronotop().session()
    assert isinstance(session.namespace, AsyncNamespace)
    assert session.namespace is session.namespace


def test_the_group_is_not_on_the_client():
    assert not hasattr(Kronotop(), "namespace")


@pytest.mark.integration
async def test_current(session):
    assert await session.namespace.current() == "global"


@pytest.mark.integration
async def test_use(session, namespace):
    assert await session.namespace.use(namespace) is True
    assert await session.namespace.current() == namespace


@pytest.mark.integration
async def test_use_on_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        await session.namespace.use(MISSING)


@pytest.mark.integration
async def test_create_makes_the_levels_above_it(session, namespace):
    assert await session.namespace.create(f"{namespace}.a.b") is True
    assert await session.namespace.use(f"{namespace}.a") is True


@pytest.mark.integration
async def test_create_if_not_exists(session, namespace):
    assert await session.namespace.create(namespace, if_not_exists=True) is True


@pytest.mark.integration
async def test_create_if_not_exists_on_a_removed_namespace(session, namespace):
    await session.namespace.remove(namespace)
    with pytest.raises(NamespaceBeingRemovedError):
        await session.namespace.create(namespace, if_not_exists=True)


@pytest.mark.integration
async def test_remove(session, namespace):
    assert await session.namespace.remove(namespace) is True
    with pytest.raises(NamespaceBeingRemovedError):
        await session.namespace.use(namespace)


@pytest.mark.integration
async def test_remove_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        await session.namespace.remove(MISSING)


@pytest.mark.integration
async def test_exists(session, namespace):
    assert await session.namespace.exists(namespace) is True


@pytest.mark.integration
async def test_exists_on_a_missing_namespace(session):
    assert await session.namespace.exists(MISSING) is False


@pytest.mark.integration
async def test_exists_on_a_removed_namespace(session, namespace):
    await session.namespace.remove(namespace)
    with pytest.raises(NamespaceBeingRemovedError):
        await session.namespace.exists(namespace)


@pytest.mark.integration
async def test_list_the_top_level(session):
    assert "global" in await session.namespace.list()


@pytest.mark.integration
async def test_list_gives_leaves_not_paths(session, namespace):
    await session.namespace.create(f"{namespace}.a")
    await session.namespace.create(f"{namespace}.b")
    assert sorted(await session.namespace.list(namespace)) == ["a", "b"]


@pytest.mark.integration
async def test_list_a_namespace_with_nothing_under_it(session, namespace):
    assert await session.namespace.list(namespace) == []


@pytest.mark.integration
async def test_list_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        await session.namespace.list(MISSING)


@pytest.mark.integration
async def test_move(session, namespaces):
    old = await namespaces()
    new = await namespaces(create=False)
    assert await session.namespace.move(old, new) is True
    assert await session.namespace.exists(old) is False
    assert await session.namespace.exists(new) is True


@pytest.mark.integration
async def test_move_carries_the_data(session, namespaces):
    old = await namespaces()
    new = await namespaces(create=False)
    await session.namespace.use(old)
    await session.zmap.set("moved", "value")
    await session.namespace.use("global")
    await session.namespace.move(old, new)
    await session.namespace.use(new)
    assert await session.zmap.get("moved") == b"value"


@pytest.mark.integration
async def test_move_onto_a_name_that_is_taken(session, namespaces, namespace):
    with pytest.raises(NamespaceAlreadyExistsError):
        await session.namespace.move(await namespaces(), namespace)


@pytest.mark.integration
async def test_move_a_missing_namespace(session, namespaces):
    with pytest.raises(NoSuchNamespaceError):
        await session.namespace.move(MISSING, await namespaces(create=False))


@pytest.mark.integration
async def test_purge(session, namespace):
    await session.namespace.remove(namespace)
    assert await session.namespace.purge(namespace) is True
    assert await session.namespace.exists(namespace) is False


@pytest.mark.integration
async def test_purge_takes_the_namespaces_below_it(session, namespace):
    await session.namespace.create(f"{namespace}.a")
    await session.namespace.remove(namespace)
    await session.namespace.purge(namespace)
    assert await session.namespace.exists(f"{namespace}.a") is False


@pytest.mark.integration
async def test_purge_without_a_removal(session, namespace):
    with pytest.raises(ResponseError, match="must be logically removed"):
        await session.namespace.purge(namespace)


@pytest.mark.integration
async def test_purge_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        await session.namespace.purge(MISSING)


@pytest.mark.integration
async def test_purge_the_default_namespace(session):
    with pytest.raises(ResponseError, match="default namespace"):
        await session.namespace.purge("global")
