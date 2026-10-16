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

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop import (
    Kronotop,
    NamespaceAlreadyExistsError,
    NamespaceBeingRemovedError,
    NoSuchNamespaceError,
)
from kronotop.commands.namespace import Namespace

MISSING = "no.such.namespace"


def test_the_group_is_on_the_session():
    """``session.namespace`` is built with the session, so it needs no server."""
    session = Kronotop().session()
    assert isinstance(session.namespace, Namespace)
    # Same object every time, so a caller can hold on to it.
    assert session.namespace is session.namespace


def test_the_group_is_not_on_the_client():
    """The namespace a client set would be gone one command later."""
    assert not hasattr(Kronotop(), "namespace")


@pytest.mark.integration
def test_current(session):
    assert session.namespace.current() == "global"


@pytest.mark.integration
def test_use(session, namespace):
    assert session.namespace.use(namespace) is True
    assert session.namespace.current() == namespace


@pytest.mark.integration
def test_use_on_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        session.namespace.use(MISSING)


@pytest.mark.integration
def test_create_makes_the_levels_above_it(session, namespace):
    child = f"{namespace}.a.b"
    assert session.namespace.create(child) is True
    assert session.namespace.use(f"{namespace}.a") is True


@pytest.mark.integration
def test_create_if_not_exists(session, namespace):
    assert session.namespace.create(namespace, if_not_exists=True) is True


@pytest.mark.integration
def test_create_if_not_exists_on_a_removed_namespace(session, namespace):
    session.namespace.remove(namespace)
    with pytest.raises(NamespaceBeingRemovedError):
        session.namespace.create(namespace, if_not_exists=True)


@pytest.mark.integration
def test_remove(session, namespace):
    assert session.namespace.remove(namespace) is True
    with pytest.raises(NamespaceBeingRemovedError):
        session.namespace.use(namespace)


@pytest.mark.integration
def test_remove_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        session.namespace.remove(MISSING)


@pytest.mark.integration
def test_exists(session, namespace):
    assert session.namespace.exists(namespace) is True


@pytest.mark.integration
def test_exists_on_a_missing_namespace(session):
    assert session.namespace.exists(MISSING) is False


@pytest.mark.integration
def test_exists_on_a_removed_namespace(session, namespace):
    """A removed namespace is neither there nor gone until it is purged."""
    session.namespace.remove(namespace)
    with pytest.raises(NamespaceBeingRemovedError):
        session.namespace.exists(namespace)


@pytest.mark.integration
def test_list_the_top_level(session):
    assert "global" in session.namespace.list()


@pytest.mark.integration
def test_list_gives_leaves_not_paths(session, namespace):
    session.namespace.create(f"{namespace}.a")
    session.namespace.create(f"{namespace}.b")
    assert sorted(session.namespace.list(namespace)) == ["a", "b"]


@pytest.mark.integration
def test_list_a_namespace_with_nothing_under_it(session, namespace):
    assert session.namespace.list(namespace) == []


@pytest.mark.integration
def test_list_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        session.namespace.list(MISSING)


@pytest.mark.integration
def test_move(session, namespaces):
    old = namespaces()
    # The move creates it, so the fixture only has to know the name.
    new = namespaces(create=False)
    assert session.namespace.move(old, new) is True
    assert session.namespace.exists(old) is False
    assert session.namespace.exists(new) is True


@pytest.mark.integration
def test_move_carries_the_data(session, namespaces):
    old = namespaces()
    new = namespaces(create=False)
    session.namespace.use(old)
    session.zmap.set("moved", "value")
    session.namespace.use("global")
    session.namespace.move(old, new)
    session.namespace.use(new)
    assert session.zmap.get("moved") == b"value"


@pytest.mark.integration
def test_move_onto_a_name_that_is_taken(session, namespaces, namespace):
    with pytest.raises(NamespaceAlreadyExistsError):
        session.namespace.move(namespaces(), namespace)


@pytest.mark.integration
def test_move_a_missing_namespace(session, namespaces):
    with pytest.raises(NoSuchNamespaceError):
        session.namespace.move(MISSING, namespaces(create=False))


@pytest.mark.integration
def test_purge(session, namespace):
    session.namespace.remove(namespace)
    assert session.namespace.purge(namespace) is True
    assert session.namespace.exists(namespace) is False


@pytest.mark.integration
def test_purge_takes_the_namespaces_below_it(session, namespace):
    session.namespace.create(f"{namespace}.a")
    session.namespace.remove(namespace)
    session.namespace.purge(namespace)
    assert session.namespace.exists(f"{namespace}.a") is False


@pytest.mark.integration
def test_purge_without_a_removal(session, namespace):
    with pytest.raises(ResponseError, match="must be logically removed"):
        session.namespace.purge(namespace)


@pytest.mark.integration
def test_purge_a_missing_namespace(session):
    with pytest.raises(NoSuchNamespaceError):
        session.namespace.purge(MISSING)


@pytest.mark.integration
def test_purge_the_default_namespace(session):
    with pytest.raises(ResponseError, match="default namespace"):
        session.namespace.purge("global")
