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

"""The store against a server."""

import pytest

pytestmark = pytest.mark.integration

NO_CURSORS = {"query": {}, "update": {}, "delete": {}}


@pytest.fixture
def memories(store):
    """Three memories of two users."""
    store.insert(
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
        payloads=[
            {"user_id": "u1", "data": "likes tea"},
            {"user_id": "u1", "data": "lives in Izmir"},
            {"user_id": "u2", "data": "likes coffee"},
        ],
        ids=["m1", "m2", "m3"],
    )
    return store


@pytest.fixture
def cursors(store):
    """The open cursors of the one session the pool holds."""

    def read():
        with store._pool.borrow() as session:
            return session.bucket.cursors()

    return read


def test_get_gives_the_memory(memories):
    memory = memories.get("m1")
    assert memory.id == "m1"
    assert memory.score is None
    assert memory.payload == {"user_id": "u1", "data": "likes tea"}


def test_get_of_an_unknown_id_gives_none(memories, cursors):
    assert memories.get("nothing") is None
    assert cursors() == NO_CURSORS


def test_update_of_the_payload_replaces_it(memories, cursors):
    memories.update("m1", payload={"user_id": "u1", "data": "likes green tea"})
    assert memories.get("m1").payload == {"user_id": "u1", "data": "likes green tea"}
    assert cursors() == NO_CURSORS


def test_update_of_the_vector_moves_the_memory(memories):
    memories.update("m2", vector=[1.0, 0.0, 0.0])
    found = memories.search("", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "u1"})
    assert {memory.id for memory in found[:2]} == {"m1", "m2"}
    assert memories.get("m2").payload == {"user_id": "u1", "data": "lives in Izmir"}


def test_delete_removes_the_memory(memories, cursors):
    memories.delete("m1")
    assert memories.get("m1") is None
    assert memories.get("m2") is not None
    assert cursors() == NO_CURSORS


def test_list_applies_the_filter(memories, cursors):
    listed = memories.list(filters={"user_id": "u1"})
    assert {memory.id for memory in listed} == {"m1", "m2"}
    assert cursors() == NO_CURSORS


def test_list_stops_at_top_k(memories, cursors):
    assert len(memories.list(top_k=2)) == 2
    assert cursors() == NO_CURSORS


def test_search_gives_the_closest_first(memories):
    found = memories.search("", [1.0, 0.0, 0.0], top_k=2)
    assert [memory.id for memory in found] == ["m1", "m3"]
    assert found[0].score == pytest.approx(1.0)
    assert found[0].payload == {"user_id": "u1", "data": "likes tea"}


def test_search_applies_the_filter(memories):
    found = memories.search("", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "u2"})
    assert [memory.id for memory in found] == ["m3"]


def test_list_cols_names_the_bucket(store):
    assert store.list_cols() == ["memories"]


def test_col_info_names_the_indexes(store):
    info = store.col_info()
    assert info["name"] == "memories"
    assert info["namespace"] == store.namespace
    assert len(info["indexes"]) == 3


def test_reset_empties_the_bucket(memories):
    memories.reset()
    assert memories.list() == []
