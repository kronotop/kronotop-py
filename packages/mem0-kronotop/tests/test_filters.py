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

"""mem0 filters as BQL, without a server."""

import pytest

from mem0_kronotop._filters import to_bql


@pytest.mark.parametrize("filters", [None, {}])
def test_no_filter_matches_every_document(filters):
    assert to_bql(filters) == {}


def test_a_plain_value_is_an_equality():
    assert to_bql({"user_id": "u1", "agent_id": "a1"}) == {
        "payload.user_id": "u1",
        "payload.agent_id": "a1",
    }


def test_a_wildcard_asks_for_the_field():
    assert to_bql({"category": "*"}) == {"payload.category": {"$exists": True}}


@pytest.mark.parametrize("operator", ["eq", "ne", "gt", "gte", "lt", "lte"])
def test_a_comparison_gets_its_bql_name(operator):
    assert to_bql({"age": {operator: 30}}) == {"payload.age": {f"${operator}": 30}}


@pytest.mark.parametrize("operator", ["in", "nin"])
def test_a_list_operator_gets_its_bql_name(operator):
    assert to_bql({"tag": {operator: ["a", "b"]}}) == {
        "payload.tag": {f"${operator}": ["a", "b"]}
    }


def test_operators_on_one_field_stay_together():
    assert to_bql({"age": {"gte": 18, "lt": 65}}) == {
        "payload.age": {"$gte": 18, "$lt": 65}
    }


def test_contains_is_a_regex_with_the_text_escaped():
    assert to_bql({"data": {"contains": "a.b"}}) == {
        "payload.data": {"$regex": "a\\.b"}
    }


def test_icontains_ignores_case():
    assert to_bql({"data": {"icontains": "tea"}}) == {
        "payload.data": {"$regex": "tea", "$options": "i"}
    }


@pytest.mark.parametrize(
    ("key", "operator"),
    [
        ("AND", "$and"),
        ("$and", "$and"),
        ("OR", "$or"),
        ("$or", "$or"),
        ("NOT", "$nor"),
        ("$not", "$nor"),
    ],
)
def test_a_logical_operator_gets_its_bql_name(key, operator):
    assert to_bql({key: [{"role": "user"}, {"age": {"gt": 3}}]}) == {
        operator: [{"payload.role": "user"}, {"payload.age": {"$gt": 3}}]
    }


def test_logical_operators_nest():
    filters = {"user_id": "u1", "$or": [{"NOT": [{"role": "user"}]}, {"age": 3}]}
    assert to_bql(filters) == {
        "payload.user_id": "u1",
        "$or": [{"$nor": [{"payload.role": "user"}]}, {"payload.age": 3}],
    }


def test_the_first_of_two_spellings_wins():
    filters = {"$or": [{"role": "user"}], "OR": [{"role": "user"}]}
    assert to_bql(filters) == {"$or": [{"payload.role": "user"}]}


def test_an_unknown_operator_is_an_error():
    with pytest.raises(ValueError, match="'near'"):
        to_bql({"place": {"near": "here"}})
