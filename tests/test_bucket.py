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

import itertools
import json
import struct
import time

import pytest
import redis
from bson import ObjectId, encode
from bson.json_util import dumps
from redis.exceptions import ResponseError

from kronotop import (
    BucketAlreadyExistsError,
    BucketBeingRemovedError,
    DuplicateKeyError,
    Kronotop,
    KronotopConfig,
    NoSuchBucketError,
    NoSuchIndexError,
    NoSuchNamespaceError,
    VectorIndexNotReadyError,
)
from kronotop._callbacks import (
    parse_bucket_advance,
    parse_bucket_delete,
    parse_bucket_explain,
    parse_bucket_index,
    parse_bucket_vector,
)
from kronotop._members import member_key
from kronotop.commands.bucket import (
    Bucket,
    BucketLocation,
    Collation,
    CompoundIndex,
    CompoundIndexField,
    Cursor,
    ObjectIdBatch,
    QueryBatch,
    ShardLocation,
    SingleFieldIndex,
    VectorIndex,
    VectorSearchResult,
    _index_schema,
    _insert_args,
    _query_args,
    _vector_args,
)

MISSING = "no-such-bucket"

#: The index and the documents of the ``BUCKET.VECTOR`` docs, in that order.
VECTOR_INDEX = VectorIndex("embedding", 3, "cosine")
VECTOR_DOCUMENTS = [
    {"label": "alpha", "embedding": [0.1, 0.2, 0.3]},
    {"label": "beta", "embedding": [0.4, 0.5, 0.6]},
    {"label": "gamma", "embedding": [0.7, 0.8, 0.9]},
]
QUERY_VECTOR = [0.4, 0.5, 0.6]


@pytest.fixture
def vector_bucket(session, bucket):
    """``bucket`` with the docs' vector index and its three documents.

    A bucket created without ``shards`` sits on one shard, which is what a
    vector index needs.
    """
    session.bucket.create_index(bucket, VECTOR_INDEX)
    session.bucket.insert(bucket, *VECTOR_DOCUMENTS)
    return bucket


def wait_for(condition, what, retry_on=ResponseError):
    """Call ``condition`` until it gives a true value, for up to 45 seconds.

    A background task drives every index change: the build, the drop, the
    analyze, and the members learn of a new index a moment later. Until then
    the server can answer with an error, so an error of ``retry_on`` counts as
    not yet. A build takes about 20 seconds: a boundary task, then the build.
    """
    deadline = time.monotonic() + 45
    while True:
        try:
            result = condition()
        except retry_on:
            result = None
        if result:
            return result
        if time.monotonic() > deadline:
            raise AssertionError(f"{what} did not settle")
        time.sleep(0.1)


def search(session, bucket, expected=3, **options):
    """``vector`` once the index is ready and holds ``expected`` results.

    The index is built and fed after the commit, so the first calls can raise
    ``VectorIndexNotReadyError`` or come back short.
    """

    def settled():
        results = session.bucket.vector(bucket, "embedding", QUERY_VECTOR, **options)
        return results if len(results) >= expected else None

    return wait_for(settled, "vector search", VectorIndexNotReadyError)


def test_the_group_is_on_the_session():
    """``session.bucket`` is built with the session, so it needs no server."""
    session = Kronotop().session()
    assert isinstance(session.bucket, Bucket)
    # Same object every time, so a caller can hold on to it.
    assert session.bucket is session.bucket


def test_the_group_is_not_on_the_client():
    """A bucket belongs to a namespace, and the namespace is session state."""
    assert not hasattr(Kronotop(), "bucket")


def test_a_collation_sends_only_what_was_set():
    """A field left at ``None`` must not reach the server as a default."""
    assert Collation("tr").to_dict() == {"locale": "tr"}
    assert Collation("en", strength=2, numeric_ordering=True).to_dict() == {
        "locale": "en",
        "strength": 2,
        "numeric_ordering": True,
    }


def test_an_index_sends_only_what_was_set():
    """The selector is the schema key, so ``to_dict`` leaves it out."""
    assert SingleFieldIndex("age", "int32").to_dict() == {"bson_type": "int32"}
    index = SingleFieldIndex("name", "string", unique=True, collation=Collation("tr"))
    assert index.to_dict() == {
        "bson_type": "string",
        "unique": True,
        "collation": {"locale": "tr"},
    }
    assert CompoundIndexField("age", "int32").to_dict() == {
        "selector": "age",
        "bson_type": "int32",
    }
    assert VectorIndex("embedding", 3, "cosine", name="emb").to_dict() == {
        "field": "embedding",
        "dimensions": 3,
        "distance": "cosine",
        "name": "emb",
    }


def test_a_compound_index_keeps_the_field_order():
    fields = [
        CompoundIndexField("a", "string"),
        CompoundIndexField("b", "int32", multi_key=True),
    ]
    assert CompoundIndex(fields, unique=True).to_dict() == {
        "fields": [
            {"selector": "a", "bson_type": "string"},
            {"selector": "b", "bson_type": "int32", "multi_key": True},
        ],
        "unique": True,
    }


def test_an_index_schema_takes_every_kind():
    """Each kind lands under its own key, and a mapping is merged as it is."""
    schema = _index_schema(
        [
            SingleFieldIndex("email", "string"),
            CompoundIndex(
                [CompoundIndexField("a", "string"), CompoundIndexField("b", "int32")]
            ),
            VectorIndex("embedding", 3, "cosine"),
            {"age": {"bson_type": "int32"}},
        ]
    )
    assert json.loads(schema) == {
        "email": {"bson_type": "string"},
        "$compound": [
            {
                "fields": [
                    {"selector": "a", "bson_type": "string"},
                    {"selector": "b", "bson_type": "int32"},
                ]
            }
        ],
        "$vector": {"field": "embedding", "dimensions": 3, "distance": "cosine"},
        "age": {"bson_type": "int32"},
    }


def test_an_index_schema_needs_an_index():
    with pytest.raises(ValueError):
        _index_schema([])


def test_an_index_schema_rejects_two_indexes_on_one_selector():
    """A dict would keep the last one and the server would never see the first."""
    with pytest.raises(ValueError):
        _index_schema(
            [SingleFieldIndex("age", "int32"), SingleFieldIndex("age", "int64")]
        )


def test_an_index_schema_takes_one_vector_index():
    with pytest.raises(ValueError):
        _index_schema([VectorIndex("a", 3, "cosine"), VectorIndex("b", 3, "cosine")])


def test_an_index_schema_checks_a_mapping_too():
    """A mapping gets the same checks as a dataclass, in either order."""
    index = SingleFieldIndex("age", "int32")
    schema = {"age": {"bson_type": "int64"}}
    with pytest.raises(ValueError):
        _index_schema([index, schema])
    with pytest.raises(ValueError):
        _index_schema([schema, index])
    vector = VectorIndex("a", 3, "cosine")
    with pytest.raises(ValueError):
        _index_schema([vector, {"$vector": vector.to_dict()}])


def test_an_index_schema_copies_a_compound_list():
    """The caller's list is not appended to, and a tuple works too."""
    compound = CompoundIndex(
        [CompoundIndexField("a", "string"), CompoundIndexField("b", "int32")]
    )
    schema = {"$compound": (compound.to_dict(),)}
    assert len(json.loads(_index_schema([schema, compound]))["$compound"]) == 2
    assert schema == {"$compound": (compound.to_dict(),)}


def test_an_index_sends_a_dict_collation_as_it_is():
    """Only a dataclass loses its ``None`` fields. The escape hatch is kept."""
    index = SingleFieldIndex("name", "string", collation={"locale": "tr", "x": None})
    assert index.to_dict()["collation"] == {"locale": "tr", "x": None}


def test_parse_bucket_index_collapses_ok_alone():
    """``LIST`` and ``DESCRIBE`` come back as they came."""
    assert parse_bucket_index(b"OK") is True
    assert parse_bucket_index([b"idx_age"]) == [b"idx_age"]


def test_parse_bucket_index_decodes_a_list():
    assert parse_bucket_index([b"idx_age"], index_subcommand="LIST") == ["idx_age"]


#: One compound index with a collation, the way RESP2 and RESP3 send it.
DESCRIBE_RESP2 = [
    b"index_type",
    b"compound",
    b"id",
    3,
    b"fields",
    [[b"selector", b"status", b"bson_type", b"STRING"]],
    b"status",
    b"READY",
    b"unique",
    1,
    b"collation",
    [b"locale", b"tr", b"strength", 2, b"case_level", 0],
    b"statistics",
    [b"cardinality", 0],
]
DESCRIBE_RESP3 = {
    b"index_type": b"compound",
    b"id": 3,
    b"fields": [{b"selector": b"status", b"bson_type": b"STRING"}],
    b"status": b"READY",
    b"unique": True,
    b"collation": {b"locale": b"tr", b"strength": 2, b"case_level": False},
    b"statistics": {b"cardinality": 0},
}
DESCRIBED = {
    "index_type": "compound",
    "id": 3,
    "fields": [{"selector": "status", "bson_type": "STRING"}],
    "status": "READY",
    "unique": True,
    "collation": {"locale": "tr", "strength": 2, "case_level": False},
    "statistics": {"cardinality": 0},
}


def test_parse_bucket_index_describes_on_either_protocol():
    """RESP2 sends a flat array and ``1`` for a boolean, RESP3 a map and
    ``True``. Both give one dict, with ``str`` keys and ``bool`` flags."""
    assert parse_bucket_index(DESCRIBE_RESP2, index_subcommand="DESCRIBE") == DESCRIBED
    assert parse_bucket_index(DESCRIBE_RESP3, index_subcommand="DESCRIBE") == DESCRIBED


#: One range scan with a residual filter after it, and a query collation, the
#: way RESP2 and RESP3 send it. RESP2 has no map and no boolean, so a map is a
#: flat array and a boolean is ``1`` or ``0``.
EXPLAIN_RESP2 = [
    b"is_cached",
    0,
    b"query_collation",
    [b"locale", b"tr", b"strength", 2, b"case_level", 1],
    b"plan",
    [
        b"planner_version",
        1,
        b"nodeType",
        b"RangeScan",
        b"id",
        7,
        b"scanType",
        b"RANGE_SCAN",
        b"index",
        b"primary-index",
        b"selector",
        b"_id",
        b"lowerBound",
        b"0",
        b"upperBound",
        None,
        b"includeLower",
        1,
        b"includeUpper",
        0,
        b"next",
        [
            b"planner_version",
            1,
            b"nodeType",
            b"TransformWithResidualPredicate",
            b"id",
            9,
            b"operation",
            b"FILTER",
            b"predicate",
            [
                b"type",
                b"AND",
                b"children",
                [
                    [
                        b"type",
                        b"PREDICATE",
                        b"selector",
                        b"tags",
                        b"operator",
                        b"IN",
                        b"operand",
                        [b"a", 2],
                    ]
                ],
            ],
        ],
    ],
]
EXPLAIN_RESP3 = {
    b"is_cached": False,
    b"query_collation": {b"locale": b"tr", b"strength": 2, b"case_level": True},
    b"plan": {
        b"planner_version": 1,
        b"nodeType": b"RangeScan",
        b"id": 7,
        b"scanType": b"RANGE_SCAN",
        b"index": b"primary-index",
        b"selector": b"_id",
        b"lowerBound": b"0",
        b"upperBound": None,
        b"includeLower": True,
        b"includeUpper": False,
        b"next": {
            b"planner_version": 1,
            b"nodeType": b"TransformWithResidualPredicate",
            b"id": 9,
            b"operation": b"FILTER",
            b"predicate": {
                b"type": b"AND",
                b"children": [
                    {
                        b"type": b"PREDICATE",
                        b"selector": b"tags",
                        b"operator": b"IN",
                        b"operand": [b"a", 2],
                    }
                ],
            },
        },
    },
}
EXPLAINED = {
    "is_cached": False,
    "query_collation": {"locale": "tr", "strength": 2, "case_level": True},
    "plan": {
        "planner_version": 1,
        "nodeType": "RangeScan",
        "id": 7,
        "scanType": "RANGE_SCAN",
        "index": "primary-index",
        "selector": "_id",
        "lowerBound": "0",
        "upperBound": None,
        "includeLower": True,
        "includeUpper": False,
        "next": {
            "planner_version": 1,
            "nodeType": "TransformWithResidualPredicate",
            "id": 9,
            "operation": "FILTER",
            "predicate": {
                "type": "AND",
                "children": [
                    {
                        "type": "PREDICATE",
                        "selector": "tags",
                        "operator": "IN",
                        "operand": ["a", 2],
                    }
                ],
            },
        },
    },
}


def test_parse_bucket_explain_on_either_protocol():
    """Both protocols give one dict: ``str`` keys, ``bool`` flags, a ``None``
    bound, and the nested nodes decoded the same way."""
    assert parse_bucket_explain(EXPLAIN_RESP2) == EXPLAINED
    assert parse_bucket_explain(EXPLAIN_RESP3) == EXPLAINED


def test_parse_bucket_index_lists_tasks_on_either_protocol():
    task = [b"kind", b"DROP", b"status", b"RUNNING", b"error", b""]
    tasks = {"5K8G": {"kind": "DROP", "status": "RUNNING", "error": ""}}
    assert parse_bucket_index([b"5K8G", task], index_subcommand="TASKS") == tasks
    resp3 = {b"5K8G": dict(zip(task[::2], task[1::2], strict=True))}
    assert parse_bucket_index(resp3, index_subcommand="TASKS") == tasks
    assert parse_bucket_index([], index_subcommand="TASKS") == {}


@pytest.mark.integration
def test_create(session, buckets):
    assert session.bucket.create(buckets(create=False)) is True


@pytest.mark.integration
def test_create_on_a_taken_name(session, bucket):
    with pytest.raises(BucketAlreadyExistsError):
        session.bucket.create(bucket)


@pytest.mark.integration
def test_create_if_not_exists(session, bucket):
    assert session.bucket.create(bucket, if_not_exists=True) is True


@pytest.mark.integration
def test_create_on_a_removed_bucket(session, bucket):
    session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        session.bucket.create(bucket)


@pytest.mark.integration
def test_create_on_a_shard(session, buckets):
    name = buckets(create=False)
    assert session.bucket.create(name, shards=[0]) is True
    assert [shard.shard_id for shard in session.bucket.locate(name).shards] == [0]


@pytest.mark.integration
def test_create_with_a_collation(session, buckets):
    name = buckets(create=False)
    collation = Collation("tr", strength=2)
    assert session.bucket.create(name, collation=collation) is True


@pytest.mark.integration
def test_create_with_a_collation_dict(session, buckets):
    """The escape hatch for a field this client does not know yet."""
    name = buckets(create=False)
    assert session.bucket.create(name, collation={"locale": "tr"}) is True


@pytest.mark.integration
def test_create_with_indexes(session, buckets):
    name = buckets(create=False)
    index = SingleFieldIndex("age", "int32", name="idx_age")
    assert session.bucket.create(name, indexes=[index]) is True
    assert "idx_age" in session.bucket.list_indexes(name)


@pytest.mark.integration
def test_create_index(session, bucket):
    index = SingleFieldIndex("age", "int32", name="idx_age")
    assert session.bucket.create_index(bucket, index) is True
    assert "idx_age" in session.bucket.list_indexes(bucket)


@pytest.mark.integration
def test_create_index_with_a_compound_index(session, bucket):
    index = CompoundIndex(
        [CompoundIndexField("status", "string"), CompoundIndexField("age", "int32")],
        name="idx_status_age",
    )
    assert session.bucket.create_index(bucket, index) is True
    assert "idx_status_age" in session.bucket.list_indexes(bucket)


@pytest.mark.integration
def test_create_with_a_unique_index(session, buckets):
    """On ``create`` the index is ready at once, so the write path is asserted."""
    name = buckets(create=False)
    index = SingleFieldIndex("email", "string", unique=True)
    session.bucket.create(name, indexes=[index])
    session.bucket.insert(name, {"email": "a@kronotop.com"})
    with pytest.raises(DuplicateKeyError):
        session.bucket.insert(name, {"email": "a@kronotop.com"})


@pytest.mark.integration
def test_create_index_with_a_unique_index(session, bucket):
    """``describe_index`` says whether the index was recorded unique."""
    index = SingleFieldIndex("email", "string", unique=True, name="idx_email")
    assert session.bucket.create_index(bucket, index) is True
    assert session.bucket.describe_index(bucket, "idx_email")["unique"] is True


@pytest.mark.integration
def test_create_index_with_a_dict(session, bucket):
    """The escape hatch for a form this client does not know yet."""
    schema = {"age": {"bson_type": "int32", "name": "idx_age"}}
    assert session.bucket.create_index(bucket, schema) is True
    assert "idx_age" in session.bucket.list_indexes(bucket)


@pytest.mark.integration
def test_create_index_twice_on_one_field(session, bucket):
    """The server answers ``ERR``, so this stays a plain ``ResponseError``."""
    session.bucket.create_index(bucket, SingleFieldIndex("age", "int32"))
    with pytest.raises(ResponseError) as info:
        session.bucket.create_index(bucket, SingleFieldIndex("age", "int32"))
    assert not isinstance(info.value, NoSuchBucketError)


@pytest.mark.integration
def test_create_index_on_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.create_index(MISSING, SingleFieldIndex("age", "int32"))


@pytest.mark.integration
def test_create_index_on_a_removed_bucket(session, bucket):
    session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        session.bucket.create_index(bucket, SingleFieldIndex("age", "int32"))


@pytest.mark.integration
def test_list_indexes(session, bucket):
    names = session.bucket.list_indexes(bucket)
    assert "primary-index" in names
    assert all(isinstance(name, str) for name in names)


def described(session, bucket, index):
    """``create_index``, then its description.

    The members learn of a compound or vector index a moment later, and until
    then ``DESCRIBE`` fails, so this waits for the first good reply.
    """
    session.bucket.create_index(bucket, index)
    return wait_for(lambda: session.bucket.describe_index(bucket, index.name), "index")


def ready(session, bucket, index):
    """The description of ``index`` once its build is done."""

    def built():
        description = session.bucket.describe_index(bucket, index)
        return description if description["status"] == "READY" else None

    return wait_for(built, "index build")


def no_tasks(session, bucket, index):
    """``wait_for`` condition: every task of the index is done."""
    return lambda: not session.bucket.index_tasks(bucket, index)


@pytest.mark.integration
def test_describe_index(session, bucket):
    """Right after ``create_index`` the index waits for its build."""
    index = SingleFieldIndex("age", "int32", name="idx")
    description = described(session, bucket, index)
    assert description["index_type"] == "single_field"
    assert description["selector"] == "age"
    assert description["bson_type"] == "INT32"
    assert description["status"] in ("WAITING", "BUILDING")
    assert description["unique"] is False
    assert description["collation"] == {}
    assert description["statistics"] == {"cardinality": 0}


@pytest.mark.integration
def test_describe_a_compound_index(session, bucket):
    index = CompoundIndex(
        [CompoundIndexField("status", "string"), CompoundIndexField("age", "int32")],
        name="idx_status_age",
    )
    description = described(session, bucket, index)
    assert description["index_type"] == "compound"
    assert [f["selector"] for f in description["fields"]] == ["status", "age"]
    assert description["fields"][1]["bson_type"] == "INT32"
    assert description["unique"] is False


@pytest.mark.integration
def test_describe_a_vector_index(session, bucket):
    description = described(session, bucket, VectorIndex("v", 3, "cosine", name="idx"))
    assert description["index_type"] == "vector"
    assert description["selector"] == "v"
    assert description["dimensions"] == 3
    assert description["distance"] == "COSINE"
    assert "unique" not in description


@pytest.mark.integration
def test_describe_index_with_a_collation(session, bucket):
    index = SingleFieldIndex(
        "name", "string", name="idx", collation=Collation("tr", strength=2)
    )
    collation = described(session, bucket, index)["collation"]
    assert collation["locale"] == "tr"
    assert collation["strength"] == 2
    assert isinstance(collation["case_level"], bool)


@pytest.mark.integration
def test_describe_a_missing_index(session, bucket):
    with pytest.raises(NoSuchIndexError):
        session.bucket.describe_index(bucket, "no-such-index")


@pytest.mark.integration
def test_drop_index(session, bucket):
    """The index is ``DROPPED`` at once, and a task removes it for good.

    ``DROP`` refuses an index that still lists a task, and a finished task
    stays listed until a sweeper pass, so ``READY`` alone is not enough.
    """
    session.bucket.create_index(bucket, SingleFieldIndex("age", "int32", name="idx"))
    ready(session, bucket, "idx")
    wait_for(no_tasks(session, bucket, "idx"), "index tasks")
    assert session.bucket.drop_index(bucket, "idx") is True
    assert session.bucket.describe_index(bucket, "idx")["status"] == "DROPPED"

    def gone():
        try:
            session.bucket.describe_index(bucket, "idx")
        except NoSuchIndexError:
            return True
        return False

    wait_for(gone, "index removal")
    assert "idx" not in session.bucket.list_indexes(bucket)


@pytest.mark.integration
def test_drop_index_during_the_build(session, bucket):
    session.bucket.create_index(bucket, SingleFieldIndex("age", "int32", name="idx"))
    assert session.bucket.index_tasks(bucket, "idx")
    with pytest.raises(ResponseError, match="active tasks"):
        session.bucket.drop_index(bucket, "idx")


@pytest.mark.integration
def test_drop_the_primary_index(session, bucket):
    with pytest.raises(ResponseError) as info:
        session.bucket.drop_index(bucket, "primary-index")
    assert not isinstance(info.value, NoSuchIndexError)


@pytest.mark.integration
def test_drop_a_missing_index(session, bucket):
    with pytest.raises(NoSuchIndexError):
        session.bucket.drop_index(bucket, "no-such-index")


@pytest.mark.integration
def test_index_tasks(session, bucket):
    """``create_index`` queues the tasks of the build, and the dict is empty
    once the index is ``READY`` and the sweeper has cleared the tasks."""
    session.bucket.create_index(bucket, SingleFieldIndex("age", "int32", name="idx"))
    tasks = session.bucket.index_tasks(bucket, "idx")
    assert tasks
    for task_id, task in tasks.items():
        assert isinstance(task_id, str)
        assert task["kind"] in ("BUILD", "BOUNDARY")
        assert task["status"] in ("WAITING", "RUNNING", "COMPLETED")
        assert task["error"] == ""
        if task["kind"] == "BUILD":
            assert {"cursor", "lower", "upper"} <= task.keys()
    ready(session, bucket, "idx")
    wait_for(no_tasks(session, bucket, "idx"), "index tasks")
    assert session.bucket.index_tasks(bucket, "primary-index") == {}


@pytest.mark.integration
def test_analyze_index(session, bucket):
    """Only a ``READY`` index can be analyzed.

    ``ANALYZE`` reads the task list, and the sweeper clears the finished build
    tasks from it right after ``READY``. The two conflict, so the test waits
    for the list to be empty first.
    """
    session.bucket.create_index(bucket, SingleFieldIndex("age", "int32", name="idx"))
    ready(session, bucket, "idx")
    wait_for(no_tasks(session, bucket, "idx"), "index tasks")
    assert session.bucket.analyze_index(bucket, "idx") is True
    wait_for(no_tasks(session, bucket, "idx"), "analyze")


@pytest.mark.integration
def test_analyze_index_before_it_is_ready(session, bucket):
    session.bucket.create_index(bucket, SingleFieldIndex("age", "int32", name="idx"))
    with pytest.raises(ResponseError) as info:
        session.bucket.analyze_index(bucket, "idx")
    assert not isinstance(info.value, NoSuchIndexError)


@pytest.mark.integration
def test_analyze_a_missing_index(session, bucket):
    with pytest.raises(NoSuchIndexError):
        session.bucket.analyze_index(bucket, "no-such-index")


@pytest.mark.integration
def test_list(session, bucket):
    names = session.bucket.list()
    assert bucket in names
    assert all(isinstance(name, str) for name in names)


@pytest.mark.integration
def test_list_still_holds_a_removed_bucket(session, bucket):
    session.bucket.remove(bucket)
    assert bucket in session.bucket.list()


@pytest.mark.integration
def test_purge_takes_the_bucket_out_of_the_list(session, bucket):
    session.bucket.remove(bucket)
    session.bucket.purge(bucket)
    assert bucket not in session.bucket.list()


@pytest.mark.integration
def test_locate(session, bucket):
    location = session.bucket.locate(bucket)
    assert isinstance(location, BucketLocation)
    assert location.shards != []
    shard = location.shards[0]
    assert isinstance(shard, ShardLocation)
    assert isinstance(shard.shard_id, int)
    assert isinstance(shard.standbys, tuple)
    # The primary is a member id. The address is in the member table.
    assert ":" in location.members[shard.primary][0]


@pytest.mark.integration
def test_locate_unpacks(session, bucket):
    shards, members = session.bucket.locate(bucket)
    assert isinstance(members, dict)
    for shard_id, primary, standbys in shards:
        assert isinstance(shard_id, int)
        assert isinstance(primary, str)
        assert isinstance(standbys, tuple)


@pytest.mark.integration
def test_locate_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.locate(MISSING)


@pytest.mark.integration
def test_remove(session, bucket):
    assert session.bucket.remove(bucket) is True


@pytest.mark.integration
def test_remove_twice(session, bucket):
    session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        session.bucket.remove(bucket)


@pytest.mark.integration
def test_remove_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.remove(MISSING)


@pytest.mark.integration
def test_purge(session, bucket):
    session.bucket.remove(bucket)
    assert session.bucket.purge(bucket) is True


@pytest.mark.integration
def test_purge_without_remove(session, bucket):
    with pytest.raises(ResponseError) as info:
        session.bucket.purge(bucket)
    assert not isinstance(info.value, NoSuchBucketError)


def test_insert_needs_a_document():
    with pytest.raises(ValueError):
        _insert_args("users", (), "bson")


def test_a_document_is_encoded_by_input_type():
    """The two formats of the same codec, checked without a server."""
    document = {"n": 1}
    assert _insert_args("users", [document], "bson") == [
        "BUCKET.INSERT",
        "users",
        "DOCS",
        encode(document),
    ]
    assert _insert_args("users", [document], "json")[3] == '{"n": 1}'


def test_an_object_id_goes_out_as_extended_json():
    """``_id`` keeps its type in JSON, as ``{"$oid": ...}``."""
    oid = ObjectId()
    args = _insert_args("users", [{"_id": oid}], "json")
    assert args[3] == f'{{"_id": {{"$oid": "{oid}"}}}}'


@pytest.mark.integration
def test_insert(session, bucket):
    ids = session.bucket.insert(bucket, {"name": "Alice", "age": 30})
    assert len(ids) == 1
    assert isinstance(ids[0], ObjectId)


@pytest.mark.integration
def test_insert_many(session, bucket):
    ids = session.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    assert len(ids) == 2
    assert ids[0] != ids[1]


@pytest.mark.integration
def test_insert_with_an_id(session, bucket):
    """A document that carries its own ``_id`` gets that id back."""
    oid = ObjectId()
    assert session.bucket.insert(bucket, {"_id": oid, "name": "Alice"}) == [oid]


@pytest.mark.integration
def test_insert_a_duplicate_id(session, bucket):
    oid = ObjectId()
    session.bucket.insert(bucket, {"_id": oid})
    with pytest.raises(DuplicateKeyError):
        session.bucket.insert(bucket, {"_id": oid})


@pytest.mark.integration
def test_insert_with_a_bad_id_type(session, bucket):
    with pytest.raises(ResponseError) as info:
        session.bucket.insert(bucket, {"_id": "abc"})
    assert not isinstance(info.value, DuplicateKeyError)


@pytest.mark.integration
def test_insert_with_json_and_hex(kr, bucket):
    """The other pair of session attributes, on a session of its own."""
    config = KronotopConfig(input_type="json", object_id_format="hex")
    with kr.session(config=config) as s:
        ids = s.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    assert all(isinstance(oid, ObjectId) for oid in ids)


@pytest.mark.integration
def test_insert_in_a_transaction(session, bucket):
    """The ids come back before the commit."""
    with session.transaction():
        ids = session.bucket.insert(bucket, {"name": "Alice"})
    assert len(ids) == 1


@pytest.mark.integration
def test_insert_into_a_removed_bucket(session, bucket):
    session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        session.bucket.insert(bucket, {"name": "Alice"})


@pytest.mark.integration
def test_insert_into_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.insert(MISSING, {"name": "Alice"})


def test_a_query_is_sent_as_json():
    """The query is JSON whatever ``input_type`` says. See ``_query_args``."""
    args = _query_args(
        "BUCKET.QUERY",
        "users",
        {"age": {"$gt": 20}},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert args == ["BUCKET.QUERY", "users", '{"age": {"$gt": 20}}']


def test_a_query_keeps_object_ids():
    """Extended JSON, so an ``_id`` in a query stays an ObjectId."""
    oid = ObjectId()
    args = _query_args(
        "BUCKET.QUERY",
        "users",
        {"_id": oid},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert args[2] == f'{{"_id": {{"$oid": "{oid}"}}}}'


def test_a_query_with_every_option():
    args = _query_args(
        "BUCKET.QUERY",
        "users",
        {},
        None,
        ("age", "desc"),
        ("score", "asc"),
        {"name": 1},
        10,
        5,
        Collation("tr"),
    )
    assert args == [
        "BUCKET.QUERY",
        "users",
        "{}",
        "SORTBY",
        "age",
        "DESC",
        "RESULTSORT",
        "score",
        "ASC",
        "PROJECTION",
        '{"name": 1}',
        "BATCH",
        10,
        "LIMIT",
        5,
        "COLLATION",
        '{"locale": "tr"}',
    ]


def test_a_query_with_close():
    """``CLOSE`` is a flag, and it goes last, as in the server syntax."""
    args = _query_args(
        "BUCKET.QUERY",
        "users",
        {},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "prod",
        close=True,
    )
    assert args == ["BUCKET.QUERY", "users", "{}", "NAMESPACE", "prod", "CLOSE"]


@pytest.mark.integration
def test_query(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice", "age": 30})
    with session.bucket.query(bucket, {}) as cursor:
        assert isinstance(cursor, Cursor)
        assert cursor.bucket == bucket
        assert len(cursor.batch) == 1
        document = cursor.batch[0]
        assert document["name"] == "Alice"
        assert isinstance(document["_id"], ObjectId)


@pytest.mark.integration
def test_query_with_a_query(session, bucket):
    session.bucket.insert(bucket, {"age": 20}, {"age": 40})
    with session.bucket.query(bucket, {"age": {"$gt": 30}}) as cursor:
        assert [document["age"] for document in cursor.batch] == [40]


@pytest.mark.integration
def test_query_with_a_batch(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    with session.bucket.query(bucket, {}, batch=2) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
def test_query_with_a_limit(session, bucket):
    """The reply that reaches ``limit`` carries ``-1``, ``batches`` ends there
    and the ``with`` block has nothing to close."""
    session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    with session.bucket.query(bucket, {}, batch=2, limit=3) as cursor:
        cursor_id = cursor.cursor_id
        sizes = [len(batch) for batch in cursor.batches()]
    assert sizes == [2, 1]
    assert cursor.limit_reached
    assert cursor.cursor_id == -1
    assert cursor_id not in session.bucket.cursors()["query"]


@pytest.mark.integration
def test_query_with_close(session, bucket):
    """The reply carries the first batch and ``-1``, and no cursor stays in
    the session."""
    session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    with session.bucket.query(bucket, {}, batch=2, close=True) as cursor:
        sizes = [len(batch) for batch in cursor.batches()]
    assert sizes == [2]
    assert cursor.cursor_id == -1
    assert session.bucket.cursors()["query"] == {}


@pytest.mark.integration
def test_limit_reached_on_the_first_reply(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    with session.bucket.query(bucket, {}, limit=1) as cursor:
        assert cursor.limit_reached
        assert cursor.cursor_id == -1


@pytest.mark.integration
def test_a_limit_that_is_not_reached_keeps_the_cursor(session, bucket):
    """Fewer matches than ``limit`` never reach it, so the cursor stays."""
    session.bucket.insert(bucket, {"n": 1})
    cursor = session.bucket.query(bucket, {}, limit=5)
    cursor.advance()
    assert not cursor.limit_reached
    cursor.close()
    assert session.bucket.cursors()["query"] == {}


@pytest.mark.integration
def test_advance_after_the_limit_raises(session, bucket):
    session.bucket.insert(bucket, {"n": 1})
    with (
        session.bucket.query(bucket, {}, limit=1) as cursor,
        pytest.raises(ResponseError),
    ):
        cursor.advance()


@pytest.mark.integration
def test_query_with_a_projection(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice", "age": 30})
    with session.bucket.query(bucket, {}, projection={"_id": 0, "name": 1}) as cursor:
        assert cursor.batch == [{"name": "Alice"}]


@pytest.mark.integration
def test_query_with_a_sort(session, bucket):
    """``_id`` is the one field ``SORTBY`` always has an order for."""
    ids = session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    with session.bucket.query(bucket, {}, sort=("_id", "desc")) as cursor:
        assert [document["_id"] for document in cursor.batch] == sorted(
            ids, reverse=True
        )


@pytest.mark.integration
def test_query_with_a_result_sort(session, bucket):
    """``RESULTSORT`` sorts one batch in memory, so it needs no index."""
    session.bucket.insert(bucket, {"n": 2}, {"n": 1}, {"n": 3})
    with session.bucket.query(bucket, {}, result_sort=("n", "asc")) as cursor:
        assert [document["n"] for document in cursor.batch] == [1, 2, 3]


@pytest.mark.integration
def test_query_with_a_collation(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice"})
    collation = Collation("en", strength=2)
    with session.bucket.query(bucket, {"name": "alice"}, collation=collation) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
def test_query_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.query(MISSING, {})


@pytest.mark.integration
def test_query_with_a_json_reply_type(kr, bucket):
    """The other encoding of the same codec, so ``_id`` is an ObjectId again."""
    with kr.session(config=KronotopConfig(reply_type="json")) as s:
        s.bucket.insert(bucket, {"name": "Alice"})
        with s.bucket.query(bucket, {}) as cursor:
            assert cursor.batch[0]["name"] == "Alice"
            assert isinstance(cursor.batch[0]["_id"], ObjectId)


@pytest.mark.integration
def test_explain(session, bucket):
    """``{}`` needs no index, so the plan is a full scan with no filter."""
    result = session.bucket.explain(bucket, {})
    assert result["is_cached"] is False
    assert "query_collation" not in result
    plan = result["plan"]
    assert plan["planner_version"] == 1
    assert isinstance(plan["id"], int)
    assert plan["nodeType"] == "FullScan"
    assert plan["scanType"] == "FULL_SCAN"
    assert plan["index"] == "primary-index"
    assert plan["predicate"] == {"type": "ALWAYS_TRUE"}


@pytest.mark.integration
def test_explain_an_index_scan(session, bucket):
    """The docs render the operand one way and the server another, so only
    its presence is asserted."""
    plan = session.bucket.explain(bucket, {"_id": ObjectId()})["plan"]
    assert plan["nodeType"] == "IndexScan"
    assert plan["scanType"] == "INDEX_SCAN"
    assert plan["index"] == "primary-index"
    assert plan["selector"] == "_id"
    assert plan["operator"] == "EQ"
    assert "operand" in plan


@pytest.mark.integration
def test_explain_a_range_scan(session, bucket):
    """The bound flags are ``bool`` on either protocol."""
    low, high = sorted([ObjectId(), ObjectId()])
    query = {"_id": {"$gte": low, "$lte": high}}
    plan = session.bucket.explain(bucket, query)["plan"]
    assert plan["nodeType"] == "RangeScan"
    assert plan["scanType"] == "RANGE_SCAN"
    assert plan["selector"] == "_id"
    assert plan["includeLower"] is True
    assert plan["includeUpper"] is True


@pytest.mark.integration
def test_explain_a_residual_predicate(session, bucket):
    """A field with no index is checked on each document of a full scan."""
    plan = session.bucket.explain(bucket, {"name": "Alice"})["plan"]
    assert plan["nodeType"] == "FullScan"
    predicate = plan["predicate"]
    assert predicate["type"] == "PREDICATE"
    assert predicate["selector"] == "name"
    assert predicate["operator"] == "EQ"


@pytest.mark.integration
def test_explain_with_a_secondary_index(session, bucket):
    """The indexed field is scanned, the other one is filtered after it.

    The members learn of a ready index a moment later, so the plan is
    polled until it uses the index.
    """
    session.bucket.create_index(bucket, SingleFieldIndex("age", "int32", name="idx"))
    ready(session, bucket, "idx")
    query = {"$and": [{"age": 25}, {"name": "Alice"}]}

    def indexed():
        plan = session.bucket.explain(bucket, query)["plan"]
        return plan if plan["nodeType"] == "IndexScan" else None

    plan = wait_for(indexed, "index scan")
    assert plan["selector"] == "age"
    residual = plan["next"]
    assert residual["nodeType"] == "TransformWithResidualPredicate"
    assert residual["operation"] == "FILTER"
    predicate = residual["predicate"]
    assert predicate["type"] == "AND"
    assert [child["selector"] for child in predicate["children"]] == ["name"]


@pytest.mark.integration
def test_explain_a_cached_plan(session, bucket):
    """A query fills the cache, and the same shape then reads from it."""
    query = {"status": "active"}
    with session.bucket.query(bucket, query):
        pass
    assert session.bucket.explain(bucket, query)["is_cached"] is True


@pytest.mark.integration
def test_explain_with_a_collation(session, bucket):
    """The reply carries the collation the planner used, set fields only."""
    result = session.bucket.explain(bucket, {}, collation=Collation("tr", strength=2))
    assert result["query_collation"] == {"locale": "tr", "strength": 2}


@pytest.mark.integration
def test_explain_with_a_sort(session, bucket):
    """A sort reaches the planner: the index that gives the order is walked
    end to end, so the bounds are ``None`` on either protocol."""
    plan = (session.bucket.explain(bucket, {}, sort=("_id", "desc"), batch=10))["plan"]
    assert plan["nodeType"] == "RangeScan"
    assert plan["selector"] == "_id"
    assert plan["lowerBound"] is None
    assert plan["upperBound"] is None


@pytest.mark.integration
def test_explain_with_a_limit(session, bucket):
    """The server accepts ``LIMIT`` here. The plan does not show it."""
    result = session.bucket.explain(bucket, {}, batch=2, limit=3)
    assert "plan" in result


@pytest.mark.integration
def test_explain_on_resp2(kronotop_url, bucket):
    """RESP2 sends a flat array and ``0`` for the flag, decoded the same way."""
    other = Kronotop.from_url(kronotop_url, protocol=2)
    try:
        with other.session() as resp2:
            result = resp2.bucket.explain(bucket, {})
            assert result["is_cached"] is False
            assert result["plan"]["nodeType"] == "FullScan"
            assert result["plan"]["predicate"] == {"type": "ALWAYS_TRUE"}
    finally:
        other.close()


@pytest.mark.integration
def test_explain_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.explain(MISSING, {})


def test_a_delete_takes_batch_and_collation_alone():
    """``delete`` reuses ``_query_args`` with the other options at ``None``."""
    args = _query_args(
        "BUCKET.DELETE",
        "users",
        {},
        None,
        None,
        None,
        None,
        10,
        None,
        Collation("tr"),
    )
    assert args == [
        "BUCKET.DELETE",
        "users",
        "{}",
        "BATCH",
        10,
        "COLLATION",
        '{"locale": "tr"}',
    ]


def test_an_update_takes_sort_batch_and_collation():
    """The update document sits between the query and the options."""
    args = _query_args(
        "BUCKET.UPDATE",
        "users",
        {"status": "pending"},
        {"$set": {"status": "active"}},
        ("age", "desc"),
        None,
        None,
        10,
        None,
        Collation("tr"),
    )
    assert args == [
        "BUCKET.UPDATE",
        "users",
        '{"status": "pending"}',
        '{"$set": {"status": "active"}}',
        "SORTBY",
        "age",
        "DESC",
        "BATCH",
        10,
        "COLLATION",
        '{"locale": "tr"}',
    ]


def test_an_update_document_keeps_object_ids():
    """Extended JSON, so an ObjectId in the update stays one."""
    oid = ObjectId()
    args = _query_args(
        "BUCKET.UPDATE",
        "users",
        {},
        {"$set": {"owner": oid}},
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert args[3] == f'{{"$set": {{"owner": {{"$oid": "{oid}"}}}}}}'


def test_parse_bucket_delete_on_a_map():
    """The RESP3 shape: ``cursor_id`` and ``object_ids``, keys as bytes."""
    oid = ObjectId()
    batch = parse_bucket_delete({b"cursor_id": 7, b"object_ids": [str(oid).encode()]})
    assert batch == ObjectIdBatch(7, [oid])


def test_parse_bucket_delete_on_an_array():
    """The RESP2 shape: the id and the payload, in that order."""
    oid = ObjectId()
    assert parse_bucket_delete([7, [str(oid).encode()]]) == ObjectIdBatch(7, [oid])


def test_parse_bucket_delete_with_bytes_format():
    oid = ObjectId()
    batch = parse_bucket_delete([7, [oid.binary]], object_id_format="bytes")
    assert batch.object_ids == [oid]


def test_parse_bucket_advance_picks_the_parser():
    """The shared ``BUCKET.ADVANCE`` slot branches on the ``operation`` option."""
    assert isinstance(parse_bucket_advance([7, []]), QueryBatch)
    assert isinstance(parse_bucket_advance([7, []], operation="update"), ObjectIdBatch)
    assert isinstance(parse_bucket_advance([7, []], operation="delete"), ObjectIdBatch)


def test_limit_reached_follows_the_cursor_id():
    """A cursor id of ``-1`` is the server's way to say the cursor is gone,
    so ``close`` has nothing to send."""

    class Group:
        def __init__(self):
            self.closed = []

        def close(self, cursor_id, operation):
            self.closed.append(cursor_id)

    group = Group()
    gone = Cursor(group, "users", -1, [], "query")
    assert gone.limit_reached
    gone.close()
    assert group.closed == []
    open_ = Cursor(group, "users", 1, [], "query")
    assert not open_.limit_reached
    open_.close()
    assert group.closed == [1]


@pytest.mark.integration
def test_delete(session, bucket):
    ids = session.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    with session.bucket.delete(bucket, {}) as cursor:
        assert isinstance(cursor, Cursor)
        assert cursor.bucket == bucket
        assert sorted(cursor.batch) == sorted(ids)
    with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch == []


@pytest.mark.integration
def test_delete_with_a_query(session, bucket):
    session.bucket.insert(bucket, {"age": 20}, {"age": 40})
    with session.bucket.delete(bucket, {"age": {"$gt": 30}}) as cursor:
        assert len(cursor.batch) == 1
    with session.bucket.query(bucket, {}) as cursor:
        assert [document["age"] for document in cursor.batch] == [20]


@pytest.mark.integration
def test_delete_with_a_batch(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    with session.bucket.delete(bucket, {}, batch=2) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
def test_delete_with_a_limit(session, bucket):
    session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    with session.bucket.delete(bucket, {}, batch=2, limit=3) as cursor:
        sizes = [len(batch) for batch in cursor.batches()]
    assert sizes == [2, 1]
    assert cursor.limit_reached
    with session.bucket.query(bucket, {}) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
def test_delete_with_close(session, bucket):
    session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    with session.bucket.delete(bucket, {}, batch=2, close=True) as cursor:
        sizes = [len(batch) for batch in cursor.batches()]
    assert sizes == [2]
    assert cursor.cursor_id == -1
    assert session.bucket.cursors()["delete"] == {}
    with session.bucket.query(bucket, {}) as cursor:
        assert len(cursor.batch) == 3


@pytest.mark.integration
def test_delete_with_a_collation(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice"})
    collation = Collation("en", strength=2)
    with session.bucket.delete(
        bucket, {"name": "alice"}, collation=collation
    ) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
def test_delete_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.delete(MISSING, {})


@pytest.mark.integration
def test_delete_advance(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    with session.bucket.delete(bucket, {}, batch=1) as cursor:
        first = cursor.batch
        second = cursor.advance()
        assert cursor.batch is second
        assert first != second


@pytest.mark.integration
def test_delete_batches(session, bucket):
    ids = session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    deleted = []
    with session.bucket.delete(bucket, {}, batch=2) as cursor:
        for batch in cursor.batches():
            if not batch:
                break
            deleted.extend(batch)
    assert sorted(deleted) == sorted(ids)


@pytest.mark.integration
def test_the_delete_cursor_id_holds_without_a_limit(session, bucket):
    session.bucket.insert(bucket, {"n": 1})
    with session.bucket.delete(bucket, {}) as cursor:
        cursor_id = cursor.cursor_id
        cursor.advance()
        assert cursor.cursor_id == cursor_id


@pytest.mark.integration
def test_delete_close(session, bucket):
    cursor = session.bucket.delete(bucket, {})
    cursor.close()
    with pytest.raises(ResponseError):
        session.bucket.advance(cursor.cursor_id, "delete")


@pytest.mark.integration
def test_delete_close_twice_on_the_cursor(session, bucket):
    cursor = session.bucket.delete(bucket, {})
    cursor.close()
    cursor.close()


@pytest.mark.integration
def test_delete_close_twice_on_the_group(session, bucket):
    cursor = session.bucket.delete(bucket, {})
    assert session.bucket.close(cursor.cursor_id, "delete") is True
    with pytest.raises(ResponseError):
        session.bucket.close(cursor.cursor_id, "delete")


@pytest.mark.integration
def test_a_delete_cursor_shows_up_in_cursors(session, bucket):
    with session.bucket.delete(bucket, {"n": 1}) as cursor:
        cursors = session.bucket.cursors()
        assert cursors["delete"][cursor.cursor_id] == {"n": 1}
    assert session.bucket.cursors()["delete"] == {}


@pytest.mark.integration
def test_delete_with_a_bytes_object_id_format(kr, bucket):
    config = KronotopConfig(object_id_format="bytes")
    with kr.session(config=config) as s:
        ids = s.bucket.insert(bucket, {"name": "Alice"})
        with s.bucket.delete(bucket, {}) as cursor:
            assert cursor.batch == ids


@pytest.mark.integration
def test_delete_in_a_transaction(session, bucket):
    ids = session.bucket.insert(bucket, {"name": "Alice"})
    with session.transaction(), session.bucket.delete(bucket, {}) as cursor:
        assert cursor.batch == ids
    with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch == []


@pytest.mark.integration
def test_delete_rolled_back(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice"})
    session.begin()
    with session.bucket.delete(bucket, {}) as cursor:
        assert len(cursor.batch) == 1
    session.rollback()
    with session.bucket.query(bucket, {}) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
def test_update(session, bucket):
    ids = session.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    with session.bucket.update(bucket, {}, {"$set": {"active": True}}) as cursor:
        assert isinstance(cursor, Cursor)
        assert cursor.bucket == bucket
        assert sorted(cursor.batch) == sorted(ids)
    with session.bucket.query(bucket, {}) as cursor:
        assert all(document["active"] for document in cursor.batch)


@pytest.mark.integration
def test_update_with_a_query(session, bucket):
    session.bucket.insert(bucket, {"age": 20}, {"age": 40})
    with session.bucket.update(
        bucket, {"age": {"$gt": 30}}, {"$set": {"senior": True}}
    ) as cursor:
        assert len(cursor.batch) == 1
    with session.bucket.query(bucket, {"senior": True}) as cursor:
        assert [document["age"] for document in cursor.batch] == [40]


@pytest.mark.integration
def test_update_with_unset(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice", "temp": 1})
    with session.bucket.update(bucket, {}, {"$unset": ["temp"]}) as cursor:
        assert len(cursor.batch) == 1
    with session.bucket.query(bucket, {}) as cursor:
        assert "temp" not in cursor.batch[0]


@pytest.mark.integration
def test_update_with_unset_document_form(session, bucket):
    """``$unset`` also takes ``{"field": 1}``, the other form the server reads."""
    session.bucket.insert(bucket, {"name": "Alice", "temp": 1})
    with session.bucket.update(bucket, {}, {"$unset": {"temp": 1}}) as cursor:
        assert len(cursor.batch) == 1
    with session.bucket.query(bucket, {}) as cursor:
        assert "temp" not in cursor.batch[0]


@pytest.mark.integration
def test_update_with_set_and_unset(session, bucket):
    session.bucket.insert(bucket, {"version": 1, "old": True})
    with session.bucket.update(
        bucket, {}, {"$set": {"version": 2}, "$unset": ["old"]}
    ) as cursor:
        assert len(cursor.batch) == 1
    with session.bucket.query(bucket, {}) as cursor:
        document = cursor.batch[0]
        assert document["version"] == 2
        assert "old" not in document


@pytest.mark.integration
def test_update_leaves_an_unchanged_document_out(session, bucket):
    """A matched document the update leaves as it was is not in the batch."""
    session.bucket.insert(bucket, {"status": "active"})
    with session.bucket.update(bucket, {}, {"$set": {"status": "active"}}) as cursor:
        assert cursor.batch == []


@pytest.mark.integration
def test_update_with_array_filters(session, bucket):
    session.bucket.insert(bucket, {"grades": [50, 80]})
    update = {
        "$set": {"grades.$[elem]": 100},
        "array_filters": [{"elem": {"$gte": 60}}],
    }
    with session.bucket.update(bucket, {}, update) as cursor:
        assert len(cursor.batch) == 1
    with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch[0]["grades"] == [50, 100]


@pytest.mark.integration
def test_update_with_upsert(session, bucket):
    update = {"$set": {"name": "alice", "active": True}, "upsert": True}
    with session.bucket.update(bucket, {"name": "alice"}, update):
        pass
    with session.bucket.query(bucket, {"name": "alice"}) as cursor:
        assert cursor.batch[0]["active"] is True


@pytest.mark.integration
def test_update_cannot_change_the_id(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice"})
    with pytest.raises(ResponseError):
        session.bucket.update(bucket, {}, {"$set": {"_id": ObjectId()}})


@pytest.mark.integration
def test_update_with_an_empty_update(session, bucket):
    with pytest.raises(ResponseError):
        session.bucket.update(bucket, {}, {})


@pytest.mark.integration
def test_update_with_a_sort(session, bucket):
    """``_id`` is the one field ``SORTBY`` always has an order for."""
    ids = session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, sort=("_id", "desc"), batch=1
    ) as cursor:
        assert cursor.batch == [max(ids)]


@pytest.mark.integration
def test_update_with_a_batch(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    with session.bucket.update(bucket, {}, {"$set": {"seen": True}}, batch=2) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
def test_update_with_a_limit(session, bucket):
    session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, batch=2, limit=3
    ) as cursor:
        sizes = [len(batch) for batch in cursor.batches()]
    assert sizes == [2, 1]
    assert cursor.limit_reached
    with session.bucket.query(bucket, {"seen": True}) as cursor:
        assert len(cursor.batch) == 3


@pytest.mark.integration
def test_update_with_close(session, bucket):
    session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, batch=2, close=True
    ) as cursor:
        sizes = [len(batch) for batch in cursor.batches()]
    assert sizes == [2]
    assert cursor.cursor_id == -1
    assert session.bucket.cursors()["update"] == {}
    with session.bucket.query(bucket, {"seen": True}) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
def test_update_with_a_collation(session, bucket):
    session.bucket.insert(bucket, {"name": "Alice"})
    collation = Collation("en", strength=2)
    with session.bucket.update(
        bucket, {"name": "alice"}, {"$set": {"verified": True}}, collation=collation
    ) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
def test_update_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.update(MISSING, {}, {"$set": {"seen": True}})


@pytest.mark.integration
def test_update_a_removed_bucket(session, bucket):
    session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        session.bucket.update(bucket, {}, {"$set": {"seen": True}})


@pytest.mark.integration
def test_update_advance(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    with session.bucket.update(bucket, {}, {"$set": {"seen": True}}, batch=1) as cursor:
        first = cursor.batch
        second = cursor.advance()
        assert cursor.batch is second
        assert first != second


@pytest.mark.integration
def test_update_batches(session, bucket):
    ids = session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    updated = []
    with session.bucket.update(bucket, {}, {"$set": {"seen": True}}, batch=2) as cursor:
        for batch in cursor.batches():
            if not batch:
                break
            updated.extend(batch)
    assert sorted(updated) == sorted(ids)


@pytest.mark.integration
def test_the_update_cursor_id_holds_without_a_limit(session, bucket):
    session.bucket.insert(bucket, {"n": 1})
    with session.bucket.update(bucket, {}, {"$set": {"seen": True}}) as cursor:
        cursor_id = cursor.cursor_id
        cursor.advance()
        assert cursor.cursor_id == cursor_id


@pytest.mark.integration
def test_update_close(session, bucket):
    cursor = session.bucket.update(bucket, {}, {"$set": {"seen": True}})
    cursor.close()
    with pytest.raises(ResponseError):
        session.bucket.advance(cursor.cursor_id, "update")


@pytest.mark.integration
def test_update_close_twice_on_the_cursor(session, bucket):
    cursor = session.bucket.update(bucket, {}, {"$set": {"seen": True}})
    cursor.close()
    cursor.close()


@pytest.mark.integration
def test_update_close_twice_on_the_group(session, bucket):
    cursor = session.bucket.update(bucket, {}, {"$set": {"seen": True}})
    assert session.bucket.close(cursor.cursor_id, "update") is True
    with pytest.raises(ResponseError):
        session.bucket.close(cursor.cursor_id, "update")


@pytest.mark.integration
def test_an_update_cursor_shows_up_in_cursors(session, bucket):
    with session.bucket.update(bucket, {"n": 1}, {"$set": {"seen": True}}) as cursor:
        cursors = session.bucket.cursors()
        assert cursors["update"][cursor.cursor_id] == {"n": 1}
    assert session.bucket.cursors()["update"] == {}


@pytest.mark.integration
def test_update_with_a_bytes_object_id_format(kr, bucket):
    config = KronotopConfig(object_id_format="bytes")
    with kr.session(config=config) as s:
        ids = s.bucket.insert(bucket, {"name": "Alice"})
        with s.bucket.update(bucket, {}, {"$set": {"seen": True}}) as cursor:
            assert cursor.batch == ids


@pytest.mark.integration
def test_update_in_a_transaction(session, bucket):
    ids = session.bucket.insert(bucket, {"status": "pending"})
    with (
        session.transaction(),
        session.bucket.update(bucket, {}, {"$set": {"status": "active"}}) as cursor,
    ):
        assert cursor.batch == ids
    with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch[0]["status"] == "active"


@pytest.mark.integration
def test_update_rolled_back(session, bucket):
    session.bucket.insert(bucket, {"status": "pending"})
    session.begin()
    with session.bucket.update(bucket, {}, {"$set": {"status": "active"}}) as cursor:
        assert len(cursor.batch) == 1
    session.rollback()
    with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch[0]["status"] == "pending"


@pytest.mark.integration
def test_advance(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    with session.bucket.query(bucket, {}, batch=1) as cursor:
        first = cursor.batch
        second = cursor.advance()
        assert cursor.batch is second
        assert first != second


@pytest.mark.integration
def test_batches(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    with session.bucket.query(bucket, {}, batch=2) as cursor:
        batches = list(itertools.islice(cursor.batches(), 3))
    assert [len(batch) for batch in batches] == [2, 1, 0]


@pytest.mark.integration
def test_the_cursor_id_holds_without_a_limit(session, bucket):
    """Every reply echoes the same id until ``limit`` is reached, which is
    why ``advance`` needs no id."""
    session.bucket.insert(bucket, {"n": 1})
    with session.bucket.query(bucket, {}) as cursor:
        cursor_id = cursor.cursor_id
        cursor.advance()
        assert cursor.cursor_id == cursor_id


@pytest.mark.integration
def test_group_advance_returns_the_record(session, bucket):
    session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    with session.bucket.query(bucket, {}, batch=1) as cursor:
        batch = session.bucket.advance(cursor.cursor_id)
        assert isinstance(batch, QueryBatch)
        assert batch.cursor_id == cursor.cursor_id
        assert len(batch.documents) == 1


@pytest.mark.integration
def test_close(session, bucket):
    cursor = session.bucket.query(bucket, {})
    cursor.close()
    with pytest.raises(ResponseError):
        session.bucket.advance(cursor.cursor_id)


@pytest.mark.integration
def test_close_twice_on_the_cursor(session, bucket):
    """The cursor holds a flag, so the second call sends nothing."""
    cursor = session.bucket.query(bucket, {})
    cursor.close()
    cursor.close()


@pytest.mark.integration
def test_cursor_close_failure_does_not_mask_the_original_error(
    kr, bucket, timeout_once
):
    """``BUCKET.CLOSE`` runs on a dead socket, so ``_acquire`` raises.

    A session of its own: breaking the ``session`` fixture's connection would
    fail the ``buckets`` teardown.
    """
    with kr.session() as s:
        s.ping()
        conn = s._redis.connection
        with pytest.raises(redis.TimeoutError), s.bucket.query(bucket, {}) as cursor:
            conn.read_response = timeout_once(conn)
            cursor.advance()


@pytest.mark.integration
def test_close_twice_on_the_group(session, bucket):
    cursor = session.bucket.query(bucket, {})
    assert session.bucket.close(cursor.cursor_id) is True
    with pytest.raises(ResponseError):
        session.bucket.close(cursor.cursor_id)


@pytest.mark.integration
def test_cursors(session, bucket):
    with session.bucket.query(bucket, {"n": 1}) as cursor:
        cursors = session.bucket.cursors()
        assert cursors["query"][cursor.cursor_id] == {"n": 1}
        assert cursors["update"] == {}
        assert cursors["delete"] == {}
    assert session.bucket.cursors()["query"] == {}


@pytest.mark.integration
def test_cursors_for_one_operation(session, bucket):
    with session.bucket.query(bucket, {}) as cursor:
        cursors = session.bucket.cursors("query")
        assert list(cursors) == ["query"]
        assert cursor.cursor_id in cursors["query"]


def _host_of(client, session):
    """The member a session opened on, as a lookup key.

    LOCATE reports the address the server has, and the client can know that
    member under another name. The key is what the two share.
    """
    name = next(name for name, pool in client._pools.items() if pool is session._pool)
    return member_key(name)


@pytest.mark.integration
def test_session_for_opens_where_the_bucket_is(kr, session, bucket):
    location = session.bucket.locate(bucket)
    holders = {
        member_key(location.members[shard.primary][0]) for shard in location.shards
    }
    with kr.session_for(buckets={session.namespace.current(): [bucket]}) as opened:
        assert _host_of(kr, opened) in holders


@pytest.mark.integration
def test_session_for_keeps_the_placement(kr, session, bucket):
    """The second call opens from what the first one kept."""
    namespace = session.namespace.current()
    with kr.session_for(buckets={namespace: [bucket]}):
        pass
    kept = kr._placement.get(namespace, bucket)
    assert kept
    with kr.session_for(buckets={namespace: [bucket]}) as opened:
        assert _host_of(kr, opened) in {member_key(name) for name in kept}


@pytest.mark.integration
def test_session_for_serves_two_buckets_in_one_transaction(kr, session, buckets):
    """Both on one shard, so one member holds them and nothing is rejected."""
    first = buckets()
    shard_id = session.bucket.locate(first).shards[0].shard_id
    second = buckets(create=False)
    session.bucket.create(second, shards=[shard_id])

    namespace = session.namespace.current()
    with kr.session_for(buckets={namespace: [first, second]}) as opened:
        opened.namespace.use(namespace)
        with opened.transaction():
            opened.bucket.insert(first, {"name": "Alice"})
            opened.bucket.insert(second, {"name": "Bob"})


@pytest.mark.integration
def test_session_for_a_missing_bucket(kr, session):
    with pytest.raises(NoSuchBucketError):
        kr.session_for(buckets={session.namespace.current(): [MISSING]})


def test_a_vector_goes_out_as_packed_floats():
    args = _vector_args("items", "embedding", [0.1, 0.2, 0.3], *[None] * 6)
    assert args == [
        "BUCKET.VECTOR",
        "items",
        "embedding",
        struct.pack("<3f", 0.1, 0.2, 0.3),
    ]


def test_a_vector_search_with_every_option():
    args = _vector_args(
        "items",
        "embedding",
        [1.0],
        {"label": "beta"},
        {"label": 1},
        5,
        0.9,
        100,
        2.0,
    )
    assert args == [
        "BUCKET.VECTOR",
        "items",
        "embedding",
        struct.pack("<f", 1.0),
        "FILTER",
        '{"label": "beta"}',
        "PROJECTION",
        '{"label": 1}',
        "TOP",
        5,
        "THRESHOLD",
        0.9,
        "MAX-SCAN-CANDIDATES",
        100,
        "OVERQUERY",
        2.0,
    ]


def test_a_vector_search_needs_a_dimension():
    with pytest.raises(ValueError):
        _vector_args("items", "embedding", [], *[None] * 6)


def test_a_vector_search_rejects_top_zero():
    """The server reads ``TOP 0`` as its default, so the client says no first."""
    with pytest.raises(ValueError):
        _vector_args("items", "embedding", [0.1], None, None, 0, None, None, None)


@pytest.mark.parametrize(
    "response",
    [
        [{b"score": 0.5, b"entry": encode({"label": "beta"})}],
        [[struct.pack(">d", 0.5), encode({"label": "beta"})]],
    ],
    ids=["resp3-map", "resp2-pair"],
)
def test_parse_bucket_vector_takes_both_wire_shapes(response):
    assert parse_bucket_vector(response) == [VectorSearchResult(0.5, {"label": "beta"})]


def test_parse_bucket_vector_with_json_entries():
    oid = ObjectId()
    response = [[struct.pack(">d", 1.0), dumps({"_id": str(oid), "label": "beta"})]]
    assert parse_bucket_vector(response, reply_type="json") == [
        VectorSearchResult(1.0, {"_id": oid, "label": "beta"})
    ]


def test_parse_bucket_vector_on_no_match():
    assert parse_bucket_vector([]) == []


@pytest.mark.integration
def test_vector(session, vector_bucket):
    results = search(session, vector_bucket)
    assert [result.entry["label"] for result in results] == ["beta", "gamma", "alpha"]
    assert results[0].score == pytest.approx(1.0)
    assert isinstance(results[0].entry["_id"], ObjectId)


@pytest.mark.integration
def test_vector_with_top(session, vector_bucket):
    results = search(session, vector_bucket, expected=2, top=2)
    assert [result.entry["label"] for result in results] == ["beta", "gamma"]


@pytest.mark.integration
def test_vector_with_a_threshold(session, vector_bucket):
    """Alpha scores about 0.987, so the bar leaves it out."""
    results = search(session, vector_bucket, expected=2, threshold=0.99)
    assert [result.entry["label"] for result in results] == ["beta", "gamma"]


@pytest.mark.integration
def test_vector_with_a_filter(session, vector_bucket):
    results = search(session, vector_bucket, expected=1, filter_={"label": "alpha"})
    assert [result.entry["label"] for result in results] == ["alpha"]


@pytest.mark.integration
def test_vector_with_a_projection(session, vector_bucket):
    results = search(session, vector_bucket, projection={"_id": 0, "label": 1})
    assert results[0].entry == {"label": "beta"}


@pytest.mark.integration
def test_vector_with_scan_options(session, vector_bucket):
    """The server accepts both and the reply does not change on three documents."""
    results = search(
        session, vector_bucket, filter_={}, max_scan_candidates=100, overquery=2.0
    )
    assert len(results) == 3


@pytest.mark.integration
def test_vector_with_a_wrong_dimension(session, vector_bucket):
    """The server checks the dimension before it checks the index is ready."""
    with pytest.raises(ResponseError):
        session.bucket.vector(vector_bucket, "embedding", [0.1, 0.2])


@pytest.mark.integration
def test_vector_without_an_index(session, bucket):
    with pytest.raises(ResponseError):
        session.bucket.vector(bucket, "embedding", QUERY_VECTOR)


@pytest.mark.integration
def test_vector_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        session.bucket.vector(MISSING, "embedding", QUERY_VECTOR)


@pytest.mark.integration
def test_vector_with_a_json_reply_type(kr, vector_bucket):
    """The other encoding of the same codec, so ``_id`` is an ObjectId again."""
    with kr.session(config=KronotopConfig(reply_type="json")) as s:
        results = search(s, vector_bucket)
        assert results[0].entry["label"] == "beta"
        assert isinstance(results[0].entry["_id"], ObjectId)


def test_insert_sends_the_namespace_before_the_documents():
    """The server reads every argument after ``DOCS`` as a document."""
    args = _insert_args("users", [{"n": 1}], "json", "app")
    assert args[:5] == ["BUCKET.INSERT", "users", "NAMESPACE", "app", "DOCS"]


def test_a_query_sends_the_namespace_last():
    args = _query_args(
        "BUCKET.QUERY", "users", {}, None, None, None, None, None, 5, None, "app"
    )
    assert args[-4:] == ["LIMIT", 5, "NAMESPACE", "app"]


def test_a_vector_sends_the_namespace_last():
    args = _vector_args(
        "items", "embedding", [0.1], None, None, 2, None, None, None, "app"
    )
    assert args[-4:] == ["TOP", 2, "NAMESPACE", "app"]


@pytest.mark.integration
def test_the_commands_run_in_the_namespace_given(session, namespace, buckets):
    bucket = buckets(namespace=namespace)
    current = session.namespace.current()
    assert bucket in session.bucket.list(namespace=namespace)
    assert bucket not in session.bucket.list()
    assert session.bucket.locate(bucket, namespace=namespace).shards

    documents = [{"n": 1}, {"n": 2}, {"n": 3}]
    ids = session.bucket.insert(bucket, *documents, namespace=namespace)
    assert len(ids) == 3
    assert "plan" in session.bucket.explain(bucket, {}, namespace=namespace)
    with session.bucket.query(bucket, {}, batch=2, namespace=namespace) as cursor:
        assert len(cursor.batch) == 2
        # ADVANCE takes no namespace. The cursor stays where it was opened.
        assert len(cursor.advance()) == 1
    change = {"$set": {"n": 10}}
    with session.bucket.update(bucket, {"n": 1}, change, namespace=namespace) as cursor:
        assert cursor.batch == [ids[0]]
    with session.bucket.delete(bucket, {"n": 10}, namespace=namespace) as cursor:
        assert cursor.batch == [ids[0]]

    # The session did not move, and its own namespace does not hold the bucket.
    assert session.namespace.current() == current
    with pytest.raises(NoSuchBucketError):
        session.bucket.insert(bucket, {"n": 1})


@pytest.mark.integration
def test_the_index_commands_run_in_the_namespace_given(session, namespace, buckets):
    bucket = buckets(namespace=namespace)
    index = SingleFieldIndex("age", "int32", name="idx_age")
    assert session.bucket.create_index(bucket, index, namespace=namespace) is True
    assert "idx_age" in session.bucket.list_indexes(bucket, namespace=namespace)
    described = session.bucket.describe_index(bucket, "idx_age", namespace=namespace)
    assert described["selector"] == "age"
    tasks = session.bucket.index_tasks(bucket, "idx_age", namespace=namespace)
    assert isinstance(tasks, dict)
    with pytest.raises(NoSuchBucketError):
        session.bucket.list_indexes(bucket)


@pytest.mark.integration
def test_vector_in_the_namespace_given(session, namespace, buckets):
    bucket = buckets(namespace=namespace)
    session.bucket.create_index(bucket, VECTOR_INDEX, namespace=namespace)
    session.bucket.insert(bucket, *VECTOR_DOCUMENTS, namespace=namespace)
    results = search(session, bucket, namespace=namespace)
    assert [result.entry["label"] for result in results] == ["beta", "gamma", "alpha"]


@pytest.mark.integration
def test_a_namespace_that_is_not_there(session, namespaces):
    with pytest.raises(NoSuchNamespaceError):
        session.bucket.list(namespace=namespaces(create=False))


@pytest.mark.integration
def test_session_for_finds_a_bucket_in_another_namespace(
    kr, session, namespace, buckets
):
    bucket = buckets(namespace=namespace)
    location = session.bucket.locate(bucket, namespace=namespace)
    holders = {
        member_key(location.members[shard.primary][0]) for shard in location.shards
    }
    with kr.session_for(buckets={namespace: [bucket]}) as opened:
        assert _host_of(kr, opened) in holders
