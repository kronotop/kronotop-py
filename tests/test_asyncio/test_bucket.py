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

"""Async twin of ``tests/test_bucket.py``."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from contextlib import AbstractAsyncContextManager

import pytest
import redis
from bson import ObjectId
from redis.exceptions import ResponseError

from kronotop import (
    BucketAlreadyExistsError,
    BucketBeingRemovedError,
    DuplicateKeyError,
    KronotopConfig,
    NoSuchBucketError,
    NoSuchIndexError,
    NoSuchNamespaceError,
    VectorIndexNotReadyError,
)
from kronotop._members import member_key
from kronotop.asyncio import Kronotop
from kronotop.commands.bucket import (
    AsyncBucket,
    AsyncCursor,
    BucketLocation,
    Collation,
    CompoundIndex,
    CompoundIndexField,
    QueryBatch,
    ShardLocation,
    SingleFieldIndex,
    VectorIndex,
)

MISSING = "no-such-bucket"

VECTOR_INDEX = VectorIndex("embedding", 3, "cosine")
VECTOR_DOCUMENTS = [
    {"label": "alpha", "embedding": [0.1, 0.2, 0.3]},
    {"label": "beta", "embedding": [0.4, 0.5, 0.6]},
    {"label": "gamma", "embedding": [0.7, 0.8, 0.9]},
]
QUERY_VECTOR = [0.4, 0.5, 0.6]


@pytest.fixture
async def vector_bucket(session, bucket):
    """``bucket`` with the docs' vector index and its three documents."""
    await session.bucket.create_index(bucket, VECTOR_INDEX)
    await session.bucket.insert(bucket, *VECTOR_DOCUMENTS)
    return bucket


async def wait_for(condition, what, retry_on=ResponseError):
    """Await ``condition`` until it gives a true value, for up to 45 seconds.

    A background task drives every index change: the build, the drop, the
    analyze, and the members learn of a new index a moment later. Until then
    the server can answer with an error, so an error of ``retry_on`` counts as
    not yet. A build takes about 20 seconds: a boundary task, then the build.
    """
    deadline = time.monotonic() + 45
    while True:
        try:
            result = await condition()
        except retry_on:
            result = None
        if result:
            return result
        if time.monotonic() > deadline:
            raise AssertionError(f"{what} did not settle")
        await asyncio.sleep(0.1)


async def search(session, bucket, expected=3, **options):
    """``vector`` once the index is ready and holds ``expected`` results."""

    async def settled():
        results = await session.bucket.vector(
            bucket, "embedding", QUERY_VECTOR, **options
        )
        return results if len(results) >= expected else None

    return await wait_for(settled, "vector search", VectorIndexNotReadyError)


async def test_the_group_is_on_the_session():
    """``session.bucket`` is built with the session, so it needs no server."""
    session = Kronotop().session()
    assert isinstance(session.bucket, AsyncBucket)
    # Same object every time, so a caller can hold on to it.
    assert session.bucket is session.bucket


def test_the_group_is_not_on_the_client():
    """A bucket belongs to a namespace, and the namespace is session state."""
    assert not hasattr(Kronotop(), "bucket")


async def test_limit_reached_follows_the_cursor_id():
    """A cursor id of ``-1`` is the server's way to say the cursor is gone,
    so ``close`` has nothing to send."""

    class Group:
        def __init__(self):
            self.closed = []

        async def close(self, cursor_id, operation):
            self.closed.append(cursor_id)

    group = Group()
    gone = AsyncCursor(group, "users", -1, [], "query")
    assert gone.limit_reached
    await gone.close()
    assert group.closed == []
    open_ = AsyncCursor(group, "users", 1, [], "query")
    assert not open_.limit_reached
    await open_.close()
    assert group.closed == [1]


@pytest.mark.integration
async def test_create(session, buckets):
    name = await buckets(create=False)
    assert await session.bucket.create(name) is True


@pytest.mark.integration
async def test_create_on_a_taken_name(session, bucket):
    with pytest.raises(BucketAlreadyExistsError):
        await session.bucket.create(bucket)


@pytest.mark.integration
async def test_create_if_not_exists(session, bucket):
    assert await session.bucket.create(bucket, if_not_exists=True) is True


@pytest.mark.integration
async def test_create_on_a_removed_bucket(session, bucket):
    await session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        await session.bucket.create(bucket)


@pytest.mark.integration
async def test_create_on_a_shard(session, buckets):
    name = await buckets(create=False)
    assert await session.bucket.create(name, shards=[0]) is True
    location = await session.bucket.locate(name)
    assert [shard.shard_id for shard in location.shards] == [0]


@pytest.mark.integration
async def test_create_with_a_collation(session, buckets):
    name = await buckets(create=False)
    collation = Collation("tr", strength=2)
    assert await session.bucket.create(name, collation=collation) is True


@pytest.mark.integration
async def test_create_with_a_collation_dict(session, buckets):
    """The escape hatch for a field this client does not know yet."""
    name = await buckets(create=False)
    assert await session.bucket.create(name, collation={"locale": "tr"}) is True


@pytest.mark.integration
async def test_create_with_indexes(session, buckets):
    name = await buckets(create=False)
    index = SingleFieldIndex("age", "int32", name="idx_age")
    assert await session.bucket.create(name, indexes=[index]) is True
    assert "idx_age" in await session.bucket.list_indexes(name)


@pytest.mark.integration
async def test_create_index(session, bucket):
    index = SingleFieldIndex("age", "int32", name="idx_age")
    assert await session.bucket.create_index(bucket, index) is True
    assert "idx_age" in await session.bucket.list_indexes(bucket)


@pytest.mark.integration
async def test_create_index_with_a_compound_index(session, bucket):
    index = CompoundIndex(
        [CompoundIndexField("status", "string"), CompoundIndexField("age", "int32")],
        name="idx_status_age",
    )
    assert await session.bucket.create_index(bucket, index) is True
    assert "idx_status_age" in await session.bucket.list_indexes(bucket)


@pytest.mark.integration
async def test_create_with_a_unique_index(session, buckets):
    """On ``create`` the index is ready at once, so the write path is asserted."""
    name = await buckets(create=False)
    index = SingleFieldIndex("email", "string", unique=True)
    await session.bucket.create(name, indexes=[index])
    await session.bucket.insert(name, {"email": "a@kronotop.com"})
    with pytest.raises(DuplicateKeyError):
        await session.bucket.insert(name, {"email": "a@kronotop.com"})


@pytest.mark.integration
async def test_create_index_with_a_unique_index(session, bucket):
    """``describe_index`` says whether the index was recorded unique."""
    index = SingleFieldIndex("email", "string", unique=True, name="idx_email")
    assert await session.bucket.create_index(bucket, index) is True
    description = await session.bucket.describe_index(bucket, "idx_email")
    assert description["unique"] is True


@pytest.mark.integration
async def test_create_index_with_a_dict(session, bucket):
    """The escape hatch for a form this client does not know yet."""
    schema = {"age": {"bson_type": "int32", "name": "idx_age"}}
    assert await session.bucket.create_index(bucket, schema) is True
    assert "idx_age" in await session.bucket.list_indexes(bucket)


@pytest.mark.integration
async def test_create_index_twice_on_one_field(session, bucket):
    """The server answers ``ERR``, so this stays a plain ``ResponseError``."""
    await session.bucket.create_index(bucket, SingleFieldIndex("age", "int32"))
    with pytest.raises(ResponseError) as info:
        await session.bucket.create_index(bucket, SingleFieldIndex("age", "int32"))
    assert not isinstance(info.value, NoSuchBucketError)


@pytest.mark.integration
async def test_create_index_on_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.create_index(MISSING, SingleFieldIndex("age", "int32"))


@pytest.mark.integration
async def test_create_index_on_a_removed_bucket(session, bucket):
    await session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        await session.bucket.create_index(bucket, SingleFieldIndex("age", "int32"))


@pytest.mark.integration
async def test_list_indexes(session, bucket):
    names = await session.bucket.list_indexes(bucket)
    assert "primary-index" in names
    assert all(isinstance(name, str) for name in names)


async def described(session, bucket, index):
    """``create_index``, then its description.

    The members learn of a compound or vector index a moment later, and until
    then ``DESCRIBE`` fails, so this waits for the first good reply.
    """
    await session.bucket.create_index(bucket, index)

    async def known():
        return await session.bucket.describe_index(bucket, index.name)

    return await wait_for(known, "index")


async def ready(session, bucket, index):
    """The description of ``index`` once its build is done."""

    async def built():
        description = await session.bucket.describe_index(bucket, index)
        return description if description["status"] == "READY" else None

    return await wait_for(built, "index build")


async def no_tasks(session, bucket, index):
    """``wait_for`` condition: every task of the index is done."""

    async def done():
        return not await session.bucket.index_tasks(bucket, index)

    return done


@pytest.mark.integration
async def test_describe_index(session, bucket):
    """Right after ``create_index`` the index waits for its build."""
    index = SingleFieldIndex("age", "int32", name="idx")
    description = await described(session, bucket, index)
    assert description["index_type"] == "single_field"
    assert description["selector"] == "age"
    assert description["bson_type"] == "INT32"
    assert description["status"] in ("WAITING", "BUILDING")
    assert description["unique"] is False
    assert description["collation"] == {}
    assert description["statistics"] == {"cardinality": 0}


@pytest.mark.integration
async def test_describe_a_compound_index(session, bucket):
    index = CompoundIndex(
        [CompoundIndexField("status", "string"), CompoundIndexField("age", "int32")],
        name="idx_status_age",
    )
    description = await described(session, bucket, index)
    assert description["index_type"] == "compound"
    assert [f["selector"] for f in description["fields"]] == ["status", "age"]
    assert description["fields"][1]["bson_type"] == "INT32"
    assert description["unique"] is False


@pytest.mark.integration
async def test_describe_a_vector_index(session, bucket):
    index = VectorIndex("v", 3, "cosine", name="idx")
    description = await described(session, bucket, index)
    assert description["index_type"] == "vector"
    assert description["selector"] == "v"
    assert description["dimensions"] == 3
    assert description["distance"] == "COSINE"
    assert "unique" not in description


@pytest.mark.integration
async def test_describe_index_with_a_collation(session, bucket):
    index = SingleFieldIndex(
        "name", "string", name="idx", collation=Collation("tr", strength=2)
    )
    collation = (await described(session, bucket, index))["collation"]
    assert collation["locale"] == "tr"
    assert collation["strength"] == 2
    assert isinstance(collation["case_level"], bool)


@pytest.mark.integration
async def test_describe_a_missing_index(session, bucket):
    with pytest.raises(NoSuchIndexError):
        await session.bucket.describe_index(bucket, "no-such-index")


@pytest.mark.integration
async def test_drop_index(session, bucket):
    """The index is ``DROPPED`` at once, and a task removes it for good.

    ``DROP`` refuses an index that still lists a task, and a finished task
    stays listed until a sweeper pass, so ``READY`` alone is not enough.
    """
    index = SingleFieldIndex("age", "int32", name="idx")
    await session.bucket.create_index(bucket, index)
    await ready(session, bucket, "idx")
    await wait_for(await no_tasks(session, bucket, "idx"), "index tasks")
    assert await session.bucket.drop_index(bucket, "idx") is True
    description = await session.bucket.describe_index(bucket, "idx")
    assert description["status"] == "DROPPED"

    async def gone():
        try:
            await session.bucket.describe_index(bucket, "idx")
        except NoSuchIndexError:
            return True
        return False

    await wait_for(gone, "index removal")
    assert "idx" not in await session.bucket.list_indexes(bucket)


@pytest.mark.integration
async def test_drop_index_during_the_build(session, bucket):
    index = SingleFieldIndex("age", "int32", name="idx")
    await session.bucket.create_index(bucket, index)
    assert await session.bucket.index_tasks(bucket, "idx")
    with pytest.raises(ResponseError, match="active tasks"):
        await session.bucket.drop_index(bucket, "idx")


@pytest.mark.integration
async def test_drop_the_primary_index(session, bucket):
    with pytest.raises(ResponseError) as info:
        await session.bucket.drop_index(bucket, "primary-index")
    assert not isinstance(info.value, NoSuchIndexError)


@pytest.mark.integration
async def test_drop_a_missing_index(session, bucket):
    with pytest.raises(NoSuchIndexError):
        await session.bucket.drop_index(bucket, "no-such-index")


@pytest.mark.integration
async def test_index_tasks(session, bucket):
    """``create_index`` queues the tasks of the build, and the dict is empty
    once the index is ``READY`` and the sweeper has cleared the tasks."""
    index = SingleFieldIndex("age", "int32", name="idx")
    await session.bucket.create_index(bucket, index)
    tasks = await session.bucket.index_tasks(bucket, "idx")
    assert tasks
    for task_id, task in tasks.items():
        assert isinstance(task_id, str)
        assert task["kind"] in ("BUILD", "BOUNDARY")
        assert task["status"] in ("WAITING", "RUNNING", "COMPLETED")
        assert task["error"] == ""
        if task["kind"] == "BUILD":
            assert {"cursor", "lower", "upper"} <= task.keys()
    await ready(session, bucket, "idx")
    await wait_for(await no_tasks(session, bucket, "idx"), "index tasks")
    assert await session.bucket.index_tasks(bucket, "primary-index") == {}


@pytest.mark.integration
async def test_analyze_index(session, bucket):
    """Only a ``READY`` index can be analyzed.

    ``ANALYZE`` reads the task list, and the sweeper clears the finished build
    tasks from it right after ``READY``. The two conflict, so the test waits
    for the list to be empty first.
    """
    index = SingleFieldIndex("age", "int32", name="idx")
    await session.bucket.create_index(bucket, index)
    await ready(session, bucket, "idx")
    await wait_for(await no_tasks(session, bucket, "idx"), "index tasks")
    assert await session.bucket.analyze_index(bucket, "idx") is True
    await wait_for(await no_tasks(session, bucket, "idx"), "analyze")


@pytest.mark.integration
async def test_analyze_index_before_it_is_ready(session, bucket):
    index = SingleFieldIndex("age", "int32", name="idx")
    await session.bucket.create_index(bucket, index)
    with pytest.raises(ResponseError) as info:
        await session.bucket.analyze_index(bucket, "idx")
    assert not isinstance(info.value, NoSuchIndexError)


@pytest.mark.integration
async def test_analyze_a_missing_index(session, bucket):
    with pytest.raises(NoSuchIndexError):
        await session.bucket.analyze_index(bucket, "no-such-index")


@pytest.mark.integration
async def test_list(session, bucket):
    names = await session.bucket.list()
    assert bucket in names
    assert all(isinstance(name, str) for name in names)


@pytest.mark.integration
async def test_list_still_holds_a_removed_bucket(session, bucket):
    await session.bucket.remove(bucket)
    assert bucket in await session.bucket.list()


@pytest.mark.integration
async def test_purge_takes_the_bucket_out_of_the_list(session, bucket):
    await session.bucket.remove(bucket)
    await session.bucket.purge(bucket)
    assert bucket not in await session.bucket.list()


@pytest.mark.integration
async def test_locate(session, bucket):
    location = await session.bucket.locate(bucket)
    assert isinstance(location, BucketLocation)
    assert location.shards != []
    shard = location.shards[0]
    assert isinstance(shard, ShardLocation)
    assert isinstance(shard.shard_id, int)
    assert isinstance(shard.standbys, tuple)
    # The primary is a member id. The address is in the member table.
    assert ":" in location.members[shard.primary][0]


@pytest.mark.integration
async def test_locate_unpacks(session, bucket):
    shards, members = await session.bucket.locate(bucket)
    assert isinstance(members, dict)
    for shard_id, primary, standbys in shards:
        assert isinstance(shard_id, int)
        assert isinstance(primary, str)
        assert isinstance(standbys, tuple)


@pytest.mark.integration
async def test_locate_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.locate(MISSING)


@pytest.mark.integration
async def test_remove(session, bucket):
    assert await session.bucket.remove(bucket) is True


@pytest.mark.integration
async def test_remove_twice(session, bucket):
    await session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        await session.bucket.remove(bucket)


@pytest.mark.integration
async def test_remove_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.remove(MISSING)


@pytest.mark.integration
async def test_purge(session, bucket):
    await session.bucket.remove(bucket)
    assert await session.bucket.purge(bucket) is True


@pytest.mark.integration
async def test_purge_without_remove(session, bucket):
    with pytest.raises(ResponseError) as info:
        await session.bucket.purge(bucket)
    assert not isinstance(info.value, NoSuchBucketError)


@pytest.mark.integration
async def test_insert(session, bucket):
    ids = await session.bucket.insert(bucket, {"name": "Alice", "age": 30})
    assert len(ids) == 1
    assert isinstance(ids[0], ObjectId)


@pytest.mark.integration
async def test_insert_many(session, bucket):
    ids = await session.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    assert len(ids) == 2
    assert ids[0] != ids[1]


@pytest.mark.integration
async def test_insert_with_an_id(session, bucket):
    """A document that carries its own ``_id`` gets that id back."""
    oid = ObjectId()
    assert await session.bucket.insert(bucket, {"_id": oid, "name": "Alice"}) == [oid]


@pytest.mark.integration
async def test_insert_a_duplicate_id(session, bucket):
    oid = ObjectId()
    await session.bucket.insert(bucket, {"_id": oid})
    with pytest.raises(DuplicateKeyError):
        await session.bucket.insert(bucket, {"_id": oid})


@pytest.mark.integration
async def test_insert_with_a_bad_id_type(session, bucket):
    with pytest.raises(ResponseError) as info:
        await session.bucket.insert(bucket, {"_id": "abc"})
    assert not isinstance(info.value, DuplicateKeyError)


@pytest.mark.integration
async def test_insert_with_json_and_hex(kr, bucket):
    """The other pair of session attributes, on a session of its own."""
    config = KronotopConfig(input_type="json", object_id_format="hex")
    async with kr.session(config=config) as s:
        ids = await s.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    assert all(isinstance(oid, ObjectId) for oid in ids)


@pytest.mark.integration
async def test_insert_in_a_transaction(session, bucket):
    """The ids come back before the commit."""
    async with session.transaction():
        ids = await session.bucket.insert(bucket, {"name": "Alice"})
    assert len(ids) == 1


@pytest.mark.integration
async def test_insert_into_a_removed_bucket(session, bucket):
    await session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        await session.bucket.insert(bucket, {"name": "Alice"})


@pytest.mark.integration
async def test_insert_into_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.insert(MISSING, {"name": "Alice"})


@pytest.mark.integration
async def test_query(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice", "age": 30})
    async with session.bucket.query(bucket, {}) as cursor:
        assert isinstance(cursor, AsyncCursor)
        assert cursor.bucket == bucket
        assert len(cursor.batch) == 1
        document = cursor.batch[0]
        assert document["name"] == "Alice"
        assert isinstance(document["_id"], ObjectId)


@pytest.mark.integration
async def test_query_with_a_query(session, bucket):
    await session.bucket.insert(bucket, {"age": 20}, {"age": 40})
    async with session.bucket.query(bucket, {"age": {"$gt": 30}}) as cursor:
        assert [document["age"] for document in cursor.batch] == [40]


@pytest.mark.integration
async def test_query_with_a_batch(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    async with session.bucket.query(bucket, {}, batch=2) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
async def test_query_with_a_limit(session, bucket):
    await session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    sizes = []
    async with session.bucket.query(bucket, {}, batch=2, limit=3) as cursor:
        cursor_id = cursor.cursor_id
        async for batch in cursor.batches():
            sizes.append(len(batch))
    assert sizes == [2, 1]
    assert cursor.limit_reached
    assert cursor.cursor_id == -1
    assert cursor_id not in (await session.bucket.cursors())["query"]


@pytest.mark.integration
async def test_query_with_close(session, bucket):
    await session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    sizes = []
    async with session.bucket.query(bucket, {}, batch=2, close=True) as cursor:
        async for batch in cursor.batches():
            sizes.append(len(batch))
    assert sizes == [2]
    assert cursor.cursor_id == -1
    assert (await session.bucket.cursors())["query"] == {}


@pytest.mark.integration
async def test_limit_reached_on_the_first_reply(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    async with session.bucket.query(bucket, {}, limit=1) as cursor:
        assert cursor.limit_reached
        assert cursor.cursor_id == -1


@pytest.mark.integration
async def test_a_limit_that_is_not_reached_keeps_the_cursor(session, bucket):
    await session.bucket.insert(bucket, {"n": 1})
    cursor = await session.bucket.query(bucket, {}, limit=5)
    await cursor.advance()
    assert not cursor.limit_reached
    await cursor.close()
    assert (await session.bucket.cursors())["query"] == {}


@pytest.mark.integration
async def test_advance_after_the_limit_raises(session, bucket):
    await session.bucket.insert(bucket, {"n": 1})
    async with session.bucket.query(bucket, {}, limit=1) as cursor:
        with pytest.raises(ResponseError):
            await cursor.advance()


@pytest.mark.integration
async def test_query_with_a_projection(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice", "age": 30})
    projection = {"_id": 0, "name": 1}
    async with session.bucket.query(bucket, {}, projection=projection) as cursor:
        assert cursor.batch == [{"name": "Alice"}]


@pytest.mark.integration
async def test_query_with_a_sort(session, bucket):
    """``_id`` is the one field ``SORTBY`` always has an order for."""
    ids = await session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    async with session.bucket.query(bucket, {}, sort=("_id", "desc")) as cursor:
        assert [document["_id"] for document in cursor.batch] == sorted(
            ids, reverse=True
        )


@pytest.mark.integration
async def test_query_with_a_result_sort(session, bucket):
    """``RESULTSORT`` sorts one batch in memory, so it needs no index."""
    await session.bucket.insert(bucket, {"n": 2}, {"n": 1}, {"n": 3})
    async with session.bucket.query(bucket, {}, result_sort=("n", "asc")) as cursor:
        assert [document["n"] for document in cursor.batch] == [1, 2, 3]


@pytest.mark.integration
async def test_query_with_a_collation(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice"})
    collation = Collation("en", strength=2)
    async with session.bucket.query(
        bucket, {"name": "alice"}, collation=collation
    ) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
async def test_query_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.query(MISSING, {})


@pytest.mark.integration
async def test_query_with_a_json_reply_type(kr, bucket):
    """The other encoding of the same codec, so ``_id`` is an ObjectId again."""
    async with kr.session(config=KronotopConfig(reply_type="json")) as s:
        await s.bucket.insert(bucket, {"name": "Alice"})
        async with s.bucket.query(bucket, {}) as cursor:
            assert cursor.batch[0]["name"] == "Alice"
            assert isinstance(cursor.batch[0]["_id"], ObjectId)


@pytest.mark.integration
async def test_explain(session, bucket):
    """``{}`` needs no index, so the plan is a full scan with no filter."""
    result = await session.bucket.explain(bucket, {})
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
async def test_explain_an_index_scan(session, bucket):
    """The docs render the operand one way and the server another, so only
    its presence is asserted."""
    plan = (await session.bucket.explain(bucket, {"_id": ObjectId()}))["plan"]
    assert plan["nodeType"] == "IndexScan"
    assert plan["scanType"] == "INDEX_SCAN"
    assert plan["index"] == "primary-index"
    assert plan["selector"] == "_id"
    assert plan["operator"] == "EQ"
    assert "operand" in plan


@pytest.mark.integration
async def test_explain_a_range_scan(session, bucket):
    """The bound flags are ``bool`` on either protocol."""
    low, high = sorted([ObjectId(), ObjectId()])
    query = {"_id": {"$gte": low, "$lte": high}}
    plan = (await session.bucket.explain(bucket, query))["plan"]
    assert plan["nodeType"] == "RangeScan"
    assert plan["scanType"] == "RANGE_SCAN"
    assert plan["selector"] == "_id"
    assert plan["includeLower"] is True
    assert plan["includeUpper"] is True


@pytest.mark.integration
async def test_explain_a_residual_predicate(session, bucket):
    """A field with no index is checked on each document of a full scan."""
    plan = (await session.bucket.explain(bucket, {"name": "Alice"}))["plan"]
    assert plan["nodeType"] == "FullScan"
    predicate = plan["predicate"]
    assert predicate["type"] == "PREDICATE"
    assert predicate["selector"] == "name"
    assert predicate["operator"] == "EQ"


@pytest.mark.integration
async def test_explain_with_a_secondary_index(session, bucket):
    """The indexed field is scanned, the other one is filtered after it.

    The members learn of a ready index a moment later, so the plan is
    polled until it uses the index.
    """
    await session.bucket.create_index(
        bucket, SingleFieldIndex("age", "int32", name="idx")
    )
    await ready(session, bucket, "idx")
    query = {"$and": [{"age": 25}, {"name": "Alice"}]}

    async def indexed():
        plan = (await session.bucket.explain(bucket, query))["plan"]
        return plan if plan["nodeType"] == "IndexScan" else None

    plan = await wait_for(indexed, "index scan")
    assert plan["selector"] == "age"
    residual = plan["next"]
    assert residual["nodeType"] == "TransformWithResidualPredicate"
    assert residual["operation"] == "FILTER"
    predicate = residual["predicate"]
    assert predicate["type"] == "AND"
    assert [child["selector"] for child in predicate["children"]] == ["name"]


@pytest.mark.integration
async def test_explain_a_cached_plan(session, bucket):
    """A query fills the cache, and the same shape then reads from it."""
    query = {"status": "active"}
    async with session.bucket.query(bucket, query):
        pass
    assert (await session.bucket.explain(bucket, query))["is_cached"] is True


@pytest.mark.integration
async def test_explain_with_a_collation(session, bucket):
    """The reply carries the collation the planner used, set fields only."""
    result = await session.bucket.explain(
        bucket, {}, collation=Collation("tr", strength=2)
    )
    assert result["query_collation"] == {"locale": "tr", "strength": 2}


@pytest.mark.integration
async def test_explain_with_a_sort(session, bucket):
    """A sort reaches the planner: the index that gives the order is walked
    end to end, so the bounds are ``None`` on either protocol."""
    plan = (await session.bucket.explain(bucket, {}, sort=("_id", "desc"), batch=10))[
        "plan"
    ]
    assert plan["nodeType"] == "RangeScan"
    assert plan["selector"] == "_id"
    assert plan["lowerBound"] is None
    assert plan["upperBound"] is None


@pytest.mark.integration
async def test_explain_with_a_limit(session, bucket):
    """The server accepts ``LIMIT`` here. The plan does not show it."""
    result = await session.bucket.explain(bucket, {}, batch=2, limit=3)
    assert "plan" in result


@pytest.mark.integration
async def test_explain_on_resp2(kronotop_url, bucket):
    """RESP2 sends a flat array and ``0`` for the flag, decoded the same way."""
    other = Kronotop.from_url(kronotop_url, protocol=2)
    try:
        async with other.session() as resp2:
            result = await resp2.bucket.explain(bucket, {})
            assert result["is_cached"] is False
            assert result["plan"]["nodeType"] == "FullScan"
            assert result["plan"]["predicate"] == {"type": "ALWAYS_TRUE"}
    finally:
        await other.aclose()


@pytest.mark.integration
async def test_explain_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.explain(MISSING, {})


@pytest.mark.integration
async def test_delete(session, bucket):
    ids = await session.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    async with session.bucket.delete(bucket, {}) as cursor:
        assert isinstance(cursor, AsyncCursor)
        assert cursor.bucket == bucket
        assert sorted(cursor.batch) == sorted(ids)
    async with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch == []


@pytest.mark.integration
async def test_delete_with_a_query(session, bucket):
    await session.bucket.insert(bucket, {"age": 20}, {"age": 40})
    async with session.bucket.delete(bucket, {"age": {"$gt": 30}}) as cursor:
        assert len(cursor.batch) == 1
    async with session.bucket.query(bucket, {}) as cursor:
        assert [document["age"] for document in cursor.batch] == [20]


@pytest.mark.integration
async def test_delete_with_a_batch(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    async with session.bucket.delete(bucket, {}, batch=2) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
async def test_delete_with_a_limit(session, bucket):
    await session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    sizes = []
    async with session.bucket.delete(bucket, {}, batch=2, limit=3) as cursor:
        async for batch in cursor.batches():
            sizes.append(len(batch))
    assert sizes == [2, 1]
    assert cursor.limit_reached
    async with session.bucket.query(bucket, {}) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
async def test_delete_with_close(session, bucket):
    await session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    sizes = []
    async with session.bucket.delete(bucket, {}, batch=2, close=True) as cursor:
        async for batch in cursor.batches():
            sizes.append(len(batch))
    assert sizes == [2]
    assert cursor.cursor_id == -1
    assert (await session.bucket.cursors())["delete"] == {}
    async with session.bucket.query(bucket, {}) as cursor:
        assert len(cursor.batch) == 3


@pytest.mark.integration
async def test_delete_with_a_collation(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice"})
    collation = Collation("en", strength=2)
    async with session.bucket.delete(
        bucket, {"name": "alice"}, collation=collation
    ) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
async def test_delete_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.delete(MISSING, {})


@pytest.mark.integration
async def test_delete_advance(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    async with session.bucket.delete(bucket, {}, batch=1) as cursor:
        first = cursor.batch
        second = await cursor.advance()
        assert cursor.batch is second
        assert first != second


@pytest.mark.integration
async def test_delete_batches(session, bucket):
    ids = await session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    deleted = []
    async with session.bucket.delete(bucket, {}, batch=2) as cursor:
        async for batch in cursor.batches():
            if not batch:
                break
            deleted.extend(batch)
    assert sorted(deleted) == sorted(ids)


@pytest.mark.integration
async def test_delete_is_awaitable_and_a_context_manager(session, bucket):
    """Both forms work, so ``async with await`` is never needed."""
    opener = session.bucket.delete(bucket, {})
    assert isinstance(opener, Awaitable)
    assert isinstance(opener, AbstractAsyncContextManager)
    await (await opener).close()


@pytest.mark.integration
async def test_the_delete_cursor_id_holds_without_a_limit(session, bucket):
    await session.bucket.insert(bucket, {"n": 1})
    async with session.bucket.delete(bucket, {}) as cursor:
        cursor_id = cursor.cursor_id
        await cursor.advance()
        assert cursor.cursor_id == cursor_id


@pytest.mark.integration
async def test_delete_close(session, bucket):
    cursor = await session.bucket.delete(bucket, {})
    await cursor.close()
    with pytest.raises(ResponseError):
        await session.bucket.advance(cursor.cursor_id, "delete")


@pytest.mark.integration
async def test_delete_close_twice_on_the_cursor(session, bucket):
    cursor = await session.bucket.delete(bucket, {})
    await cursor.close()
    await cursor.close()


@pytest.mark.integration
async def test_delete_close_twice_on_the_group(session, bucket):
    cursor = await session.bucket.delete(bucket, {})
    assert await session.bucket.close(cursor.cursor_id, "delete") is True
    with pytest.raises(ResponseError):
        await session.bucket.close(cursor.cursor_id, "delete")


@pytest.mark.integration
async def test_a_delete_cursor_shows_up_in_cursors(session, bucket):
    async with session.bucket.delete(bucket, {"n": 1}) as cursor:
        cursors = await session.bucket.cursors()
        assert cursors["delete"][cursor.cursor_id] == {"n": 1}
    assert (await session.bucket.cursors())["delete"] == {}


@pytest.mark.integration
async def test_delete_with_a_bytes_object_id_format(kr, bucket):
    config = KronotopConfig(object_id_format="bytes")
    async with kr.session(config=config) as s:
        ids = await s.bucket.insert(bucket, {"name": "Alice"})
        async with s.bucket.delete(bucket, {}) as cursor:
            assert cursor.batch == ids


@pytest.mark.integration
async def test_delete_in_a_transaction(session, bucket):
    ids = await session.bucket.insert(bucket, {"name": "Alice"})
    async with session.transaction(), session.bucket.delete(bucket, {}) as cursor:
        assert cursor.batch == ids
    async with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch == []


@pytest.mark.integration
async def test_delete_rolled_back(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice"})
    await session.begin()
    async with session.bucket.delete(bucket, {}) as cursor:
        assert len(cursor.batch) == 1
    await session.rollback()
    async with session.bucket.query(bucket, {}) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
async def test_update(session, bucket):
    ids = await session.bucket.insert(bucket, {"name": "Alice"}, {"name": "Bob"})
    async with session.bucket.update(bucket, {}, {"$set": {"active": True}}) as cursor:
        assert isinstance(cursor, AsyncCursor)
        assert cursor.bucket == bucket
        assert sorted(cursor.batch) == sorted(ids)
    async with session.bucket.query(bucket, {}) as cursor:
        assert all(document["active"] for document in cursor.batch)


@pytest.mark.integration
async def test_update_with_a_query(session, bucket):
    await session.bucket.insert(bucket, {"age": 20}, {"age": 40})
    async with session.bucket.update(
        bucket, {"age": {"$gt": 30}}, {"$set": {"senior": True}}
    ) as cursor:
        assert len(cursor.batch) == 1
    async with session.bucket.query(bucket, {"senior": True}) as cursor:
        assert [document["age"] for document in cursor.batch] == [40]


@pytest.mark.integration
async def test_update_with_unset(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice", "temp": 1})
    async with session.bucket.update(bucket, {}, {"$unset": ["temp"]}) as cursor:
        assert len(cursor.batch) == 1
    async with session.bucket.query(bucket, {}) as cursor:
        assert "temp" not in cursor.batch[0]


@pytest.mark.integration
async def test_update_with_unset_document_form(session, bucket):
    """``$unset`` also takes ``{"field": 1}``, the other form the server reads."""
    await session.bucket.insert(bucket, {"name": "Alice", "temp": 1})
    async with session.bucket.update(bucket, {}, {"$unset": {"temp": 1}}) as cursor:
        assert len(cursor.batch) == 1
    async with session.bucket.query(bucket, {}) as cursor:
        assert "temp" not in cursor.batch[0]


@pytest.mark.integration
async def test_update_with_set_and_unset(session, bucket):
    await session.bucket.insert(bucket, {"version": 1, "old": True})
    async with session.bucket.update(
        bucket, {}, {"$set": {"version": 2}, "$unset": ["old"]}
    ) as cursor:
        assert len(cursor.batch) == 1
    async with session.bucket.query(bucket, {}) as cursor:
        document = cursor.batch[0]
        assert document["version"] == 2
        assert "old" not in document


@pytest.mark.integration
async def test_update_leaves_an_unchanged_document_out(session, bucket):
    """A matched document the update leaves as it was is not in the batch."""
    await session.bucket.insert(bucket, {"status": "active"})
    async with session.bucket.update(
        bucket, {}, {"$set": {"status": "active"}}
    ) as cursor:
        assert cursor.batch == []


@pytest.mark.integration
async def test_update_with_array_filters(session, bucket):
    await session.bucket.insert(bucket, {"grades": [50, 80]})
    update = {
        "$set": {"grades.$[elem]": 100},
        "array_filters": [{"elem": {"$gte": 60}}],
    }
    async with session.bucket.update(bucket, {}, update) as cursor:
        assert len(cursor.batch) == 1
    async with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch[0]["grades"] == [50, 100]


@pytest.mark.integration
async def test_update_with_upsert(session, bucket):
    update = {"$set": {"name": "alice", "active": True}, "upsert": True}
    async with session.bucket.update(bucket, {"name": "alice"}, update):
        pass
    async with session.bucket.query(bucket, {"name": "alice"}) as cursor:
        assert cursor.batch[0]["active"] is True


@pytest.mark.integration
async def test_update_cannot_change_the_id(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice"})
    with pytest.raises(ResponseError):
        await session.bucket.update(bucket, {}, {"$set": {"_id": ObjectId()}})


@pytest.mark.integration
async def test_update_with_an_empty_update(session, bucket):
    with pytest.raises(ResponseError):
        await session.bucket.update(bucket, {}, {})


@pytest.mark.integration
async def test_update_with_a_sort(session, bucket):
    """``_id`` is the one field ``SORTBY`` always has an order for."""
    ids = await session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    async with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, sort=("_id", "desc"), batch=1
    ) as cursor:
        assert cursor.batch == [max(ids)]


@pytest.mark.integration
async def test_update_with_a_batch(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    async with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, batch=2
    ) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
async def test_update_with_a_limit(session, bucket):
    await session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    sizes = []
    async with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, batch=2, limit=3
    ) as cursor:
        async for batch in cursor.batches():
            sizes.append(len(batch))
    assert sizes == [2, 1]
    assert cursor.limit_reached
    async with session.bucket.query(bucket, {"seen": True}) as cursor:
        assert len(cursor.batch) == 3


@pytest.mark.integration
async def test_update_with_close(session, bucket):
    await session.bucket.insert(bucket, *({"n": n} for n in range(5)))
    sizes = []
    async with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, batch=2, close=True
    ) as cursor:
        async for batch in cursor.batches():
            sizes.append(len(batch))
    assert sizes == [2]
    assert cursor.cursor_id == -1
    assert (await session.bucket.cursors())["update"] == {}
    async with session.bucket.query(bucket, {"seen": True}) as cursor:
        assert len(cursor.batch) == 2


@pytest.mark.integration
async def test_update_with_a_collation(session, bucket):
    await session.bucket.insert(bucket, {"name": "Alice"})
    collation = Collation("en", strength=2)
    async with session.bucket.update(
        bucket, {"name": "alice"}, {"$set": {"verified": True}}, collation=collation
    ) as cursor:
        assert len(cursor.batch) == 1


@pytest.mark.integration
async def test_update_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.update(MISSING, {}, {"$set": {"seen": True}})


@pytest.mark.integration
async def test_update_a_removed_bucket(session, bucket):
    await session.bucket.remove(bucket)
    with pytest.raises(BucketBeingRemovedError):
        await session.bucket.update(bucket, {}, {"$set": {"seen": True}})


@pytest.mark.integration
async def test_update_advance(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    async with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, batch=1
    ) as cursor:
        first = cursor.batch
        second = await cursor.advance()
        assert cursor.batch is second
        assert first != second


@pytest.mark.integration
async def test_update_batches(session, bucket):
    ids = await session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    updated = []
    async with session.bucket.update(
        bucket, {}, {"$set": {"seen": True}}, batch=2
    ) as cursor:
        async for batch in cursor.batches():
            if not batch:
                break
            updated.extend(batch)
    assert sorted(updated) == sorted(ids)


@pytest.mark.integration
async def test_update_is_awaitable_and_a_context_manager(session, bucket):
    opener = session.bucket.update(bucket, {}, {"$set": {"seen": True}})
    assert isinstance(opener, Awaitable)
    assert isinstance(opener, AbstractAsyncContextManager)
    await (await opener).close()


@pytest.mark.integration
async def test_the_update_cursor_id_holds_without_a_limit(session, bucket):
    await session.bucket.insert(bucket, {"n": 1})
    async with session.bucket.update(bucket, {}, {"$set": {"seen": True}}) as cursor:
        cursor_id = cursor.cursor_id
        await cursor.advance()
        assert cursor.cursor_id == cursor_id


@pytest.mark.integration
async def test_update_close(session, bucket):
    cursor = await session.bucket.update(bucket, {}, {"$set": {"seen": True}})
    await cursor.close()
    with pytest.raises(ResponseError):
        await session.bucket.advance(cursor.cursor_id, "update")


@pytest.mark.integration
async def test_update_close_twice_on_the_cursor(session, bucket):
    cursor = await session.bucket.update(bucket, {}, {"$set": {"seen": True}})
    await cursor.close()
    await cursor.close()


@pytest.mark.integration
async def test_update_close_twice_on_the_group(session, bucket):
    cursor = await session.bucket.update(bucket, {}, {"$set": {"seen": True}})
    assert await session.bucket.close(cursor.cursor_id, "update") is True
    with pytest.raises(ResponseError):
        await session.bucket.close(cursor.cursor_id, "update")


@pytest.mark.integration
async def test_an_update_cursor_shows_up_in_cursors(session, bucket):
    async with session.bucket.update(
        bucket, {"n": 1}, {"$set": {"seen": True}}
    ) as cursor:
        cursors = await session.bucket.cursors()
        assert cursors["update"][cursor.cursor_id] == {"n": 1}
    assert (await session.bucket.cursors())["update"] == {}


@pytest.mark.integration
async def test_update_with_a_bytes_object_id_format(kr, bucket):
    config = KronotopConfig(object_id_format="bytes")
    async with kr.session(config=config) as s:
        ids = await s.bucket.insert(bucket, {"name": "Alice"})
        async with s.bucket.update(bucket, {}, {"$set": {"seen": True}}) as cursor:
            assert cursor.batch == ids


@pytest.mark.integration
async def test_update_in_a_transaction(session, bucket):
    ids = await session.bucket.insert(bucket, {"status": "pending"})
    async with (
        session.transaction(),
        session.bucket.update(bucket, {}, {"$set": {"status": "active"}}) as cursor,
    ):
        assert cursor.batch == ids
    async with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch[0]["status"] == "active"


@pytest.mark.integration
async def test_update_rolled_back(session, bucket):
    await session.bucket.insert(bucket, {"status": "pending"})
    await session.begin()
    async with session.bucket.update(
        bucket, {}, {"$set": {"status": "active"}}
    ) as cursor:
        assert len(cursor.batch) == 1
    await session.rollback()
    async with session.bucket.query(bucket, {}) as cursor:
        assert cursor.batch[0]["status"] == "pending"


@pytest.mark.integration
async def test_advance(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    async with session.bucket.query(bucket, {}, batch=1) as cursor:
        first = cursor.batch
        second = await cursor.advance()
        assert cursor.batch is second
        assert first != second


@pytest.mark.integration
async def test_batches(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2}, {"n": 3})
    sizes = []
    async with session.bucket.query(bucket, {}, batch=2) as cursor:
        async for batch in cursor.batches():
            sizes.append(len(batch))
            if len(sizes) == 3:
                break
    assert sizes == [2, 1, 0]


@pytest.mark.integration
async def test_query_is_awaitable_and_a_context_manager(session, bucket):
    opener = session.bucket.query(bucket, {})
    assert isinstance(opener, Awaitable)
    assert isinstance(opener, AbstractAsyncContextManager)
    await (await opener).close()


@pytest.mark.integration
async def test_the_cursor_id_holds_without_a_limit(session, bucket):
    """Every reply echoes the same id until ``limit`` is reached, which is
    why ``advance`` needs no id."""
    await session.bucket.insert(bucket, {"n": 1})
    async with session.bucket.query(bucket, {}) as cursor:
        cursor_id = cursor.cursor_id
        await cursor.advance()
        assert cursor.cursor_id == cursor_id


@pytest.mark.integration
async def test_group_advance_returns_the_record(session, bucket):
    await session.bucket.insert(bucket, {"n": 1}, {"n": 2})
    async with session.bucket.query(bucket, {}, batch=1) as cursor:
        batch = await session.bucket.advance(cursor.cursor_id)
        assert isinstance(batch, QueryBatch)
        assert batch.cursor_id == cursor.cursor_id
        assert len(batch.documents) == 1


@pytest.mark.integration
async def test_close(session, bucket):
    cursor = await session.bucket.query(bucket, {})
    await cursor.close()
    with pytest.raises(ResponseError):
        await session.bucket.advance(cursor.cursor_id)


@pytest.mark.integration
async def test_close_twice_on_the_cursor(session, bucket):
    """The cursor holds a flag, so the second call sends nothing."""
    cursor = await session.bucket.query(bucket, {})
    await cursor.close()
    await cursor.close()


@pytest.mark.integration
async def test_cursor_close_failure_does_not_mask_the_original_error(
    kr, bucket, timeout_once
):
    async with kr.session() as s:
        await s.ping()
        conn = s._redis.connection
        with pytest.raises(redis.TimeoutError):
            async with await s.bucket.query(bucket, {}) as cursor:
                conn.read_response = timeout_once(conn)
                await cursor.advance()


@pytest.mark.integration
async def test_close_twice_on_the_group(session, bucket):
    cursor = await session.bucket.query(bucket, {})
    assert await session.bucket.close(cursor.cursor_id) is True
    with pytest.raises(ResponseError):
        await session.bucket.close(cursor.cursor_id)


@pytest.mark.integration
async def test_cursors(session, bucket):
    async with session.bucket.query(bucket, {"n": 1}) as cursor:
        cursors = await session.bucket.cursors()
        assert cursors["query"][cursor.cursor_id] == {"n": 1}
        assert cursors["update"] == {}
        assert cursors["delete"] == {}
    assert (await session.bucket.cursors())["query"] == {}


@pytest.mark.integration
async def test_cursors_for_one_operation(session, bucket):
    async with session.bucket.query(bucket, {}) as cursor:
        cursors = await session.bucket.cursors("query")
        assert list(cursors) == ["query"]
        assert cursor.cursor_id in cursors["query"]


def _host_of(client, session):
    """The member a session opened on, as a lookup key. See the sync twin."""
    name = next(name for name, pool in client._pools.items() if pool is session._pool)
    return member_key(name)


@pytest.mark.integration
async def test_session_for_opens_where_the_bucket_is(kr, session, bucket):
    located = await session.bucket.locate(bucket)
    holders = {
        member_key(located.members[shard.primary][0]) for shard in located.shards
    }
    namespace = await session.namespace.current()
    async with kr.session_for(buckets={namespace: [bucket]}) as opened:
        assert _host_of(kr, opened) in holders


@pytest.mark.integration
async def test_session_for_keeps_the_placement(kr, session, bucket):
    namespace = await session.namespace.current()
    async with kr.session_for(buckets={namespace: [bucket]}):
        pass
    kept = kr._placement.get(namespace, bucket)
    assert kept
    async with kr.session_for(buckets={namespace: [bucket]}) as opened:
        assert _host_of(kr, opened) in {member_key(name) for name in kept}


@pytest.mark.integration
async def test_session_for_is_awaitable(kr, session, bucket):
    """Both forms open the same session, as ``bucket.query`` does."""
    namespace = await session.namespace.current()
    opened = await kr.session_for(buckets={namespace: [bucket]})
    try:
        assert await opened.ping() is True
    finally:
        await opened.close()


@pytest.mark.integration
async def test_session_for_serves_two_buckets_in_one_transaction(kr, session, buckets):
    """Both on one shard, so one member holds them and nothing is rejected."""
    first = await buckets()
    shard_id = (await session.bucket.locate(first)).shards[0].shard_id
    second = await buckets(create=False)
    await session.bucket.create(second, shards=[shard_id])

    namespace = await session.namespace.current()
    async with kr.session_for(buckets={namespace: [first, second]}) as opened:
        await opened.namespace.use(namespace)
        async with opened.transaction():
            await opened.bucket.insert(first, {"name": "Alice"})
            await opened.bucket.insert(second, {"name": "Bob"})


@pytest.mark.integration
async def test_session_for_a_missing_bucket(kr, session):
    namespace = await session.namespace.current()
    with pytest.raises(NoSuchBucketError):
        await kr.session_for(buckets={namespace: [MISSING]})


@pytest.mark.integration
async def test_vector(session, vector_bucket):
    results = await search(session, vector_bucket)
    assert [result.entry["label"] for result in results] == ["beta", "gamma", "alpha"]
    assert results[0].score == pytest.approx(1.0)
    assert isinstance(results[0].entry["_id"], ObjectId)


@pytest.mark.integration
async def test_vector_with_top(session, vector_bucket):
    results = await search(session, vector_bucket, expected=2, top=2)
    assert [result.entry["label"] for result in results] == ["beta", "gamma"]


@pytest.mark.integration
async def test_vector_with_a_threshold(session, vector_bucket):
    results = await search(session, vector_bucket, expected=2, threshold=0.99)
    assert [result.entry["label"] for result in results] == ["beta", "gamma"]


@pytest.mark.integration
async def test_vector_with_a_filter(session, vector_bucket):
    results = await search(
        session, vector_bucket, expected=1, filter_={"label": "alpha"}
    )
    assert [result.entry["label"] for result in results] == ["alpha"]


@pytest.mark.integration
async def test_vector_with_a_projection(session, vector_bucket):
    results = await search(session, vector_bucket, projection={"_id": 0, "label": 1})
    assert results[0].entry == {"label": "beta"}


@pytest.mark.integration
async def test_vector_with_scan_options(session, vector_bucket):
    results = await search(
        session, vector_bucket, filter_={}, max_scan_candidates=100, overquery=2.0
    )
    assert len(results) == 3


@pytest.mark.integration
async def test_vector_with_a_wrong_dimension(session, vector_bucket):
    """The server checks the dimension before it checks the index is ready."""
    with pytest.raises(ResponseError):
        await session.bucket.vector(vector_bucket, "embedding", [0.1, 0.2])


@pytest.mark.integration
async def test_vector_without_an_index(session, bucket):
    with pytest.raises(ResponseError):
        await session.bucket.vector(bucket, "embedding", QUERY_VECTOR)


@pytest.mark.integration
async def test_vector_a_missing_bucket(session):
    with pytest.raises(NoSuchBucketError):
        await session.bucket.vector(MISSING, "embedding", QUERY_VECTOR)


@pytest.mark.integration
async def test_vector_with_a_json_reply_type(kr, vector_bucket):
    async with kr.session(config=KronotopConfig(reply_type="json")) as s:
        results = await search(s, vector_bucket)
        assert results[0].entry["label"] == "beta"
        assert isinstance(results[0].entry["_id"], ObjectId)


@pytest.mark.integration
async def test_the_commands_run_in_the_namespace_given(session, namespace, buckets):
    bucket = await buckets(namespace=namespace)
    current = await session.namespace.current()
    assert bucket in await session.bucket.list(namespace=namespace)
    assert bucket not in await session.bucket.list()
    assert (await session.bucket.locate(bucket, namespace=namespace)).shards

    documents = [{"n": 1}, {"n": 2}, {"n": 3}]
    ids = await session.bucket.insert(bucket, *documents, namespace=namespace)
    assert len(ids) == 3
    assert "plan" in await session.bucket.explain(bucket, {}, namespace=namespace)
    async with session.bucket.query(bucket, {}, batch=2, namespace=namespace) as cursor:
        assert len(cursor.batch) == 2
        # ADVANCE takes no namespace. The cursor stays where it was opened.
        assert len(await cursor.advance()) == 1
    change = {"$set": {"n": 10}}
    async with session.bucket.update(
        bucket, {"n": 1}, change, namespace=namespace
    ) as cursor:
        assert cursor.batch == [ids[0]]
    async with session.bucket.delete(bucket, {"n": 10}, namespace=namespace) as cursor:
        assert cursor.batch == [ids[0]]

    # The session did not move, and its own namespace does not hold the bucket.
    assert await session.namespace.current() == current
    with pytest.raises(NoSuchBucketError):
        await session.bucket.insert(bucket, {"n": 1})


@pytest.mark.integration
async def test_the_index_commands_run_in_the_namespace_given(
    session, namespace, buckets
):
    bucket = await buckets(namespace=namespace)
    index = SingleFieldIndex("age", "int32", name="idx_age")
    created = await session.bucket.create_index(bucket, index, namespace=namespace)
    assert created is True
    names = await session.bucket.list_indexes(bucket, namespace=namespace)
    assert "idx_age" in names
    described = await session.bucket.describe_index(
        bucket, "idx_age", namespace=namespace
    )
    assert described["selector"] == "age"
    tasks = await session.bucket.index_tasks(bucket, "idx_age", namespace=namespace)
    assert isinstance(tasks, dict)
    with pytest.raises(NoSuchBucketError):
        await session.bucket.list_indexes(bucket)


@pytest.mark.integration
async def test_vector_in_the_namespace_given(session, namespace, buckets):
    bucket = await buckets(namespace=namespace)
    await session.bucket.create_index(bucket, VECTOR_INDEX, namespace=namespace)
    await session.bucket.insert(bucket, *VECTOR_DOCUMENTS, namespace=namespace)
    results = await search(session, bucket, namespace=namespace)
    assert [result.entry["label"] for result in results] == ["beta", "gamma", "alpha"]


@pytest.mark.integration
async def test_a_namespace_that_is_not_there(session, namespaces):
    with pytest.raises(NoSuchNamespaceError):
        await session.bucket.list(namespace=await namespaces(create=False))


@pytest.mark.integration
async def test_session_for_finds_a_bucket_in_another_namespace(
    kr, session, namespace, buckets
):
    bucket = await buckets(namespace=namespace)
    location = await session.bucket.locate(bucket, namespace=namespace)
    holders = {
        member_key(location.members[shard.primary][0]) for shard in location.shards
    }
    async with kr.session_for(buckets={namespace: [bucket]}) as opened:
        assert _host_of(kr, opened) in holders
