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

"""Bucket commands, reached through ``session.bucket``.

A bucket holds documents and belongs to a namespace, so the same name in two
namespaces is two different buckets.

A command works in the session's current namespace. Every command that names a
bucket, and ``list``, takes ``namespace`` to run that one command elsewhere.
The session's namespace does not change, and the one given has to exist or the
command raises :class:`~kronotop.exceptions.NoSuchNamespaceError`. The cursor
commands take none, because a cursor stays in the namespace it was opened in.
"""

from __future__ import annotations

import json
import struct
from collections.abc import AsyncIterator, Awaitable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass
from types import TracebackType
from typing import Any, Generic, Literal, NamedTuple, TypeVar, overload

from bson import ObjectId, encode
from bson.json_util import dumps

from kronotop._awaitable import AwaitableContext
from kronotop._cleanup import ignore_cleanup_error
from kronotop.commands.base import (
    AsyncClient,
    CommandGroup,
    CommandsBase,
    SyncClient,
    namespace_args,
)

__all__ = [
    "AsyncBucket",
    "AsyncCursor",
    "BsonType",
    "Bucket",
    "BucketCommands",
    "BucketLocation",
    "Collation",
    "CompoundIndex",
    "CompoundIndexField",
    "Cursor",
    "DistanceFunction",
    "IndexSpec",
    "ObjectIdBatch",
    "Operation",
    "QueryBatch",
    "ShardLocation",
    "SingleFieldIndex",
    "SortDirection",
    "SortSpec",
    "VectorIndex",
    "VectorSearchResult",
]


def _set_fields(spec: Any) -> dict[str, Any]:
    """The fields of a dataclass that are set, under the names the server uses.

    ``asdict`` walks into a nested dataclass, so a :class:`Collation` inside an
    index loses its ``None`` fields too. A plain dict the caller passed is
    copied as it is, because the factory runs on dataclass levels only.
    """
    fields: dict[str, Any] = asdict(
        spec, dict_factory=lambda items: {k: v for k, v in items if v is not None}
    )
    return fields


class _Spec:
    """A dataclass that goes over the wire as JSON, without its ``None`` fields."""

    def to_dict(self) -> dict[str, Any]:
        """The fields that are set, under the names the server uses."""
        return _set_fields(self)


@dataclass(frozen=True)
class Collation(_Spec):
    """How strings are compared and ordered.

    Kronotop compares strings byte by byte by default. A collation applies the
    rules of one language instead, so a search can ignore case or accents.

    ``locale`` is an ICU locale name such as ``"tr"``, the only field the
    server needs. Every other field is keyword only, and ``None`` is not sent,
    so the server default holds.

    ```python
    Collation("tr", strength=2)  # Turkish, case is ignored
    ```
    """

    #: ICU locale name, for example ``"tr"``.
    locale: str
    _: KW_ONLY
    #: How fine the comparison is, 1 to 5. Lower ignores more.
    strength: int | None = None
    #: Add a case level between the first and the second comparison pass.
    case_level: bool | None = None
    #: Which case sorts first.
    case_first: Literal["upper", "lower", "off"] | None = None
    #: Compare a run of digits as a number, so ``a2`` sorts before ``a10``.
    numeric_ordering: bool | None = None
    #: ``"shifted"`` makes spaces and punctuation ignorable.
    alternate: Literal["non-ignorable", "shifted"] | None = None
    #: Read the accent pass backwards. Some French rules need it.
    backwards: bool | None = None
    #: Normalise Unicode before comparing.
    normalization: bool | None = None
    #: With ``alternate="shifted"``, what becomes ignorable.
    max_variable: Literal["punct", "space"] | None = None


#: The BSON types an index can be built on, as ``bson_type`` spells them.
BsonType = Literal[
    "string",
    "int32",
    "int64",
    "double",
    "boolean",
    "datetime",
    "timestamp",
    "binary",
    "objectid",
    "decimal128",
]

#: How a vector index measures similarity, as ``distance`` spells it.
DistanceFunction = Literal["cosine", "euclidean", "dot_product"]


@dataclass(frozen=True)
class SingleFieldIndex(_Spec):
    """An index on one field.

    ``selector`` is the field, in dot notation for a nested one such as
    ``"address.city"``. ``bson_type`` is the one type the index holds. Every
    other field is keyword only, and ``None`` is not sent.

    ```python
    SingleFieldIndex("email", "string", unique=True)
    SingleFieldIndex("tags", "string", multi_key=True)
    ```
    """

    #: Field the index is built on, in dot notation.
    selector: str
    #: Type of the indexed values. A value of another type is not indexed.
    bson_type: BsonType
    _: KW_ONLY
    #: Index each element of an array on its own.
    multi_key: bool | None = None
    #: Reject a document whose value is already in the index. Cannot be
    #: combined with ``multi_key``.
    unique: bool | None = None
    #: Name of the index. The server makes one from the selector and the type
    #: when it is left out.
    name: str | None = None
    #: String ordering of the index. ``string`` type only. Without it the
    #: bucket's collation holds.
    collation: Collation | dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The fields that are set, without ``selector``.

        The server reads the selector as the key of the schema object, so
        :func:`_index_schema` puts it there.
        """
        fields = super().to_dict()
        del fields["selector"]
        return fields


@dataclass(frozen=True)
class CompoundIndexField(_Spec):
    """One field of a :class:`CompoundIndex`, in the order the index has them."""

    #: Field the index is built on, in dot notation.
    selector: str
    #: Type of the indexed values.
    bson_type: BsonType
    _: KW_ONLY
    #: Index each element of an array on its own. One field per index at most.
    multi_key: bool | None = None


@dataclass(frozen=True)
class CompoundIndex(_Spec):
    """An index on several fields, in a fixed order.

    ``fields`` are two to 32 :class:`CompoundIndexField`, in the order queries
    have to follow. Every other field is keyword only, and ``None`` is not
    sent.

    ```python
    CompoundIndex(
        [
            CompoundIndexField("category", "string"),
            CompoundIndexField("price", "double"),
        ],
        name="idx_cat_price",
    )
    ```
    """

    #: The fields, leading one first.
    fields: Sequence[CompoundIndexField]
    _: KW_ONLY
    #: Reject a document whose combination of values is already in the index.
    #: Cannot be combined with a ``multi_key`` field.
    unique: bool | None = None
    #: Name of the index. The server makes one from the fields when it is
    #: left out.
    name: str | None = None
    #: String ordering of the index. Needs at least one ``string`` field.
    collation: Collation | dict[str, Any] | None = None


@dataclass(frozen=True)
class VectorIndex(_Spec):
    """An index for similarity search, read by ``BUCKET.VECTOR``.

    ``field`` holds the vector, ``dimensions`` is its length and ``distance``
    is how two vectors are compared. One vector index per field, and the
    bucket has to be on one shard.

    ```python
    VectorIndex("embedding", 768, "cosine")
    ```
    """

    #: Field that holds the vector, in dot notation.
    field: str
    #: Number of elements of each vector, at least 1.
    dimensions: int
    #: How similarity is measured.
    distance: DistanceFunction
    _: KW_ONLY
    #: Name of the index. The server makes one from the field, the dimensions
    #: and the distance when it is left out.
    name: str | None = None


#: What ``create_index`` and ``create(indexes=...)`` take, next to a plain
#: mapping.
IndexSpec = SingleFieldIndex | CompoundIndex | VectorIndex


class ShardLocation(NamedTuple):
    """One shard of a bucket, the way ``locate`` reports it.

    A tuple as well as a record, so it unpacks:
    ``for shard_id, primary, standbys in session.bucket.locate("users").shards``.

    The two member fields hold an id. The addresses are in
    :attr:`BucketLocation.members`, under that id.
    """

    #: Shard that holds this part of the bucket.
    shard_id: int
    #: Id of the member that owns the shard.
    primary: str
    #: Id of each standby. Empty when the shard has none.
    standbys: tuple[str, ...]


class BucketLocation(NamedTuple):
    """Where a bucket lives, the way ``locate`` reports it.

    The shards name their members by id, and ``members`` maps each id to its
    addresses, so a member that owns several shards is spelled out once.
    """

    #: One record per shard, in the order the server sends them.
    shards: list[ShardLocation]
    #: Member id -> the ``host:port`` it accepts clients on. The first address
    #: is the preferred one. A standby is in here too.
    members: dict[str, tuple[str, ...]]


#: Direction of a sort, as ``SORTBY`` and ``RESULTSORT`` spell it.
SortDirection = Literal["asc", "desc"]

#: One field and one direction, e.g. ``("age", "desc")``. Kronotop sorts on a
#: single field.
SortSpec = tuple[str, SortDirection]

#: The kind of cursor a command opened.
Operation = Literal["query", "update", "delete"]


class QueryBatch(NamedTuple):
    """One reply of ``BUCKET.QUERY`` or ``BUCKET.ADVANCE QUERY``.

    The cursor id is the same in every reply of one cursor. The reply that
    reaches ``limit``, or the reply of a command sent with ``close``, carries
    ``-1`` and the cursor is gone.
    """

    #: Cursor the batch came from.
    cursor_id: int
    #: The documents of this batch. Empty does not mean the cursor is done.
    documents: list[dict[str, Any]]


class ObjectIdBatch(NamedTuple):
    """One reply of ``BUCKET.UPDATE``, ``BUCKET.DELETE`` or their
    ``BUCKET.ADVANCE``.

    The write twin of :class:`QueryBatch`. The payload is the ids of the
    documents the batch changed, and the cursor id follows the same ``-1``
    rule.
    """

    #: Cursor the batch came from.
    cursor_id: int
    #: Ids of the documents this batch updated or deleted. Empty does not mean
    #: the cursor is done.
    object_ids: list[ObjectId]


class VectorSearchResult(NamedTuple):
    """One result of ``BUCKET.VECTOR``. The field names are the server's."""

    #: Similarity score. Higher is closer, and the range depends on the
    #: distance function of the index.
    score: float
    #: The matching document, decoded the same way as a query result.
    entry: dict[str, Any]


#: What ``list`` returns. Spelled out here because inside the class ``list`` is
#: the method, and an IDE reads the annotation that way.
_Names = list[str]
_ObjectIds = list[ObjectId]
_Documents = list[dict[str, Any]]
_Cursors = dict[str, dict[int, dict[str, Any]]]
_IndexDescription = dict[str, Any]
_IndexTasks = dict[str, dict[str, str]]
_QueryPlan = dict[str, Any]
_VectorSearchResults = list[VectorSearchResult]

#: What one batch of a cursor holds: ``_Documents`` on a query cursor,
#: ``_ObjectIds`` on a delete cursor. A bound rather than a value restriction,
#: because PyCharm reads a restricted TypeVar as the union and flags ``yield
#: self.batch`` in ``batches``.
BatchT = TypeVar("BatchT", bound=list[Any])


def _collation_json(collation: Collation | dict[str, Any]) -> str:
    """Serialise a collation as JSON, the form the server reads everywhere."""
    if isinstance(collation, Collation):
        collation = collation.to_dict()
    return json.dumps(collation)


def _index_schema(indexes: Sequence[IndexSpec | Mapping[str, Any]]) -> str:
    """Build the schema ``BUCKET.INDEX CREATE`` and ``BUCKET.CREATE INDEXES`` read.

    A single-field index is keyed by its selector, the compound indexes go
    under ``$compound`` and the vector index under ``$vector``. A mapping is
    merged in as it is, for a form this client does not know yet.

    A second index on one selector, or a second vector index, would overwrite
    the first inside the dict and reach the server as one index, so both raise
    ``ValueError`` here. A ``$compound`` list is copied, so the caller's
    mapping is not changed.
    """
    schema: dict[str, Any] = {}
    for index in indexes:
        entries: Mapping[str, Any]
        if isinstance(index, SingleFieldIndex):
            entries = {index.selector: index.to_dict()}
        elif isinstance(index, CompoundIndex):
            entries = {"$compound": [index.to_dict()]}
        elif isinstance(index, VectorIndex):
            entries = {"$vector": index.to_dict()}
        else:
            entries = index
        for key, value in entries.items():
            if key == "$compound":
                schema.setdefault(key, []).extend(value)
            elif key == "$vector" and key in schema:
                raise ValueError("one vector index per call")
            elif key in schema:
                raise ValueError(f"two indexes on {key!r}")
            else:
                schema[key] = value
    if not schema:
        raise ValueError("at least one index is needed")
    return json.dumps(schema)


def _insert_args(
    bucket: str,
    documents: Sequence[Mapping[str, Any]],
    input_type: str | int,
    namespace: str | None = None,
) -> list[Any]:
    """Build a ``BUCKET.INSERT`` call, with each document encoded.

    ``bson.encode`` and ``bson.json_util.dumps`` are the codec the server runs,
    so an ``ObjectId`` survives as a BSON ObjectId or as ``{"$oid": ...}``.
    """
    if not documents:
        raise ValueError("insert needs at least one document")
    # The server reads every argument after DOCS as a document.
    args: list[Any] = ["BUCKET.INSERT", bucket, *namespace_args(namespace), "DOCS"]
    for document in documents:
        args.append(encode(document) if input_type == "bson" else dumps(document))
    return args


def _insert(
    group: Any,
    bucket: str,
    documents: Sequence[Mapping[str, Any]],
    namespace: str | None,
) -> list[ObjectId]:
    """The synchronous arm of :meth:`BucketCommands.insert`.

    ``group`` is the :class:`~kronotop.commands.base.CommandGroup`, as ``Any``.
    Only a group carries the session attributes.
    """
    # ``_attributes`` is empty until the connection is pinned, and this reads it
    # before the first command goes out.
    attributes = group.attributes
    args = _insert_args(bucket, documents, attributes["input_type"], namespace)
    result: list[ObjectId] = group.execute_command(
        *args, object_id_format=attributes["object_id_format"]
    )
    return result


async def _ainsert(
    group: Any,
    bucket: str,
    documents: Sequence[Mapping[str, Any]],
    namespace: str | None,
) -> list[ObjectId]:
    """The asyncio arm of :meth:`BucketCommands.insert`.

    Separate because reading the attributes is a coroutine here.
    """
    attributes = await group.attributes
    args = _insert_args(bucket, documents, attributes["input_type"], namespace)
    result: list[ObjectId] = await group.execute_command(
        *args, object_id_format=attributes["object_id_format"]
    )
    return result


def _query_args(
    command: str,
    bucket: str,
    query: Mapping[str, Any],
    update: Mapping[str, Any] | None,
    sort: SortSpec | None,
    result_sort: SortSpec | None,
    projection: Mapping[str, Any] | None,
    batch: int | None,
    limit: int | None,
    collation: Collation | dict[str, Any] | None,
    namespace: str | None = None,
    close: bool = False,
) -> list[Any]:
    """Build a ``BUCKET.QUERY``, ``BUCKET.UPDATE`` or ``BUCKET.DELETE`` call.

    Each command takes a subset of the options, so a caller passes ``None``
    for what its command does not have. ``update`` is the update document of
    ``BUCKET.UPDATE`` and sits between the query and the options. ``close`` is
    the ``CLOSE`` flag, which ``BUCKET.EXPLAIN`` rejects.

    The query and the update go out as extended JSON, whatever ``input_type``
    says, because that attribute covers ``BUCKET.INSERT`` alone. The server
    tells JSON and BSON apart itself. The projection and the collation are
    JSON only.
    """
    args: list[Any] = [command, bucket, dumps(query)]
    if update is not None:
        args.append(dumps(update))
    if sort is not None:
        args.extend(("SORTBY", sort[0], sort[1].upper()))
    if result_sort is not None:
        args.extend(("RESULTSORT", result_sort[0], result_sort[1].upper()))
    if projection is not None:
        args.extend(("PROJECTION", dumps(projection)))
    if batch is not None:
        args.extend(("BATCH", batch))
    if limit is not None:
        args.extend(("LIMIT", limit))
    if collation is not None:
        args.extend(("COLLATION", _collation_json(collation)))
    args.extend(namespace_args(namespace))
    if close:
        args.append("CLOSE")
    return args


def _execute(group: Any, args: list[Any], attribute: str) -> Any:
    """Send a command that carries one session attribute to its callback.

    ``reply_type`` names the encoding of the documents in a reply and
    ``object_id_format`` the form of the ObjectIds. The callback cannot ask the
    session, so the attribute is read here and passed along. ``_aexecute`` is
    the asyncio arm, because the read is a coroutine there.
    """
    return group.execute_command(*args, **{attribute: group.attributes[attribute]})


async def _aexecute(group: Any, args: list[Any], attribute: str) -> Any:
    """The asyncio arm of ``_execute``."""
    attributes = await group.attributes
    return await group.execute_command(*args, **{attribute: attributes[attribute]})


def _query(group: Any, bucket: str, args: list[Any]) -> Cursor[_Documents]:
    """The synchronous arm of :meth:`BucketCommands.query`."""
    batch: QueryBatch = _execute(group, args, "reply_type")
    return Cursor(group, bucket, batch.cursor_id, batch.documents, "query")


async def _aquery(group: Any, bucket: str, args: list[Any]) -> AsyncCursor[_Documents]:
    """The asyncio arm of :meth:`BucketCommands.query`."""
    batch: QueryBatch = await _aexecute(group, args, "reply_type")
    return AsyncCursor(group, bucket, batch.cursor_id, batch.documents, "query")


def _write(
    group: Any, bucket: str, args: list[Any], operation: Operation
) -> Cursor[_ObjectIds]:
    """The synchronous arm of :meth:`BucketCommands.update` and
    :meth:`BucketCommands.delete`."""
    batch: ObjectIdBatch = _execute(group, args, "object_id_format")
    return Cursor(group, bucket, batch.cursor_id, batch.object_ids, operation)


async def _awrite(
    group: Any, bucket: str, args: list[Any], operation: Operation
) -> AsyncCursor[_ObjectIds]:
    """The asyncio arm of :meth:`BucketCommands.update` and
    :meth:`BucketCommands.delete`."""
    batch: ObjectIdBatch = await _aexecute(group, args, "object_id_format")
    return AsyncCursor(group, bucket, batch.cursor_id, batch.object_ids, operation)


def _vector_args(
    bucket: str,
    selector: str,
    vector: Sequence[float],
    filter_: Mapping[str, Any] | None,
    projection: Mapping[str, Any] | None,
    top: int | None,
    threshold: float | None,
    max_scan_candidates: int | None,
    overquery: float | None,
    namespace: str | None = None,
) -> list[Any]:
    """Build a ``BUCKET.VECTOR`` call.

    The vector goes out as little-endian 32 bit floats, four bytes per
    dimension. The server also reads a JSON array, but its code marks that
    form as a development aid, so this client does not send it.

    The filter and the projection go out as extended JSON, as in
    ``_query_args``. An option left at ``None`` is not sent. The client
    rejects ``top=0``, because the server reads ``TOP 0`` as its default of
    10.
    """
    if len(vector) == 0:
        raise ValueError("vector needs at least one dimension")
    if top is not None and top < 1:
        raise ValueError("top must be at least 1")
    args: list[Any] = [
        "BUCKET.VECTOR",
        bucket,
        selector,
        struct.pack(f"<{len(vector)}f", *vector),
    ]
    if filter_ is not None:
        args.extend(("FILTER", dumps(filter_)))
    if projection is not None:
        args.extend(("PROJECTION", dumps(projection)))
    if top is not None:
        args.extend(("TOP", top))
    if threshold is not None:
        args.extend(("THRESHOLD", threshold))
    if max_scan_candidates is not None:
        args.extend(("MAX-SCAN-CANDIDATES", max_scan_candidates))
    if overquery is not None:
        args.extend(("OVERQUERY", overquery))
    args.extend(namespace_args(namespace))
    return args


def _advance_options(
    operation: Operation, attributes: Mapping[str, Any]
) -> dict[str, Any]:
    """The callback options of one ``BUCKET.ADVANCE``.

    The callback is shared by every operation and needs ``operation`` to pick
    the reply shape. A query batch is decoded per ``reply_type``, a write
    batch per ``object_id_format``.
    """
    if operation == "query":
        return {"operation": operation, "reply_type": attributes["reply_type"]}
    return {
        "operation": operation,
        "object_id_format": attributes["object_id_format"],
    }


def _advance(group: Any, cursor_id: int, operation: Operation) -> Any:
    """The synchronous arm of :meth:`BucketCommands.advance`."""
    options = _advance_options(operation, group.attributes)
    return group.execute_command(
        "BUCKET.ADVANCE", operation.upper(), cursor_id, **options
    )


async def _aadvance(group: Any, cursor_id: int, operation: Operation) -> Any:
    """The asyncio arm of :meth:`BucketCommands.advance`."""
    options = _advance_options(operation, await group.attributes)
    return await group.execute_command(
        "BUCKET.ADVANCE", operation.upper(), cursor_id, **options
    )


class _CursorBase(Generic[BatchT]):
    """What a cursor holds. The I/O is in the two subclasses.

    A cursor is server side state, bound to the session that opened it. It
    never times out, so close it or close the session. ``BatchT`` is what one
    batch holds, documents on a query cursor and ObjectIds on a write cursor.
    """

    def __init__(
        self,
        group: Any,
        bucket: str,
        cursor_id: int,
        batch: BatchT,
        operation: Operation,
    ) -> None:
        self._group = group
        self._closed = False
        #: ``advance`` and ``close`` name the operation on the wire, so the
        #: cursor keeps its own.
        self._operation = operation
        #: Bucket the command ran against.
        self.bucket = bucket
        #: Id the server gave this cursor. ``-1`` once ``limit`` is reached or
        #: the command was sent with ``close``.
        self.cursor_id = cursor_id
        #: The batch that arrived last. The first one is the opening command's
        #: own reply.
        self.batch: BatchT = batch

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} bucket={self.bucket!r} cursor_id={self.cursor_id}>"
        )

    @property
    def limit_reached(self) -> bool:
        """Whether the server removed the cursor.

        The reply that reaches ``limit``, or the reply of a command sent with
        ``close``, carries cursor id ``-1``, and nothing is left to advance or
        close.
        """
        return self.cursor_id == -1

    def _take(self, record: Any) -> BatchT:
        """Keep the id and the payload of one ``BUCKET.ADVANCE`` reply."""
        # Both batch records are (cursor_id, payload).
        self.cursor_id = record.cursor_id
        self.batch = record[1]
        return self.batch


class Cursor(_CursorBase[BatchT]):
    """What ``session.bucket.query``, ``session.bucket.update`` and
    ``session.bucket.delete`` return.

    A query cursor's batch holds documents, an update or delete cursor's the
    ObjectIds of the documents the batch changed.

    ``batches`` does not end on its own. A query that matches little can hand
    back several empty batches while the scan goes on, so the caller decides
    when to stop. With ``limit`` it ends on the reply that reaches it, and
    with ``close`` after the first batch.

    ```python
    with session.bucket.query("users", {"age": {"$gt": 20}}) as cursor:
        for batch in cursor.batches():
            if not batch:
                break
            for document in batch:
                print(document)
    ```
    """

    def advance(self) -> BatchT:
        """``BUCKET.ADVANCE``. Fetch the next batch and keep it."""
        return self._take(self._group.advance(self.cursor_id, self._operation))

    def batches(self) -> Iterator[BatchT]:
        """Yield the batch in hand, then one per ``advance``, until the cursor
        id is ``-1``. Without ``limit`` or ``close`` it does not end."""
        yield self.batch
        while not self.limit_reached:
            yield self.advance()

    def close(self) -> None:
        """``BUCKET.CLOSE``. Free the cursor. Calling it twice is safe, and a
        cursor the server removed sends nothing."""
        if self._closed or self.limit_reached:
            return
        self._closed = True
        self._group.close(self.cursor_id, self._operation)

    def __enter__(self) -> Cursor[BatchT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
            return
        # The socket may be what failed, so ``BUCKET.CLOSE`` may fail too.
        with ignore_cleanup_error("close", self):
            self.close()


class AsyncCursor(_CursorBase[BatchT]):
    """The asyncio twin of :class:`Cursor`.

    ```python
    async with session.bucket.query("users", {}) as cursor:
        async for batch in cursor.batches():
            if not batch:
                break
    ```
    """

    async def advance(self) -> BatchT:
        """``BUCKET.ADVANCE``. Fetch the next batch and keep it."""
        return self._take(await self._group.advance(self.cursor_id, self._operation))

    async def batches(self) -> AsyncIterator[BatchT]:
        """Yield the batch in hand, then one per ``advance``, until the cursor
        id is ``-1``. Without ``limit`` or ``close`` it does not end."""
        yield self.batch
        while not self.limit_reached:
            yield await self.advance()

    async def close(self) -> None:
        """``BUCKET.CLOSE``. Free the cursor. Calling it twice is safe, and a
        cursor the server removed sends nothing."""
        if self._closed or self.limit_reached:
            return
        self._closed = True
        await self._group.close(self.cursor_id, self._operation)

    async def __aenter__(self) -> AsyncCursor[BatchT]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            await self.close()
            return
        with ignore_cleanup_error("close", self):
            await self.close()


class BucketCommands(CommandsBase):
    """The ``BUCKET`` command family."""

    @overload
    def create(
        self: SyncClient,
        bucket: str,
        *,
        shards: Sequence[int] | None = None,
        indexes: Sequence[IndexSpec | Mapping[str, Any]] | None = None,
        collation: Collation | dict[str, Any] | None = None,
        if_not_exists: bool = False,
        namespace: str | None = None,
    ) -> bool: ...

    @overload
    def create(
        self: AsyncClient,
        bucket: str,
        *,
        shards: Sequence[int] | None = None,
        indexes: Sequence[IndexSpec | Mapping[str, Any]] | None = None,
        collation: Collation | dict[str, Any] | None = None,
        if_not_exists: bool = False,
        namespace: str | None = None,
    ) -> Awaitable[bool]: ...

    def create(
        self,
        bucket: str,
        *,
        shards: Sequence[int] | None = None,
        indexes: Sequence[IndexSpec | Mapping[str, Any]] | None = None,
        collation: Collation | dict[str, Any] | None = None,
        if_not_exists: bool = False,
        namespace: str | None = None,
    ) -> Any:
        """``BUCKET.CREATE``. Create ``bucket`` in the current namespace, or in
        ``namespace``.

        Without ``shards`` the server picks one, round robin. With ``shards``
        the bucket spans every id given, and each id needs a known route or
        nothing is created.

        ``indexes`` is ``INDEXES``, the secondary indexes created with the
        bucket in one transaction. Each one is a :class:`SingleFieldIndex`, a
        :class:`CompoundIndex`, a :class:`VectorIndex` or a plain mapping in
        the server's schema form. See :meth:`create_index`.

        ``collation`` sets the string ordering of the whole bucket, a
        :class:`Collation` or a plain ``dict``. The server validates it.

        ``if_not_exists`` turns ``BUCKETALREADYEXISTS`` into ``OK``. An
        existing bucket is left as it is, so ``indexes`` and ``collation`` are
        not applied to it.

        A bucket that ``remove`` marked keeps its name until ``purge`` runs, so
        creating it again raises
        :class:`~kronotop.exceptions.BucketAlreadyExistsError`.
        """
        args: list[Any] = ["BUCKET.CREATE", bucket]
        if shards:
            args.append("SHARDS")
            args.extend(shards)
        if indexes:
            args.extend(("INDEXES", _index_schema(indexes)))
        if collation is not None:
            args.extend(("COLLATION", _collation_json(collation)))
        if if_not_exists:
            args.append("IF-NOT-EXISTS")
        args.extend(namespace_args(namespace))
        return self.execute_command(*args)

    @overload
    def create_index(
        self: SyncClient,
        bucket: str,
        *indexes: IndexSpec | Mapping[str, Any],
        namespace: str | None = None,
    ) -> bool: ...

    @overload
    def create_index(
        self: AsyncClient,
        bucket: str,
        *indexes: IndexSpec | Mapping[str, Any],
        namespace: str | None = None,
    ) -> Awaitable[bool]: ...

    def create_index(
        self,
        bucket: str,
        *indexes: IndexSpec | Mapping[str, Any],
        namespace: str | None = None,
    ) -> Any:
        """``BUCKET.INDEX CREATE``. Add secondary indexes to ``bucket``.

        Each index is a :class:`SingleFieldIndex`, a :class:`CompoundIndex`, a
        :class:`VectorIndex` or a plain mapping in the server's schema form.
        At least one is needed, and one call takes one vector index.

        ```python
        session.bucket.create_index(
            "users",
            SingleFieldIndex("email", "string", unique=True),
            CompoundIndex(
                [
                    CompoundIndexField("status", "string"),
                    CompoundIndexField("age", "int32"),
                ]
            ),
        )
        ```

        The reply is ``True`` once the indexes are recorded. They are built in
        the background, so a query does not use one until it is ``READY``, and
        a write sent right after this can still miss it. An index given to
        :meth:`create` is ready with the bucket. A second index on an indexed
        field, or a schema the server rejects, raises a plain
        ``redis.ResponseError``.
        """
        return self.execute_command(
            "BUCKET.INDEX",
            "CREATE",
            bucket,
            _index_schema(indexes),
            *namespace_args(namespace),
        )

    @overload
    def list_indexes(
        self: SyncClient, bucket: str, *, namespace: str | None = None
    ) -> _Names: ...

    @overload
    def list_indexes(
        self: AsyncClient, bucket: str, *, namespace: str | None = None
    ) -> Awaitable[_Names]: ...

    def list_indexes(self, bucket: str, *, namespace: str | None = None) -> Any:
        """``BUCKET.INDEX LIST``. The index names of ``bucket``.

        ``str`` on any client. ``primary-index`` is always in the list, and a
        name the server made reads ``selector:age.bsonType:INT32``.
        """
        return self.execute_command(
            "BUCKET.INDEX",
            "LIST",
            bucket,
            *namespace_args(namespace),
            index_subcommand="LIST",
        )

    @overload
    def describe_index(
        self: SyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> _IndexDescription: ...

    @overload
    def describe_index(
        self: AsyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> Awaitable[_IndexDescription]: ...

    def describe_index(
        self, bucket: str, index: str, *, namespace: str | None = None
    ) -> Any:
        """``BUCKET.INDEX DESCRIBE``. What the server holds on one index.

        A dict with ``str`` keys. ``index_type`` is ``single_field``,
        ``compound`` or ``vector`` and decides the rest. Every kind has ``id``,
        ``status`` and ``statistics``, a dict with ``cardinality``. A
        single-field index has ``selector``, ``bson_type``, ``unique`` and
        ``collation``. A compound index has ``fields``, one dict with
        ``selector`` and ``bson_type`` per field, in order. A vector index has
        ``selector``, ``dimensions`` and ``distance``.

        ``bson_type``, ``status`` and ``distance`` come in upper case, as the
        server spells them. ``unique`` and the collation flags are ``bool`` on
        both protocols, and a collation that was never set is ``{}``.

        ``status`` is ``WAITING`` or ``BUILDING`` after :meth:`create_index`,
        ``READY`` once the build is done, at once for an index :meth:`create`
        made, and ``DROPPED`` after :meth:`drop_index`. A name the bucket does
        not have raises :class:`~kronotop.exceptions.NoSuchIndexError`. The
        members learn of a new index a moment later, so a compound or vector
        index can be missing here right after :meth:`create_index`.
        """
        return self.execute_command(
            "BUCKET.INDEX",
            "DESCRIBE",
            bucket,
            index,
            *namespace_args(namespace),
            index_subcommand="DESCRIBE",
        )

    @overload
    def drop_index(
        self: SyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def drop_index(
        self: AsyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def drop_index(
        self, bucket: str, index: str, *, namespace: str | None = None
    ) -> Any:
        """``BUCKET.INDEX DROP``. Remove an index from ``bucket``.

        The index is marked ``DROPPED`` at once and stops serving queries. A
        background task removes its data, shown by :meth:`index_tasks`, and
        after that the name is gone.

        ``primary-index`` and an index with a running task, the build from
        :meth:`create_index` included, raise a plain ``redis.ResponseError``.
        A name the bucket does not have raises
        :class:`~kronotop.exceptions.NoSuchIndexError`.
        """
        return self.execute_command(
            "BUCKET.INDEX", "DROP", bucket, index, *namespace_args(namespace)
        )

    @overload
    def index_tasks(
        self: SyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> _IndexTasks: ...

    @overload
    def index_tasks(
        self: AsyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> Awaitable[_IndexTasks]: ...

    def index_tasks(
        self, bucket: str, index: str, *, namespace: str | None = None
    ) -> Any:
        """``BUCKET.INDEX TASKS``. The background tasks of one index.

        A dict keyed by task id. Each task is a dict of strings with ``kind``
        (``BUILD``, ``DROP``, ``BOUNDARY`` or ``ANALYZE``), ``status`` and
        ``error``. A ``BUILD`` task also carries ``cursor``, ``lower`` and
        ``upper``, its position in the scan. Empty once every task is done.
        """
        return self.execute_command(
            "BUCKET.INDEX",
            "TASKS",
            bucket,
            index,
            *namespace_args(namespace),
            index_subcommand="TASKS",
        )

    @overload
    def analyze_index(
        self: SyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def analyze_index(
        self: AsyncClient, bucket: str, index: str, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def analyze_index(
        self, bucket: str, index: str, *, namespace: str | None = None
    ) -> Any:
        """``BUCKET.INDEX ANALYZE``. Refresh the statistics of one index.

        The reply is ``True`` once a background task is queued. The task fills
        the ``statistics`` that :meth:`describe_index` shows and the planner
        reads. The index has to be a ``READY`` single-field or compound index
        with no analyze task running, or this raises a plain
        ``redis.ResponseError``. A name the bucket does not have raises
        :class:`~kronotop.exceptions.NoSuchIndexError`.
        """
        return self.execute_command(
            "BUCKET.INDEX", "ANALYZE", bucket, index, *namespace_args(namespace)
        )

    @overload
    def list(self: SyncClient, *, namespace: str | None = None) -> _Names: ...

    @overload
    def list(
        self: AsyncClient, *, namespace: str | None = None
    ) -> Awaitable[_Names]: ...

    def list(self, *, namespace: str | None = None) -> Any:
        """``BUCKET.LIST``. The bucket names in the current namespace, or in
        ``namespace``.

        ``str`` on any client, an empty list for a namespace with no buckets. A
        bucket that ``remove`` marked stays in the list until ``purge`` runs.
        """
        return self.execute_command("BUCKET.LIST", *namespace_args(namespace))

    @overload
    def locate(
        self: SyncClient, bucket: str, *, namespace: str | None = None
    ) -> BucketLocation: ...

    @overload
    def locate(
        self: AsyncClient, bucket: str, *, namespace: str | None = None
    ) -> Awaitable[BucketLocation]: ...

    def locate(self, bucket: str, *, namespace: str | None = None) -> Any:
        """``BUCKET.LOCATE``. Where the data of ``bucket`` lives.

        A :class:`BucketLocation` with one :class:`ShardLocation` per shard, in
        the order the server sends them, and the address of every member they
        name. A shard with no route is left out, so ``shards`` can be shorter
        than the number the bucket was created with.
        """
        return self.execute_command("BUCKET.LOCATE", bucket, *namespace_args(namespace))

    @overload
    def remove(
        self: SyncClient, bucket: str, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def remove(
        self: AsyncClient, bucket: str, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def remove(self, bucket: str, *, namespace: str | None = None) -> Any:
        """``BUCKET.REMOVE``. Mark ``bucket`` as removed.

        The first half of a deletion. The bucket stops answering commands and
        background work on it stops, but the data and the name stay until
        ``purge``. A second ``remove`` raises
        :class:`~kronotop.exceptions.BucketBeingRemovedError`.
        """
        return self.execute_command("BUCKET.REMOVE", bucket, *namespace_args(namespace))

    @overload
    def purge(
        self: SyncClient, bucket: str, *, namespace: str | None = None
    ) -> bool: ...

    @overload
    def purge(
        self: AsyncClient, bucket: str, *, namespace: str | None = None
    ) -> Awaitable[bool]: ...

    def purge(self, bucket: str, *, namespace: str | None = None) -> Any:
        """``BUCKET.PURGE``. Delete ``bucket`` for good.

        The second half of a deletion. ``remove`` has to run first, or the
        server refuses with a plain ``redis.ResponseError``.

        Every shard has to have seen the removal, and Kronotop waits about five
        seconds for that. Otherwise this raises
        :class:`~kronotop.exceptions.BarrierNotSatisfiedError`, and one retry
        is usually enough because the failed attempt publishes the removal
        again.
        """
        return self.execute_command("BUCKET.PURGE", bucket, *namespace_args(namespace))

    @overload
    def insert(
        self: SyncClient,
        bucket: str,
        *documents: Mapping[str, Any],
        namespace: str | None = None,
    ) -> _ObjectIds: ...

    @overload
    def insert(
        self: AsyncClient,
        bucket: str,
        *documents: Mapping[str, Any],
        namespace: str | None = None,
    ) -> Awaitable[_ObjectIds]: ...

    def insert(
        self,
        bucket: str,
        *documents: Mapping[str, Any],
        namespace: str | None = None,
    ) -> Any:
        """``BUCKET.INSERT``. Store ``documents`` in ``bucket``.

        Each document is a mapping, encoded here in the format the session
        attribute ``input_type`` names, BSON or extended JSON.

        The reply is one :class:`bson.ObjectId` per document, in order, and it
        comes back inside a transaction too. A document without an ``_id``
        gets one from the server. A document that carries one has to carry a
        free ``ObjectId``, or the insert raises
        :class:`~kronotop.exceptions.DuplicateKeyError`.
        """
        # Two bodies, because this reads the session attributes before it
        # builds the command, and that read is a coroutine on the asyncio
        # session.
        if self._is_async_client:
            return _ainsert(self, bucket, documents, namespace)
        return _insert(self, bucket, documents, namespace)

    @overload
    def query(
        self: SyncClient,
        bucket: str,
        query: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        result_sort: SortSpec | None = None,
        projection: Mapping[str, Any] | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> Cursor[_Documents]: ...

    @overload
    def query(
        self: AsyncClient,
        bucket: str,
        query: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        result_sort: SortSpec | None = None,
        projection: Mapping[str, Any] | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> AwaitableContext[AsyncCursor[_Documents]]: ...

    def query(
        self,
        bucket: str,
        query: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        result_sort: SortSpec | None = None,
        projection: Mapping[str, Any] | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> Any:
        """``BUCKET.QUERY``. Match documents in ``bucket`` and open a cursor.

        ``query`` is a mapping such as ``{"age": {"$gt": 20}}``, and ``{}``
        matches every document. The command runs at once and the first batch
        is in :attr:`Cursor.batch`. Read the rest with :meth:`Cursor.batches`
        and close the cursor when done.

        ``sort`` is ``SORTBY``, a field and a direction such as
        ``("age", "desc")``. It needs an index that gives that order, which
        then holds across every batch. ``result_sort`` is ``RESULTSORT``. It
        sorts each batch in memory, needs no index and says nothing about the
        order between batches. Both may be given, on two different fields.

        ``projection`` picks the fields of the reply. ``{"name": 1}`` keeps
        those, ``{"email": 0}`` drops those, and the two forms mix only for
        ``{"_id": 0}``.

        ``batch`` is how many documents come back per batch, else the session
        attribute ``batch`` decides. ``limit`` caps the whole cursor, ``0``
        means no limit, and the reply that reaches it carries cursor id ``-1``,
        see :attr:`Cursor.limit_reached`. ``collation`` is the string ordering
        of the query and does not reach ``sort``.

        ``close`` is ``CLOSE``, for a caller that needs one batch. The reply
        carries the first batch and cursor id ``-1``, and no cursor stays in
        the session.

        The documents come back as ``dict``, decoded in the format the session
        attribute ``reply_type`` names, so an ``_id`` is a
        :class:`bson.ObjectId` either way. On the asyncio client the return
        value is awaited, or entered with ``async with``.
        """
        args = _query_args(
            "BUCKET.QUERY",
            bucket,
            query,
            None,
            sort,
            result_sort,
            projection,
            batch,
            limit,
            collation,
            namespace,
            close,
        )
        # The asyncio body is wrapped, so ``async with`` needs no ``await``.
        # See ``insert`` for the split.
        if self._is_async_client:
            return AwaitableContext(_aquery(self, bucket, args))
        return _query(self, bucket, args)

    @overload
    def explain(
        self: SyncClient,
        bucket: str,
        query: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> _QueryPlan: ...

    @overload
    def explain(
        self: AsyncClient,
        bucket: str,
        query: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> Awaitable[_QueryPlan]: ...

    def explain(
        self,
        bucket: str,
        query: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> Any:
        """``BUCKET.EXPLAIN``. The plan of a query, without running it.

        ``query``, ``sort``, ``batch``, ``limit`` and ``collation`` are as in
        :meth:`query`, and they are what the planner reads. No cursor is
        opened.

        The reply is a dict with ``is_cached`` and ``plan``, plus
        ``query_collation`` when ``collation`` was given. ``plan`` is a node
        with ``planner_version``, ``nodeType``, ``id``, the fields of its kind
        and ``next`` for the node after it. A ``FullScan`` carries ``index``
        and ``predicate``, an ``IndexScan`` ``selector``, ``operator`` and
        ``operand``, a ``RangeScan`` the two bounds, a ``CompoundIndexScan``
        its ``filters``, and ``Union`` and ``OrderedConcat`` their
        ``children``. A ``predicate`` has a ``type`` of ``PREDICATE``,
        ``AND``, ``OR`` or ``ALWAYS_TRUE``. Keys are ``str`` and booleans
        ``bool`` on both protocols.

        ``is_cached`` is ``True`` when a query command already ran the same
        structure and operators, whatever the values. This command never fills
        the cache. A bucket the namespace does not have raises
        :class:`~kronotop.exceptions.NoSuchBucketError`.
        """
        args = _query_args(
            "BUCKET.EXPLAIN",
            bucket,
            query,
            None,
            sort,
            None,
            None,
            batch,
            limit,
            collation,
            namespace,
        )
        return self.execute_command(*args)

    @overload
    def delete(
        self: SyncClient,
        bucket: str,
        query: Mapping[str, Any],
        *,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> Cursor[_ObjectIds]: ...

    @overload
    def delete(
        self: AsyncClient,
        bucket: str,
        query: Mapping[str, Any],
        *,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> AwaitableContext[AsyncCursor[_ObjectIds]]: ...

    def delete(
        self,
        bucket: str,
        query: Mapping[str, Any],
        *,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> Any:
        """``BUCKET.DELETE``. Delete the documents in ``bucket`` that match
        ``query``, and open a cursor.

        ``query``, ``limit``, ``collation`` and ``close`` are as in
        :meth:`query`, and so is the cursor. ``{}`` matches every document.
        Each batch holds the :class:`bson.ObjectId` of the documents it
        deleted, and ``batch`` is how many come back per batch. With ``close``
        the command deletes the first batch and leaves no cursor. A delete has
        no sort and no projection.

        Outside a transaction each batch is committed at once. Inside
        ``BEGIN``/``COMMIT`` nothing is deleted before the commit, and
        ``rollback`` cancels it.
        """
        args = _query_args(
            "BUCKET.DELETE",
            bucket,
            query,
            None,
            None,
            None,
            None,
            batch,
            limit,
            collation,
            namespace,
            close,
        )
        if self._is_async_client:
            return AwaitableContext(_awrite(self, bucket, args, "delete"))
        return _write(self, bucket, args, "delete")

    @overload
    def update(
        self: SyncClient,
        bucket: str,
        query: Mapping[str, Any],
        update: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> Cursor[_ObjectIds]: ...

    @overload
    def update(
        self: AsyncClient,
        bucket: str,
        query: Mapping[str, Any],
        update: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> AwaitableContext[AsyncCursor[_ObjectIds]]: ...

    def update(
        self,
        bucket: str,
        query: Mapping[str, Any],
        update: Mapping[str, Any],
        *,
        sort: SortSpec | None = None,
        batch: int | None = None,
        limit: int | None = None,
        collation: Collation | dict[str, Any] | None = None,
        namespace: str | None = None,
        close: bool = False,
    ) -> Any:
        """``BUCKET.UPDATE``. Change the documents in ``bucket`` that match
        ``query``, and open a cursor.

        ``query``, ``sort``, ``limit``, ``collation`` and ``close`` are as in
        :meth:`query`, and so is the cursor. ``{}`` matches every document.
        ``update`` carries the operators: ``{"$set": {...}}`` sets fields,
        ``{"$unset": [...]}`` removes them, the two combine, and
        ``array_filters`` and ``upsert`` go in the same mapping. The server
        rejects an empty update, and ``_id`` cannot be set.

        Each batch holds the :class:`bson.ObjectId` of the documents it
        modified, and ``batch`` is how many come back per batch. A matched
        document the update left as it was is not in there. An upsert counts
        as one toward ``limit``, and with ``close`` the command updates the
        first batch and leaves no cursor. An update has no ``result_sort`` and
        no ``projection``.

        Outside a transaction each batch is committed at once. Inside
        ``BEGIN``/``COMMIT`` nothing changes before the commit, and
        ``rollback`` cancels it.
        """
        args = _query_args(
            "BUCKET.UPDATE",
            bucket,
            query,
            update,
            sort,
            None,
            None,
            batch,
            limit,
            collation,
            namespace,
            close,
        )
        if self._is_async_client:
            return AwaitableContext(_awrite(self, bucket, args, "update"))
        return _write(self, bucket, args, "update")

    @overload
    def vector(
        self: SyncClient,
        bucket: str,
        selector: str,
        vector: Sequence[float],
        *,
        filter_: Mapping[str, Any] | None = None,
        projection: Mapping[str, Any] | None = None,
        top: int | None = None,
        threshold: float | None = None,
        max_scan_candidates: int | None = None,
        overquery: float | None = None,
        namespace: str | None = None,
    ) -> _VectorSearchResults: ...

    @overload
    def vector(
        self: AsyncClient,
        bucket: str,
        selector: str,
        vector: Sequence[float],
        *,
        filter_: Mapping[str, Any] | None = None,
        projection: Mapping[str, Any] | None = None,
        top: int | None = None,
        threshold: float | None = None,
        max_scan_candidates: int | None = None,
        overquery: float | None = None,
        namespace: str | None = None,
    ) -> Awaitable[_VectorSearchResults]: ...

    def vector(
        self,
        bucket: str,
        selector: str,
        vector: Sequence[float],
        *,
        filter_: Mapping[str, Any] | None = None,
        projection: Mapping[str, Any] | None = None,
        top: int | None = None,
        threshold: float | None = None,
        max_scan_candidates: int | None = None,
        overquery: float | None = None,
        namespace: str | None = None,
    ) -> Any:
        """``BUCKET.VECTOR``. The documents of ``bucket`` whose vector is
        closest to ``vector``.

        ``selector`` names the field of the :class:`VectorIndex`, in dot
        notation. ``vector`` is a sequence of floats with as many elements as
        the index has dimensions, and the bucket has to be on one shard.

        The reply is a list of :class:`VectorSearchResult`, best match first,
        with the documents decoded as in :meth:`query`. No cursor is opened.

        ``top`` caps the number of results and must be at least 1.
        ``threshold`` drops a result with a lower score. ``filter_`` is
        ``FILTER``, a query mapping applied after the ranking, and the server
        walks the graph until ``top`` results pass it or
        ``max_scan_candidates`` are seen. The underscore keeps the builtin
        ``filter`` in reach. ``overquery`` is a multiplier of ``1.0`` or more
        that makes the walk look at more candidates. ``projection`` is as in
        :meth:`query`. An option left at ``None`` is not sent.

        Under ``protocol=2`` the score arrives as 8 raw bytes, so a client
        with ``decode_responses=True`` fails to decode them, as with
        ``zmap.get_f64``.

        The index is updated after a commit, so a document inserted a moment
        ago may not be in the reply yet. While the index is still being built
        this raises :class:`~kronotop.exceptions.VectorIndexNotReadyError`.
        """
        args = _vector_args(
            bucket,
            selector,
            vector,
            filter_,
            projection,
            top,
            threshold,
            max_scan_candidates,
            overquery,
            namespace,
        )
        if self._is_async_client:
            return _aexecute(self, args, "reply_type")
        return _execute(self, args, "reply_type")

    @overload
    def advance(
        self: SyncClient, cursor_id: int, operation: Literal["query"] = "query"
    ) -> QueryBatch: ...

    @overload
    def advance(
        self: SyncClient, cursor_id: int, operation: Literal["update"]
    ) -> ObjectIdBatch: ...

    @overload
    def advance(
        self: SyncClient, cursor_id: int, operation: Literal["delete"]
    ) -> ObjectIdBatch: ...

    @overload
    def advance(
        self: AsyncClient, cursor_id: int, operation: Literal["query"] = "query"
    ) -> Awaitable[QueryBatch]: ...

    @overload
    def advance(
        self: AsyncClient, cursor_id: int, operation: Literal["update"]
    ) -> Awaitable[ObjectIdBatch]: ...

    @overload
    def advance(
        self: AsyncClient, cursor_id: int, operation: Literal["delete"]
    ) -> Awaitable[ObjectIdBatch]: ...

    def advance(self, cursor_id: int, operation: Operation = "query") -> Any:
        """``BUCKET.ADVANCE``. The next batch of a cursor.

        The cursor keeps everything the opening command was called with, so
        this takes the id and the operation alone. The reply is a
        :class:`QueryBatch` on a query and an :class:`ObjectIdBatch` on an
        update or a delete. A cursor id of ``-1`` means ``limit`` was reached
        and the cursor is gone.

        An empty batch does not mean the cursor is done. A cursor that was
        closed, gone at ``limit`` or by ``close``, or opened by another
        command, raises a plain ``redis.ResponseError``.
        :meth:`Cursor.advance` is the same command through the cursor.
        """
        if self._is_async_client:
            return _aadvance(self, cursor_id, operation)
        return _advance(self, cursor_id, operation)

    @overload
    def close(
        self: SyncClient, cursor_id: int, operation: Operation = "query"
    ) -> bool: ...

    @overload
    def close(
        self: AsyncClient, cursor_id: int, operation: Operation = "query"
    ) -> Awaitable[bool]: ...

    def close(self, cursor_id: int, operation: Operation = "query") -> Any:
        """``BUCKET.CLOSE``. Free a cursor.

        A cursor holds state in the session and never times out. Closing the
        session frees every cursor it opened. A second close raises a plain
        ``redis.ResponseError``.
        """
        return self.execute_command("BUCKET.CLOSE", operation.upper(), cursor_id)

    @overload
    def cursors(self: SyncClient, operation: Operation | None = None) -> _Cursors: ...

    @overload
    def cursors(
        self: AsyncClient, operation: Operation | None = None
    ) -> Awaitable[_Cursors]: ...

    def cursors(self, operation: Operation | None = None) -> Any:
        """``BUCKET.CURSORS``. The open cursors of this session.

        Keyed by operation, then by cursor id, with the query the cursor was
        opened with as the value. Without ``operation`` all three groups come
        back, and a group with no cursor is an empty dict.

        ```python
        session.bucket.cursors("query")
        # {'query': {1: {'age': {'$gt': 20}}}}
        ```
        """
        args: list[Any] = ["BUCKET.CURSORS"]
        if operation is not None:
            args.append(operation.upper())
        return self.execute_command(*args)


class Bucket(CommandGroup, BucketCommands):
    """``session.bucket`` on the synchronous session."""

    _is_async_client: Literal[False] = False


class AsyncBucket(CommandGroup, BucketCommands):
    """``session.bucket`` on the asyncio session."""

    _is_async_client: Literal[True] = True
