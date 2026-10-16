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

"""Response callbacks for Kronotop-specific replies.

redis-py 8.x builds ``response_callbacks`` per instance, in ``Redis.__init__``,
and the class level ``RESPONSE_CALLBACKS`` dict is gone. So the callbacks are
registered on the live client, in ``register_callbacks``.

redis-py keys a callback by ``args[0]``, so one callback serves every
subcommand of a family and branches on an option the method passes. Without
the option, an ``OK`` reply becomes ``True`` and anything else passes through,
which covers a caller who sends the command raw.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Protocol

from bson import ObjectId, decode
from bson.json_util import loads
from redis.utils import str_if_bytes

from kronotop.commands.bucket import (
    BucketLocation,
    ObjectIdBatch,
    QueryBatch,
    ShardLocation,
    VectorSearchResult,
)

__all__ = ["register_callbacks"]


class SupportsResponseCallbacks(Protocol):
    """The slice of a redis-py client this module needs.

    A Protocol, because the sync and async clients share no base class.
    """

    def set_response_callback(
        self, command: str, callback: Callable[..., Any]
    ) -> None: ...


def parse_ping(response: Any, ping_message: bool = False) -> Any:
    """Turn a bare ``PING`` into ``True``, or pass an echoed message through.

    ``ConnectionCommands.ping`` sets ``ping_message`` when it sends an argument.
    """
    if ping_message:
        return response
    return str_if_bytes(response) == "PONG"


def parse_info(response: Any, **_options: Any) -> dict[str, str]:
    """Turn ``INFO`` into a flat dict of strings.

    The server sends a verbatim string on RESP3 and a bulk string on RESP2, and
    both parsers drop the ``txt:`` prefix. The text holds ``# Section`` headers
    and ``key:value`` lines. The headers are dropped, so every group ends up in
    one dict.

    Every value stays a string, because a number parser would turn
    ``kronotop_version:1.10`` into ``1.1``.
    """
    info: dict[str, str] = {}
    for line in str_if_bytes(response).splitlines():
        key, separator, value = line.partition(":")
        if separator and not key.startswith("#"):
            info[key] = value
    return info


def bool_ok(response: Any, **_options: Any) -> bool:
    """Turn an ``OK`` reply into ``True``."""
    return str_if_bytes(response) == "OK"


def parse_commit(response: Any, commit_returning: bool = False, **_options: Any) -> Any:
    """Collapse a plain ``COMMIT`` to ``True``, or pass a ``RETURNING`` value on.

    ``TransactionCommands.commit`` sets ``commit_returning``. The reply is then
    an integer or a bulk string instead of ``OK``.
    """
    if commit_returning:
        return response
    return bool_ok(response)


def _as_dict(value: Any) -> dict[Any, Any]:
    """A map reply as a dict on either protocol.

    RESP3 sends a map, which arrives as a dict. RESP2 has no map, so the server
    sends a flat key, value array and this pairs it up.
    """
    if isinstance(value, dict):
        return value
    return dict(zip(value[::2], value[1::2], strict=True))


def parse_session_attribute(response: Any, **_options: Any) -> Any:
    """Normalise ``SESSION.ATTRIBUTE``.

    Branches on the reply shape, because callers also send the command raw.
    ``SET`` replies ``OK``, ``LIST`` a map that ``_as_dict`` reads on both
    protocols.
    """
    if isinstance(response, (dict, list)):
        return _as_dict(response)
    return bool_ok(response)


def parse_namespace(
    response: Any,
    namespace_ok: bool = False,
    namespace_exists: bool = False,
    **_options: Any,
) -> Any:
    """Normalise a ``NAMESPACE`` subcommand reply.

    ``USE``, ``CREATE``, ``REMOVE``, ``MOVE`` and ``PURGE`` set ``namespace_ok``
    and reply ``OK``. ``EXISTS`` sets ``namespace_exists`` and answers ``1`` or
    ``0`` on RESP2 and ``true`` or ``false`` on RESP3, so ``bool`` covers both.
    ``CURRENT`` and ``LIST`` need no option and are decoded to ``str``.

    The reply shape cannot replace the options, because ``CURRENT`` returns a
    bulk string and a namespace may be named ``OK``. So a raw subcommand gets
    an ``OK`` reply as the string ``"OK"``.
    """
    if namespace_ok:
        return bool_ok(response)
    if namespace_exists:
        return bool(response)
    if isinstance(response, list):
        return [str_if_bytes(name) for name in response]
    return str_if_bytes(response)


#: The ``BUCKET.INDEX DESCRIBE`` fields that hold a map, and the ones that hold
#: a boolean. RESP2 sends a boolean as ``1`` or ``0``, so the name decides.
_INDEX_MAPS = frozenset({"collation", "statistics"})
_INDEX_FLAGS = frozenset(
    {"unique", "case_level", "numeric_ordering", "backwards", "normalization"}
)


def _decode_map(
    fields: Any,
    maps: frozenset[str],
    lists: frozenset[str],
    flags: frozenset[str],
) -> dict[str, Any]:
    """Decode a map reply whose field names say what each value holds.

    On RESP2 a map and an array are both arrays, and a boolean is ``1`` or
    ``0``, so the shape cannot tell them apart. ``maps`` names the fields that
    hold a map, ``lists`` the arrays of maps, ``flags`` the booleans. Keys and
    text values are decoded, ``int`` and ``None`` stay as they are, and any
    other array is decoded item by item.
    """
    decoded: dict[str, Any] = {}
    for key, value in _as_dict(fields).items():
        key = str_if_bytes(key)
        if key in maps:
            value = _decode_map(value, maps, lists, flags)
        elif key in lists:
            value = [_decode_map(item, maps, lists, flags) for item in value]
        elif key in flags:
            value = bool(value)
        elif isinstance(value, list):
            value = [str_if_bytes(item) for item in value]
        else:
            value = str_if_bytes(value)
        decoded[key] = value
    return decoded


def _index_description(fields: Any) -> dict[str, Any]:
    """Decode ``BUCKET.INDEX DESCRIBE``, and the maps inside it.

    A compound index carries ``fields``, an array of maps, one per field.
    """
    return _decode_map(fields, _INDEX_MAPS, frozenset({"fields"}), _INDEX_FLAGS)


def _index_tasks(response: Any) -> dict[str, dict[str, str]]:
    """Decode ``BUCKET.INDEX TASKS``. Task id -> its fields, all text."""
    return {
        str_if_bytes(task_id): {
            str_if_bytes(key): str_if_bytes(value)
            for key, value in _as_dict(task).items()
        }
        for task_id, task in _as_dict(response).items()
    }


def parse_bucket_index(
    response: Any, index_subcommand: str | None = None, **_options: Any
) -> Any:
    """Normalise a ``BUCKET.INDEX`` subcommand reply.

    ``list_indexes``, ``describe_index`` and ``index_tasks`` pass
    ``index_subcommand``, because on RESP2 all three replies are flat arrays.
    ``CREATE``, ``DROP`` and ``ANALYZE`` reply ``OK``.
    """
    if index_subcommand == "LIST":
        return parse_names(response)
    if index_subcommand == "DESCRIBE":
        return _index_description(response)
    if index_subcommand == "TASKS":
        return _index_tasks(response)
    if isinstance(response, (bytes, str)):
        return bool_ok(response)
    return response


#: The ``BUCKET.EXPLAIN`` fields that hold a map, an array of maps, and a
#: boolean. ``query_collation`` and ``index_collation`` carry the flags in
#: ``_INDEX_FLAGS``.
_EXPLAIN_MAPS = frozenset(
    {"plan", "query_collation", "index_collation", "predicate", "next"}
)
_EXPLAIN_LISTS = frozenset({"filters", "children"})
_EXPLAIN_FLAGS = _INDEX_FLAGS | {
    "is_cached",
    "includeLower",
    "includeUpper",
    "collation_mismatch",
}


def parse_bucket_explain(response: Any, **_options: Any) -> dict[str, Any]:
    """Decode ``BUCKET.EXPLAIN``: ``is_cached``, ``plan`` and, with a
    ``COLLATION``, ``query_collation``.

    An operand has no fixed type, so it is decoded by shape. Under
    ``protocol=2`` a double operand arrives as the 8 raw bytes ``parse_double``
    reads, and stays that way here.
    """
    return _decode_map(response, _EXPLAIN_MAPS, _EXPLAIN_LISTS, _EXPLAIN_FLAGS)


def parse_names(response: Any, **_options: Any) -> list[str]:
    """Decode an array of names: ``BUCKET.LIST``, ``COMMAND LIST``,
    ``COMMAND GETKEYS`` and ``COMMAND HELP``.

    A name is text, so it comes back as ``str`` even without
    ``decode_responses=True``.
    """
    return [str_if_bytes(name) for name in response]


def parse_bucket_locate(response: Any, **_options: Any) -> BucketLocation:
    """Group ``BUCKET.LOCATE`` into one record per shard, plus the members.

    The server sends two flat arrays on both protocols, three elements per
    shard and two per member. Ids and addresses are text, so they are decoded
    here too.
    """
    routes, members = response
    return BucketLocation(
        [
            ShardLocation(
                shard_id,
                str_if_bytes(primary),
                tuple(str_if_bytes(standby) for standby in standbys),
            )
            for shard_id, primary, standbys in zip(
                routes[::3], routes[1::3], routes[2::3], strict=True
            )
        ],
        {
            str_if_bytes(member_id): tuple(str_if_bytes(a) for a in addresses)
            for member_id, addresses in zip(members[::2], members[1::2], strict=True)
        },
    )


def _object_ids(response: Any, object_id_format: str | int) -> list[ObjectId]:
    """Decode an array of ObjectIds in the wire form ``object_id_format`` names,
    24 characters of hex or the raw 12 bytes."""
    if object_id_format == "bytes":
        return [ObjectId(bytes(oid)) for oid in response]
    return [ObjectId(str_if_bytes(oid)) for oid in response]


def parse_bucket_insert(
    response: Any, object_id_format: str | int = "hex", **_options: Any
) -> list[ObjectId]:
    """Turn ``BUCKET.INSERT`` into a list of :class:`bson.ObjectId`.

    ``Bucket.insert`` passes ``object_id_format``. A raw command passes
    nothing, so the server default ``hex`` is assumed.

    A client built with both ``object_id_format="bytes"`` and
    ``decode_responses=True`` never gets here, because the parser decodes the
    bulk string first and raises ``UnicodeDecodeError`` on the raw bytes.
    """
    return _object_ids(response, object_id_format)


def _document_from_json(entry: Any) -> dict[str, Any]:
    """Decode one JSON document and give ``_id`` its type back.

    The server writes ``_id`` as 24 characters of hex instead of the extended
    JSON form ``{"$oid": ...}``, so the field is turned back by hand. A
    document then reads the same under either ``reply_type``.
    """
    document: dict[str, Any] = loads(str_if_bytes(entry))
    oid = document.get("_id")
    if isinstance(oid, str):
        document["_id"] = ObjectId(oid)
    return document


def _cursor_fields(response: Any, key: str) -> tuple[int, Any]:
    """Split a cursor reply into the id and the payload under ``key``.

    RESP3 sends a map of ``cursor_id`` and the payload, RESP2 an array of the
    two. The map keys are ``bytes`` unless the client decodes, so they are
    decoded here.
    """
    if isinstance(response, dict):
        fields = {str_if_bytes(k): value for k, value in response.items()}
        return int(fields["cursor_id"]), fields[key]
    cursor_id, payload = response
    return int(cursor_id), payload


def _documents(entries: Any, reply_type: str | int) -> list[dict[str, Any]]:
    """Decode the documents of a reply in the format ``reply_type`` names.

    A client built with both ``reply_type="bson"`` and ``decode_responses=True``
    never gets here, because the parser decodes the bulk string first and
    raises ``UnicodeDecodeError`` on the BSON bytes.
    """
    if reply_type == "json":
        return [_document_from_json(entry) for entry in entries]
    return [decode(bytes(entry)) for entry in entries]


def parse_bucket_query(
    response: Any, reply_type: str | int = "bson", **_options: Any
) -> QueryBatch:
    """Turn ``BUCKET.QUERY`` into a :class:`QueryBatch`.

    ``Bucket.query`` and ``Bucket.advance`` pass ``reply_type``. A raw command
    passes nothing, so the server default ``bson`` is assumed.
    """
    cursor_id, entries = _cursor_fields(response, "entries")
    return QueryBatch(cursor_id, _documents(entries, reply_type))


def parse_bucket_vector(
    response: Any, reply_type: str | int = "bson", **_options: Any
) -> list[VectorSearchResult]:
    """Turn ``BUCKET.VECTOR`` into a list of :class:`VectorSearchResult`.

    RESP3 sends an array of maps with ``score`` and ``entry``, RESP2 an array
    of pairs. The score is a double on RESP3 and its 8 byte big-endian form on
    RESP2, the two shapes ``parse_double`` reads. ``Bucket.vector`` passes
    ``reply_type``, as in ``parse_bucket_query``.
    """
    results = []
    for item in response:
        if isinstance(item, dict):
            fields = {str_if_bytes(k): value for k, value in item.items()}
            score, entry = fields["score"], fields["entry"]
        else:
            score, entry = item
        (document,) = _documents([entry], reply_type)
        results.append(VectorSearchResult(parse_double(score), document))
    return results


def parse_bucket_delete(
    response: Any, object_id_format: str | int = "hex", **_options: Any
) -> ObjectIdBatch:
    """Turn ``BUCKET.DELETE`` or ``BUCKET.UPDATE`` into an :class:`ObjectIdBatch`.

    The two replies share one shape. The payload is the ids of the changed
    documents, under the same ``object_id_format`` rules as
    ``parse_bucket_insert``.
    """
    cursor_id, object_ids = _cursor_fields(response, "object_ids")
    return ObjectIdBatch(cursor_id, _object_ids(object_ids, object_id_format))


def parse_bucket_advance(
    response: Any, operation: str = "query", **options: Any
) -> Any:
    """Pick the parser of one ``BUCKET.ADVANCE`` reply.

    Both replies are a cursor id plus an array, so ``Bucket.advance`` passes
    ``operation``. A raw command passes nothing, so ``query`` is assumed.
    """
    if operation in ("update", "delete"):
        return parse_bucket_delete(response, **options)
    return parse_bucket_query(response, **options)


def parse_bucket_cursors(
    response: Any, **_options: Any
) -> dict[str, dict[int, dict[str, Any]]]:
    """Group ``BUCKET.CURSORS`` by operation, then by cursor id.

    RESP3 sends a map of maps, RESP2 a flat array of flat arrays, and
    ``_as_dict`` reads both. The operation names are lowercased to match what
    ``cursors`` takes as an argument.

    The server serialises every query as JSON, whatever the cursor was opened
    with, so a query comes back as the ``dict`` it was.
    """
    cursors: dict[str, dict[int, dict[str, Any]]] = {}
    for operation, open_cursors in _as_dict(response).items():
        cursors[str_if_bytes(operation).lower()] = {
            int(cursor_id): loads(str_if_bytes(query))
            for cursor_id, query in _as_dict(open_cursors).items()
        }
    return cursors


#: The ``COMMAND INFO`` key specification fields that hold a map.
_KEY_SPEC_MAPS = frozenset({"begin_search", "find_keys", "spec"})


def _command_info(entry: Any) -> dict[str, Any]:
    """Decode one ``COMMAND INFO`` entry, and the subcommand entries in it.

    The entry is a fixed array in the order the spec's table gives. The three
    sets are text, so they become lists of ``str``. A key spec is a map of
    maps, which ``_decode_map`` reads on both protocols. The subcommands have
    the same shape, keyed by their ``container|sub`` name.
    """
    name, arity, flags, first, last, step, categories, tips, specs, subs = entry
    return {
        "name": str_if_bytes(name),
        "arity": arity,
        "flags": parse_names(flags),
        "first_key": first,
        "last_key": last,
        "step": step,
        "acl_categories": parse_names(categories),
        "tips": parse_names(tips),
        "key_specifications": [
            _decode_map(spec, _KEY_SPEC_MAPS, frozenset(), frozenset())
            for spec in specs
        ],
        "subcommands": parse_command_info(subs),
    }


def parse_command_info(
    response: Any, command_names: tuple[str, ...] = (), **_options: Any
) -> dict[str, dict[str, Any] | None]:
    """Key ``COMMAND`` and ``COMMAND INFO`` by command name.

    An unknown name is a null entry in its position. ``Command.info`` passes
    the names it asked for as ``command_names``, so the null keeps its key,
    lowercased. A raw command without names gets no null entry.
    """
    info: dict[str, dict[str, Any] | None] = {}
    for position, entry in enumerate(response):
        if entry is None:
            info[command_names[position].lower()] = None
        else:
            decoded = _command_info(entry)
            info[decoded["name"]] = decoded
    return info


def _command_argument(fields: Any) -> dict[str, Any]:
    """Decode one ``COMMAND DOCS`` argument. A ``oneof`` or a ``block``
    nests its own ``arguments``."""
    argument: dict[str, Any] = {}
    for key, value in _as_dict(fields).items():
        key = str_if_bytes(key)
        if key == "arguments":
            value = [_command_argument(item) for item in value]
        elif key == "flags":
            value = parse_names(value)
        else:
            value = str_if_bytes(value)
        argument[key] = value
    return argument


#: JSON Schema keys whose value is an array of scalars.
_SCHEMA_SCALAR_LISTS = frozenset({"enum", "required", "type", "examples"})


def _reply_schema(value: Any, key: str = "") -> Any:
    """Decode the ``reply_schema`` of a ``COMMAND DOCS`` entry, a JSON Schema.

    The server sends a JSON object as a map and a JSON array as an array. On
    RESP2 a map is a flat key/value array too, so the shape alone cannot tell
    ``properties`` (a map) from ``oneOf`` (an array). An array of schemas holds
    only maps and a flat map starts with a key, which tells them apart.
    ``items`` takes either form. An array of strings such as ``enum`` looks
    like a flat map on RESP2, so those keys are named.

    A scalar keeps its type. RESP2 sends a boolean as ``0`` or ``1``.
    """
    if isinstance(value, dict):
        return {
            str_if_bytes(k): _reply_schema(v, str_if_bytes(k)) for k, v in value.items()
        }
    if isinstance(value, list):
        if key in _SCHEMA_SCALAR_LISTS:
            return [str_if_bytes(item) for item in value]
        if value and all(isinstance(item, (list, dict)) for item in value):
            return [_reply_schema(item) for item in value]
        return _reply_schema(_as_dict(value))
    return str_if_bytes(value)


def _command_doc(fields: Any) -> dict[str, Any]:
    """Decode one ``COMMAND DOCS`` entry, and the subcommand docs in it.

    ``_decode_map`` does not fit here. ``history`` holds pairs, the keys of
    ``subcommands`` are command names, and ``reply_schema`` is a JSON Schema
    of any depth.
    """
    doc: dict[str, Any] = {}
    for key, value in _as_dict(fields).items():
        key = str_if_bytes(key)
        if key == "arguments":
            value = [_command_argument(item) for item in value]
        elif key == "history":
            value = [tuple(parse_names(pair)) for pair in value]
        elif key == "reply_schema":
            value = _reply_schema(value)
        elif key == "doc_flags":
            value = parse_names(value)
        elif key == "subcommands":
            value = parse_command_docs(value)
        else:
            value = str_if_bytes(value)
        doc[key] = value
    return doc


def parse_command_docs(response: Any, **_options: Any) -> dict[str, dict[str, Any]]:
    """Key ``COMMAND DOCS`` by command name.

    RESP3 sends a map, RESP2 a flat array, and ``_as_dict`` reads both. The
    server skips an unknown name, so there is no null entry to handle.
    """
    return {
        str_if_bytes(name): _command_doc(doc)
        for name, doc in _as_dict(response).items()
    }


def parse_command_getkeysandflags(
    response: Any, **_options: Any
) -> list[tuple[str, list[str]]]:
    """Pair up ``COMMAND GETKEYSANDFLAGS``. Each key with the flags of its key
    specification, on both protocols."""
    return [(str_if_bytes(key), parse_names(flags)) for key, flags in response]


def _admin_map(fields: Any) -> dict[str, Any]:
    """Decode one ``KR.ADMIN`` map, a member or a shard.

    Every value is text, an array of text, or an integer, so ``_decode_map``
    needs no field names.
    """
    return _decode_map(fields, frozenset(), frozenset(), frozenset())


def _admin_cluster(response: Any) -> dict[str, Any]:
    """Decode ``DESCRIBE-CLUSTER``.

    ``metadata_version`` and ``cluster_name`` are text. Every other key is a
    shard kind, ``bucket`` or ``stash``, holding a map keyed by shard id, an
    integer on both protocols.
    """
    cluster: dict[str, Any] = {}
    for key, value in _as_dict(response).items():
        key = str_if_bytes(key)
        if isinstance(value, (bytes, str)):
            cluster[key] = str_if_bytes(value)
        else:
            cluster[key] = {
                int(shard_id): _admin_map(shard)
                for shard_id, shard in _as_dict(value).items()
            }
    return cluster


def parse_kr_admin(
    response: Any, admin_subcommand: str | None = None, **_options: Any
) -> Any:
    """Normalise a ``KR.ADMIN`` subcommand reply.

    ``AdminCommands`` passes ``admin_subcommand``, because on RESP2 the map
    replies are flat arrays. ``DROP-CLUSTER`` answers a token in its first
    phase and ``OK`` in its second, so the caller sends the first as
    ``DROP-CLUSTER-TOKEN``.
    """
    if admin_subcommand == "DESCRIBE-CLUSTER":
        return _admin_cluster(response)
    if admin_subcommand in ("DESCRIBE-SHARD", "DESCRIBE-MEMBER", "FIND-MEMBER"):
        return _admin_map(response)
    if admin_subcommand == "LIST-MEMBERS":
        return {
            str_if_bytes(member_id): _admin_map(member)
            for member_id, member in _as_dict(response).items()
        }
    if admin_subcommand == "LIST-SILENT-MEMBERS":
        return parse_names(response)
    if admin_subcommand == "DROP-CLUSTER-TOKEN":
        return str_if_bytes(response)
    if isinstance(response, (bytes, str)):
        return bool_ok(response)
    return response


#: The ``TASK.ADMIN LIST`` fields that hold a boolean.
_TASK_FLAGS = frozenset({"running", "finished"})


def parse_task_admin(response: Any, **_options: Any) -> dict[str, Any]:
    """Decode ``TASK.ADMIN LIST``, a map of task name to task fields.

    ``LIST`` is the only subcommand, so no option is needed.
    """
    return {
        str_if_bytes(name): _decode_map(task, frozenset(), frozenset(), _TASK_FLAGS)
        for name, task in _as_dict(response).items()
    }


def parse_zgetrange(response: Any, **_options: Any) -> list[tuple[Any, Any]]:
    """Pair up ``ZGETRANGE``.

    The server sends an array of two element arrays on both protocols. Tuples
    unpack in a ``for key, value in`` loop, and the list keeps the key order
    the command asked for.
    """
    return [(key, value) for key, value in response]


def parse_double(response: Any, **_options: Any) -> Any:
    """Give a double back as a ``float`` on either protocol: the ``ZGET.F64``
    reply, a ``BUCKET.VECTOR`` score, a volume field.

    RESP3 has a double type, so the reply arrives decoded and passes through.
    RESP2 has none, so the server sends the 8 byte big-endian IEEE 754 form
    and this unpacks it.

    A client built with both ``protocol=2`` and ``decode_responses=True`` gets
    no float, because the parser decodes the bulk string first. The 8 bytes of
    a value such as ``12.5`` are valid UTF-8 and pass through as a ``str``.
    Most other values raise ``UnicodeDecodeError`` before this runs.
    """
    if isinstance(response, bytes):
        return struct.unpack(">d", response)[0]
    return response


def parse_zgetd128(response: Any, **_options: Any) -> Decimal | None:
    """Turn ``ZGET.D128`` into a :class:`decimal.Decimal`.

    The server answers plain decimal text, never scientific notation, so the
    conversion is exact and keeps trailing zeros.
    """
    if response is None:
        return None
    return Decimal(str_if_bytes(response))


#: The volume fields that hold a double.
_VOLUME_DOUBLES = frozenset({"garbage_percentage", "fill_ratio"})


def _volume_map(fields: Any) -> dict[str, Any]:
    """Decode a flat volume map, a ``DESCRIBE`` segment or a ``VOLUME.STATS`` reply.

    ``_decode_map`` does not fit here, because a double on RESP2 is the 8 raw
    bytes ``parse_double`` reads.
    """
    volume: dict[str, Any] = {}
    for key, value in _as_dict(fields).items():
        key = str_if_bytes(key)
        if key in _VOLUME_DOUBLES:
            volume[key] = parse_double(value)
        else:
            volume[key] = str_if_bytes(value)
    return volume


def _volume_description(response: Any) -> dict[str, Any]:
    """Decode ``VOLUME.ADMIN DESCRIBE``.

    ``segments`` is a map keyed by segment id, an integer on both protocols.
    The rest is text or an integer.
    """
    volume: dict[str, Any] = {}
    for key, value in _as_dict(response).items():
        key = str_if_bytes(key)
        if key == "segments":
            volume[key] = {
                int(segment_id): _volume_map(segment)
                for segment_id, segment in _as_dict(value).items()
            }
        else:
            volume[key] = str_if_bytes(value)
    return volume


#: The ``VOLUME.ADMIN VACUUM STATUS`` fields that hold a map, an array of
#: maps, and a boolean.
_VACUUM_MAPS = frozenset({"metadata"})
_VACUUM_LISTS = frozenset({"segments"})
_VACUUM_FLAGS = frozenset({"active"})


def parse_volume_admin(
    response: Any, volume_subcommand: str | None = None, **_options: Any
) -> Any:
    """Normalise a ``VOLUME.ADMIN`` subcommand reply.

    ``VolumeCommands`` passes ``volume_subcommand``, because on RESP2 the map
    replies are flat arrays. ``VACUUM`` and ``MARK-STALE-PREFIXES`` answer
    ``OK`` for every operation but one, so the caller names that one as
    ``VACUUM STATUS`` or ``LOCATE``.
    """
    if volume_subcommand in ("LIST", "CLEANUP-ORPHAN-FILES"):
        return parse_names(response)
    if volume_subcommand == "DESCRIBE":
        return _volume_description(response)
    if volume_subcommand == "VACUUM STATUS":
        return _decode_map(response, _VACUUM_MAPS, _VACUUM_LISTS, _VACUUM_FLAGS)
    if volume_subcommand == "LOCATE":
        return _admin_map(response)
    if isinstance(response, (bytes, str)):
        return bool_ok(response)
    return response


#: The ``VOLUME.INSPECT REPLICATION`` fields that hold a map.
_REPLICATION_MAPS = frozenset({"cursor", "cdc_stage", "segment_replication_stage"})


def parse_volume_inspect(
    response: Any, inspect_subcommand: str | None = None, **_options: Any
) -> Any:
    """Normalise a ``VOLUME.INSPECT`` subcommand reply.

    Both replies are maps of text and integers. ``REPLICATION`` nests three
    maps, and ``_decode_map`` needs their names on RESP2.
    """
    if inspect_subcommand == "CURSOR":
        return _admin_map(response)
    if inspect_subcommand == "REPLICATION":
        return _decode_map(response, _REPLICATION_MAPS, frozenset(), frozenset())
    return response


def parse_volume_stats(
    response: Any, stats_subcommand: str | None = None, **_options: Any
) -> Any:
    """Normalise a ``VOLUME.STATS`` reply.

    ``VolumeCommands`` passes ``stats_subcommand``, because on RESP2 a map and
    an array of maps are both arrays. ``OVERVIEW`` names the call without a
    subcommand.
    """
    if stats_subcommand in ("OVERVIEW", "OPCOUNTERS", "REPLICATION"):
        return _volume_map(response)
    if stats_subcommand == "SEGMENTS":
        return [_volume_map(segment) for segment in response]
    if isinstance(response, (bytes, str)):
        return bool_ok(response)
    return response


def parse_segment_tail_pointer(response: Any, **_options: Any) -> dict[str, int]:
    """Name the two integers of ``SEGMENT.TAILPOINTER``."""
    next_position, sequence_number = response
    return {"next_position": next_position, "sequence_number": sequence_number}


#: The ``CHANGELOG.RANGE`` entry fields that hold a map.
_CHANGELOG_MAPS = frozenset({"before", "after"})


def parse_changelog_range(response: Any, **_options: Any) -> list[dict[str, Any]]:
    """Decode ``CHANGELOG.RANGE``, an array of entry maps."""
    return [
        _decode_map(entry, _CHANGELOG_MAPS, frozenset(), frozenset())
        for entry in response
    ]


#: The ``TRANSACTION LIST`` and ``SLOW`` entry fields that hold a boolean.
_TRANSACTION_FLAGS = frozenset({"watching", "committed", "conflicted"})


def parse_transaction(response: Any, **_options: Any) -> Any:
    """Decode ``TRANSACTION``. ``KILL`` answers ``OK``, ``LIST`` an array of maps."""
    if isinstance(response, (bytes, str)):
        return bool_ok(response)
    return [
        _decode_map(entry, frozenset(), frozenset(), _TRANSACTION_FLAGS)
        for entry in response
    ]


#: Command name -> callback, matched case insensitively by redis-py. A callback
#: takes the reply plus the keyword options ``execute_command`` forwards, so it
#: has to accept them even when it reads none.
CALLBACKS: dict[str, Callable[..., Any]] = {
    "PING": parse_ping,
    "INFO": parse_info,
    "BEGIN": bool_ok,
    "COMMIT": parse_commit,
    "ROLLBACK": bool_ok,
    "SNAPSHOTREAD": bool_ok,
    "TRANSACTION": parse_transaction,
    "SESSION.CLOSE": bool_ok,
    "SESSION.ATTRIBUTE": parse_session_attribute,
    "NAMESPACE": parse_namespace,
    "BUCKET.CREATE": bool_ok,
    "BUCKET.INDEX": parse_bucket_index,
    "BUCKET.LIST": parse_names,
    "BUCKET.LOCATE": parse_bucket_locate,
    "BUCKET.INSERT": parse_bucket_insert,
    "BUCKET.QUERY": parse_bucket_query,
    "BUCKET.EXPLAIN": parse_bucket_explain,
    "BUCKET.UPDATE": parse_bucket_delete,
    "BUCKET.DELETE": parse_bucket_delete,
    "BUCKET.VECTOR": parse_bucket_vector,
    "BUCKET.ADVANCE": parse_bucket_advance,
    "BUCKET.CLOSE": bool_ok,
    "BUCKET.CURSORS": parse_bucket_cursors,
    "BUCKET.REMOVE": bool_ok,
    "BUCKET.PURGE": bool_ok,
    # One key per subcommand, because ``Command`` sends ``"COMMAND INFO"`` as
    # one string and redis-py looks the callback up by that whole string. The
    # ``COMMAND`` entry replaces redis-py's own, which raises on the null entry
    # the server sends for an unknown name. ``COMMAND GETKEYS`` replaces its
    # RESP2 entry too.
    "COMMAND": parse_command_info,
    "COMMAND INFO": parse_command_info,
    "COMMAND DOCS": parse_command_docs,
    "COMMAND LIST": parse_names,
    "COMMAND GETKEYS": parse_names,
    "COMMAND GETKEYSANDFLAGS": parse_command_getkeysandflags,
    "COMMAND HELP": parse_names,
    "KR.ADMIN": parse_kr_admin,
    "TASK.ADMIN": parse_task_admin,
    "VOLUME.ADMIN": parse_volume_admin,
    "VOLUME.INSPECT": parse_volume_inspect,
    "VOLUME.STATS": parse_volume_stats,
    "SEGMENT.TAILPOINTER": parse_segment_tail_pointer,
    "CHANGELOG.RANGE": parse_changelog_range,
    "ZSET": bool_ok,
    "ZDEL": bool_ok,
    "ZGETRANGE": parse_zgetrange,
    "ZDELRANGE": bool_ok,
    # Its own key, because redis-py matches the whole ``args[0]``, never a prefix.
    "ZSET.I64": bool_ok,
    "ZINC.I64": bool_ok,
    "ZSET.F64": bool_ok,
    "ZINC.F64": bool_ok,
    "ZGET.F64": parse_double,
    "ZSET.D128": bool_ok,
    "ZINC.D128": bool_ok,
    "ZGET.D128": parse_zgetd128,
    "ZMUTATE": bool_ok,
    "ZWATCH": bool_ok,
}


def register_callbacks(client: SupportsResponseCallbacks) -> None:
    """Register every Kronotop callback on ``client``."""
    for command, callback in CALLBACKS.items():
        client.set_response_callback(command, callback)
