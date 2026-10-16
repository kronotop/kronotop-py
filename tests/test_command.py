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

"""``session.command``, the ``COMMAND`` family.

The dev server loads a handful of definitions. ``ping`` is always one of them,
and ``set`` is the one with key specifications, the example the server's own
documentation uses for ``COMMAND GETKEYS``.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop import Kronotop
from kronotop._callbacks import (
    parse_command_docs,
    parse_command_getkeysandflags,
    parse_command_info,
)
from kronotop.commands.command import Command

#: ``COMMAND INFO set`` as RESP2 hands it over: every set and map is an array.
SET_INFO_RESP2 = [
    b"set",
    -3,
    [b"write", b"denyoom"],
    1,
    1,
    1,
    [b"@write", b"@string", b"@slow"],
    [],
    [
        [
            b"notes",
            b"RW and ACCESS due to the optional `GET` argument",
            b"flags",
            [b"RW", b"access", b"update", b"variable_flags"],
            b"begin_search",
            [b"type", b"index", b"spec", [b"index", 1]],
            b"find_keys",
            [b"type", b"range", b"spec", [b"lastkey", 0, b"keystep", 1, b"limit", 0]],
        ]
    ],
    [],
]

#: The same entry on RESP3: a map is a dict, its keys still ``bytes``.
SET_INFO_RESP3 = [
    b"set",
    -3,
    [b"write", b"denyoom"],
    1,
    1,
    1,
    [b"@write", b"@string", b"@slow"],
    [],
    [
        {
            b"notes": b"RW and ACCESS due to the optional `GET` argument",
            b"flags": [b"RW", b"access", b"update", b"variable_flags"],
            b"begin_search": {b"type": b"index", b"spec": {b"index": 1}},
            b"find_keys": {
                b"type": b"range",
                b"spec": {b"lastkey": 0, b"keystep": 1, b"limit": 0},
            },
        }
    ],
    [],
]

SET_INFO = {
    "name": "set",
    "arity": -3,
    "flags": ["write", "denyoom"],
    "first_key": 1,
    "last_key": 1,
    "step": 1,
    "acl_categories": ["@write", "@string", "@slow"],
    "tips": [],
    "key_specifications": [
        {
            "notes": "RW and ACCESS due to the optional `GET` argument",
            "flags": ["RW", "access", "update", "variable_flags"],
            "begin_search": {"type": "index", "spec": {"index": 1}},
            "find_keys": {
                "type": "range",
                "spec": {"lastkey": 0, "keystep": 1, "limit": 0},
            },
        }
    ],
    "subcommands": {},
}

#: ``COMMAND DOCS auth`` on RESP2, with a ``history`` and an argument ``since``.
AUTH_DOCS_RESP2 = [
    b"auth",
    [
        b"summary",
        b"Authenticate to the server",
        b"since",
        b"1.0.0",
        b"group",
        b"connection",
        b"history",
        [[b"6.0.0", b"Added ACL style (username and password)."]],
        b"arguments",
        [
            [
                b"name",
                b"username",
                b"type",
                b"string",
                b"display_text",
                b"username",
                b"since",
                b"6.0.0",
                b"flags",
                [b"optional"],
            ],
            [b"name", b"password", b"type", b"string", b"display_text", b"password"],
        ],
    ],
]

AUTH_DOCS_RESP3 = {
    b"auth": {
        b"summary": b"Authenticate to the server",
        b"since": b"1.0.0",
        b"group": b"connection",
        b"history": [[b"6.0.0", b"Added ACL style (username and password)."]],
        b"arguments": [
            {
                b"name": b"username",
                b"type": b"string",
                b"display_text": b"username",
                b"since": b"6.0.0",
                b"flags": [b"optional"],
            },
            {b"name": b"password", b"type": b"string", b"display_text": b"password"},
        ],
    }
}

AUTH_DOCS = {
    "auth": {
        "summary": "Authenticate to the server",
        "since": "1.0.0",
        "group": "connection",
        "history": [("6.0.0", "Added ACL style (username and password).")],
        "arguments": [
            {
                "name": "username",
                "type": "string",
                "display_text": "username",
                "since": "6.0.0",
                "flags": ["optional"],
            },
            {"name": "password", "type": "string", "display_text": "password"},
        ],
    }
}


def test_the_group_is_on_the_session():
    """``session.command`` is built with the session, so it needs no server."""
    session = Kronotop().session()
    assert isinstance(session.command, Command)
    assert session.command is session.command


def test_the_group_is_not_on_the_client():
    assert not hasattr(Kronotop(), "command")


@pytest.mark.parametrize("entry", [SET_INFO_RESP2, SET_INFO_RESP3])
def test_parse_command_info_reads_both_protocols(entry):
    assert parse_command_info([entry]) == {"set": SET_INFO}


def test_parse_command_info_keys_a_null_entry_by_the_name_asked():
    """The server sends ``None`` in the position of an unknown name, so the
    key comes from the names ``info`` was called with, lowercased."""
    reply = parse_command_info([SET_INFO_RESP3, None], command_names=("SET", "NoSuch"))
    assert list(reply) == ["set", "nosuch"]
    assert reply["nosuch"] is None


def test_parse_command_info_nests_subcommands():
    sub = list(SET_INFO_RESP3)
    sub[0] = b"kr.admin|describe-cluster"
    container = list(SET_INFO_RESP3)
    container[0] = b"kr.admin"
    container[9] = [sub]
    reply = parse_command_info([container])
    nested = reply["kr.admin"]["subcommands"]
    assert list(nested) == ["kr.admin|describe-cluster"]
    assert nested["kr.admin|describe-cluster"]["subcommands"] == {}


@pytest.mark.parametrize("reply", [AUTH_DOCS_RESP2, AUTH_DOCS_RESP3])
def test_parse_command_docs_reads_both_protocols(reply):
    assert parse_command_docs(reply) == AUTH_DOCS


#: A ``reply_schema`` with every shape the bundled definitions use: an array
#: of schemas, a map of schemas, tuple ``items``, a boolean and an array of
#: strings. RESP2 sends ``False`` as ``0``, and ``0 == False``.
REPLY_SCHEMA_RESP2 = [
    b"anyOf",
    [
        [b"const", b"PONG"],
        [
            b"type",
            b"object",
            b"additionalProperties",
            0,
            b"properties",
            [
                b"cursor_id",
                [b"type", b"integer"],
                b"entries",
                [
                    b"type",
                    b"array",
                    b"items",
                    [[b"type", b"string"], [b"type", [b"string", b"null"]]],
                ],
            ],
        ],
    ],
]

REPLY_SCHEMA_RESP3 = {
    b"anyOf": [
        {b"const": b"PONG"},
        {
            b"type": b"object",
            b"additionalProperties": False,
            b"properties": {
                b"cursor_id": {b"type": b"integer"},
                b"entries": {
                    b"type": b"array",
                    b"items": [{b"type": b"string"}, {b"type": [b"string", b"null"]}],
                },
            },
        },
    ]
}

REPLY_SCHEMA = {
    "anyOf": [
        {"const": "PONG"},
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cursor_id": {"type": "integer"},
                "entries": {
                    "type": "array",
                    "items": [{"type": "string"}, {"type": ["string", "null"]}],
                },
            },
        },
    ]
}


@pytest.mark.parametrize("schema", [REPLY_SCHEMA_RESP2, REPLY_SCHEMA_RESP3])
def test_parse_command_docs_decodes_the_reply_schema(schema):
    reply = parse_command_docs(
        {b"ping": {b"group": b"connection", b"reply_schema": schema}}
    )
    assert reply["ping"]["reply_schema"] == REPLY_SCHEMA


def test_parse_command_docs_nests_subcommands_and_arguments():
    """A ``oneof`` argument carries its own ``arguments``, and a container
    carries its subcommands under their full name. ``key_spec_index`` and
    ``doc_flags`` keep their types."""
    reply = parse_command_docs(
        {
            b"kr.admin": {
                b"group": b"cluster",
                b"doc_flags": [b"syscmd"],
                b"subcommands": {
                    b"kr.admin|describe-cluster": {
                        b"group": b"cluster",
                        b"arguments": [
                            {
                                b"name": b"key",
                                b"type": b"key",
                                b"key_spec_index": 0,
                            },
                            {
                                b"name": b"direction",
                                b"type": b"oneof",
                                b"arguments": [
                                    {b"name": b"asc", b"token": b"ASC"},
                                ],
                            },
                        ],
                    }
                },
            }
        }
    )
    assert reply["kr.admin"]["doc_flags"] == ["syscmd"]
    sub = reply["kr.admin"]["subcommands"]["kr.admin|describe-cluster"]
    assert sub["arguments"][0] == {"name": "key", "type": "key", "key_spec_index": 0}
    assert sub["arguments"][1]["arguments"] == [{"name": "asc", "token": "ASC"}]


def test_parse_command_getkeysandflags():
    reply = parse_command_getkeysandflags([[b"foo", [b"RW", b"access"]]])
    assert reply == [("foo", ["RW", "access"])]


@pytest.mark.integration
def test_info(session):
    reply = session.command.info("ping")
    assert list(reply) == ["ping"]
    ping = reply["ping"]
    assert ping["name"] == "ping"
    assert ping["arity"] == -1
    assert "fast" in ping["flags"]
    assert "@connection" in ping["acl_categories"]
    assert ping["first_key"] == 0
    assert ping["key_specifications"] == []
    assert ping["subcommands"] == {}


@pytest.mark.integration
def test_info_matches_without_regard_to_case(session):
    assert list(session.command.info("PING")) == ["ping"]


@pytest.mark.integration
def test_info_with_key_specifications(session):
    (spec,) = session.command.info("zset")["zset"]["key_specifications"]
    assert "OW" in spec["flags"]
    assert spec["begin_search"] == {"type": "index", "spec": {"index": 1}}
    assert spec["find_keys"]["type"] == "range"
    assert spec["find_keys"]["spec"]["keystep"] == 1


@pytest.mark.integration
def test_info_with_an_unknown_name_maps_it_to_none(session):
    reply = session.command.info("ping", "nosuch")
    assert list(reply) == ["ping", "nosuch"]
    assert reply["nosuch"] is None
    assert reply["ping"]["name"] == "ping"


@pytest.mark.integration
def test_info_without_a_name_returns_every_command(session):
    reply = session.command.info()
    assert len(reply) == session.command.count()
    # A bare ``COMMAND`` is the same reply, through the same callback.
    assert list(session.execute_command("COMMAND")) == list(reply)


@pytest.mark.integration
def test_count(session):
    assert session.command.count() > 0


@pytest.mark.integration
def test_docs(session):
    reply = session.command.docs("ping")
    assert list(reply) == ["ping"]
    ping = reply["ping"]
    assert ping["summary"] == "Returns the server's liveliness response."
    assert ping["group"] == "connection"
    (message,) = ping["arguments"]
    assert message["name"] == "message"
    assert message["flags"] == ["optional"]


@pytest.mark.integration
def test_docs_skips_an_unknown_name(session):
    assert session.command.docs("nosuch") == {}
    assert list(session.command.docs("ping", "nosuch")) == ["ping"]


@pytest.mark.integration
def test_docs_without_a_name_returns_every_command(session):
    names = [name for name in session.command.list() if "|" not in name]
    assert list(session.command.docs()) == names


@pytest.mark.integration
def test_list(session):
    assert "ping" in session.command.list()


@pytest.mark.integration
def test_list_filtered(session):
    names = session.command.list(pattern="bucket.*")
    assert "bucket.query" in names
    assert all(name.startswith("bucket.") for name in names)
    assert "ping" in session.command.list(category="connection")
    assert session.command.list(category="nosuch") == []
    # There are no modules, so any module name gives nothing.
    assert session.command.list(module="x") == []


@pytest.mark.integration
def test_list_with_two_filters_is_a_syntax_error(session):
    with pytest.raises(ResponseError, match="syntax error"):
        session.command.list(module="a", pattern="b")


@pytest.mark.integration
def test_getkeys(session):
    assert session.command.getkeys("zset", "foo", "bar") == ["foo"]


@pytest.mark.integration
def test_getkeysandflags(session):
    assert session.command.getkeysandflags("zget", "foo") == [("foo", ["RO", "access"])]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("nosuch", "x"), "Invalid command specified"),
        (("ping",), "The command has no key arguments"),
        (("zset",), "Invalid number of arguments specified for command"),
    ],
)
def test_getkeys_errors(session, args, message):
    with pytest.raises(ResponseError, match=message):
        session.command.getkeys(*args)


@pytest.mark.integration
def test_help(session):
    assert session.command.help()[0].startswith("COMMAND <subcommand>")


@pytest.mark.integration
def test_an_unknown_subcommand(session):
    with pytest.raises(ResponseError, match="unknown subcommand"):
        session.execute_command("COMMAND", "BOGUS")


@pytest.mark.integration
def test_resp2_gives_the_same_dicts(kronotop_url, session):
    """RESP2 sends flat arrays for the sets and the maps, decoded the same way."""
    other = Kronotop.from_url(kronotop_url, protocol=2)
    try:
        with other.session() as resp2:
            assert resp2.command.info("zset") == session.command.info("zset")
            assert resp2.command.docs("zset") == session.command.docs("zset")
    finally:
        other.close()
