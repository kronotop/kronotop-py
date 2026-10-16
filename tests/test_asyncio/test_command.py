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

"""``session.command`` on the asyncio session.

The parsers are shared with the sync client and are covered in
``tests/test_command.py``.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop.asyncio import Kronotop
from kronotop.commands.command import AsyncCommand


def test_the_group_is_on_the_session():
    session = Kronotop().session()
    assert isinstance(session.command, AsyncCommand)
    assert session.command is session.command


def test_the_group_is_not_on_the_client():
    assert not hasattr(Kronotop(), "command")


@pytest.mark.integration
async def test_info(session):
    reply = await session.command.info("ping")
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
async def test_info_matches_without_regard_to_case(session):
    assert list(await session.command.info("PING")) == ["ping"]


@pytest.mark.integration
async def test_info_with_key_specifications(session):
    (spec,) = (await session.command.info("zset"))["zset"]["key_specifications"]
    assert "OW" in spec["flags"]
    assert spec["begin_search"] == {"type": "index", "spec": {"index": 1}}
    assert spec["find_keys"]["type"] == "range"
    assert spec["find_keys"]["spec"]["keystep"] == 1


@pytest.mark.integration
async def test_info_with_an_unknown_name_maps_it_to_none(session):
    reply = await session.command.info("ping", "nosuch")
    assert list(reply) == ["ping", "nosuch"]
    assert reply["nosuch"] is None
    assert reply["ping"]["name"] == "ping"


@pytest.mark.integration
async def test_info_without_a_name_returns_every_command(session):
    reply = await session.command.info()
    assert len(reply) == await session.command.count()
    assert list(await session.execute_command("COMMAND")) == list(reply)


@pytest.mark.integration
async def test_count(session):
    assert await session.command.count() > 0


@pytest.mark.integration
async def test_docs(session):
    reply = await session.command.docs("ping")
    assert list(reply) == ["ping"]
    ping = reply["ping"]
    assert ping["summary"] == "Returns the server's liveliness response."
    assert ping["group"] == "connection"
    (message,) = ping["arguments"]
    assert message["name"] == "message"
    assert message["flags"] == ["optional"]


@pytest.mark.integration
async def test_docs_skips_an_unknown_name(session):
    assert await session.command.docs("nosuch") == {}
    assert list(await session.command.docs("ping", "nosuch")) == ["ping"]


@pytest.mark.integration
async def test_docs_without_a_name_returns_every_command(session):
    names = [name for name in await session.command.list() if "|" not in name]
    assert list(await session.command.docs()) == names


@pytest.mark.integration
async def test_list(session):
    assert "ping" in await session.command.list()


@pytest.mark.integration
async def test_list_filtered(session):
    names = await session.command.list(pattern="bucket.*")
    assert "bucket.query" in names
    assert all(name.startswith("bucket.") for name in names)
    assert "ping" in await session.command.list(category="connection")
    assert await session.command.list(category="nosuch") == []
    assert await session.command.list(module="x") == []


@pytest.mark.integration
async def test_list_with_two_filters_is_a_syntax_error(session):
    with pytest.raises(ResponseError, match="syntax error"):
        await session.command.list(module="a", pattern="b")


@pytest.mark.integration
async def test_getkeys(session):
    assert await session.command.getkeys("zset", "foo", "bar") == ["foo"]


@pytest.mark.integration
async def test_getkeysandflags(session):
    assert await session.command.getkeysandflags("zget", "foo") == [
        ("foo", ["RO", "access"])
    ]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("nosuch", "x"), "Invalid command specified"),
        (("ping",), "The command has no key arguments"),
        (("zset",), "Invalid number of arguments specified for command"),
    ],
)
async def test_getkeys_errors(session, args, message):
    with pytest.raises(ResponseError, match=message):
        await session.command.getkeys(*args)


@pytest.mark.integration
async def test_help(session):
    assert (await session.command.help())[0].startswith("COMMAND <subcommand>")


@pytest.mark.integration
async def test_an_unknown_subcommand(session):
    with pytest.raises(ResponseError, match="unknown subcommand"):
        await session.execute_command("COMMAND", "BOGUS")


@pytest.mark.integration
async def test_resp2_gives_the_same_dicts(kronotop_url, session):
    other = Kronotop.from_url(kronotop_url, protocol=2)
    try:
        async with other.session() as resp2:
            info = await session.command.info("zset")
            assert await resp2.command.info("zset") == info
            docs = await session.command.docs("zset")
            assert await resp2.command.docs("zset") == docs
    finally:
        await other.aclose()
