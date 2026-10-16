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

"""``KronotopConfig`` on the async client, mirroring ``tests/test_config.py``.

The dataclass itself is shared, so its own tests are not repeated here.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from kronotop import KronotopConfig, KronotopConnectionError, NoSuchNamespaceError
from kronotop.asyncio import Kronotop


async def test_a_session_inherits_the_client_config():
    client = Kronotop(config=KronotopConfig(batch=10))
    session = client.session()
    assert session._config is client.config
    assert session._config_pending is True


async def test_a_session_config_replaces_the_client_one():
    other = KronotopConfig(batch=20)
    client = Kronotop(config=KronotopConfig(batch=10))
    assert client.session(config=other)._config is other


async def test_a_session_without_a_config_still_reads_the_attributes():
    """No ``SET`` goes out, but the ``LIST`` does. ``insert`` needs it."""
    assert Kronotop().session()._config_pending is True


async def test_a_session_namespace_falls_back_to_the_config_one():
    client = Kronotop(config=KronotopConfig(namespace="app.prod"))
    assert client.session()._namespace == "app.prod"
    assert client.session(namespace="app.staging")._namespace == "app.staging"


@pytest.mark.integration
async def test_the_config_is_applied_on_the_first_command(kr):
    async with kr.session(config=KronotopConfig(batch=7, reply_type="json")) as s:
        attributes = await s.attribute_list()
        assert attributes[b"batch"] == 7
        assert attributes[b"reply_type"] == b"json"
        assert attributes[b"input_type"] == b"bson"


@pytest.mark.integration
async def test_the_session_namespace_is_applied_on_the_first_command(kr, namespace):
    """``session(namespace=...)`` goes out as ``NAMESPACE USE``."""
    async with kr.session(namespace=namespace) as s:
        assert await s.namespace.current() == namespace


@pytest.mark.integration
async def test_the_session_namespace_beats_the_config_one_on_the_server(kr, namespaces):
    picked = await namespaces()
    config = KronotopConfig(namespace=await namespaces())
    async with kr.session(config=config, namespace=picked) as s:
        assert await s.namespace.current() == picked


@pytest.mark.integration
async def test_a_missing_session_namespace_keeps_raising(kr):
    """The command the caller asked for must not run in the wrong namespace."""
    async with kr.session(namespace="no.such.namespace") as s:
        for _ in range(2):
            with pytest.raises(NoSuchNamespaceError):
                await s.ping()
            assert s._config_pending is True


@pytest.mark.integration
async def test_a_plain_session_gets_the_server_defaults(session):
    assert await session.attribute_list() == {
        b"reply_type": b"bson",
        b"input_type": b"bson",
        b"batch": 100,
        b"object_id_format": b"bytes",
    }


@pytest.mark.integration
async def test_the_stored_attributes_are_normalised(kr):
    async with kr.session(config=KronotopConfig(batch=7, reply_type="json")) as s:
        await s.ping()
        assert await s.attributes == {
            "input_type": "bson",
            "reply_type": "json",
            "batch": 7,
            "object_id_format": "bytes",
        }


@pytest.mark.integration
async def test_the_config_goes_out_once(kr):
    async with kr.session(config=KronotopConfig(batch=7)) as s:
        sent = []
        wrapped = s._redis.execute_command

        async def spy(*args, **options):
            sent.append(args)
            return await wrapped(*args, **options)

        s._redis.execute_command = spy
        await s.ping()
        await s.ping()
        assert sent == [
            ("SESSION.ATTRIBUTE", "SET", "batch", 7),
            ("SESSION.ATTRIBUTE", "LIST"),
            ("PING",),
            ("PING",),
        ]


@pytest.mark.integration
async def test_a_session_without_a_config_sends_only_the_list(kr):
    """No ``SET``, one ``LIST``, and the attributes are filled from it."""
    async with kr.session() as s:
        sent = []
        wrapped = s._redis.execute_command

        async def spy(*args, **options):
            sent.append(args)
            return await wrapped(*args, **options)

        s._redis.execute_command = spy
        await s.ping()
        await s.ping()
        assert sent == [
            ("SESSION.ATTRIBUTE", "LIST"),
            ("PING",),
            ("PING",),
        ]
        attributes = await s.attributes
        assert attributes["input_type"] in {"bson", "json"}
        assert attributes["object_id_format"] in {"bytes", "hex"}


@pytest.mark.integration
async def test_a_closed_session_has_no_attributes(kr):
    async with kr.session() as s:
        await s.ping()
    with pytest.raises(KronotopConnectionError, match="session is closed"):
        await s.attributes


@pytest.mark.integration
async def test_a_missing_namespace_keeps_raising(kr):
    config = KronotopConfig(namespace="no.such.namespace")
    async with kr.session(config=config) as s:
        for _ in range(2):
            with pytest.raises(NoSuchNamespaceError):
                await s.ping()
            assert s._config_pending is True


@pytest.mark.integration
async def test_attribute_set_and_list(session):
    assert await session.attribute_set("batch", 42) is True
    assert (await session.attribute_list())[b"batch"] == 42


@pytest.mark.integration
async def test_a_set_updates_the_stored_attributes(session):
    await session.attribute_set("batch", "50")
    assert (await session.attributes)["batch"] == 50
    await session.execute_command("SESSION.ATTRIBUTE", "SET", "reply_type", "JSON")
    assert (await session.attributes)["reply_type"] == "json"


@pytest.mark.integration
async def test_a_rejected_set_leaves_the_stored_attributes_alone(kr):
    async with kr.session(config=KronotopConfig(reply_type="bson")) as s:
        with pytest.raises(ResponseError):
            await s.attribute_set("reply_type", "xml")
        assert (await s.attributes)["reply_type"] == "bson"


@pytest.mark.integration
async def test_a_closed_session_leaves_an_authenticated_connection(
    kronotop_url, kronotop_password
):
    client = Kronotop.from_url(kronotop_url, password=kronotop_password)
    try:
        async with client.session() as first:
            await first.ping()
        async with client.session() as second:
            assert await second.ping() is True
    finally:
        await client.aclose()
