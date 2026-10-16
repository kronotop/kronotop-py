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

"""``KronotopConfig``, and what a session sends before its first command."""

from __future__ import annotations

import dataclasses

import pytest
from redis.exceptions import ResponseError

from kronotop import (
    ConfigurationError,
    Kronotop,
    KronotopConfig,
    KronotopConnectionError,
    NoSuchNamespaceError,
)


def test_an_empty_config_sends_nothing():
    assert KronotopConfig().to_commands(namespace=None) == []


def test_namespace_goes_first():
    """A namespace that does not exist should cost one round trip, not five."""
    config = KronotopConfig(
        namespace="app.prod",
        input_type="json",
        reply_type="json",
        batch=10,
        object_id_format="hex",
    )
    assert config.to_commands(namespace=None) == [
        ("NAMESPACE", "USE", "app.prod"),
        ("SESSION.ATTRIBUTE", "SET", "input_type", "json"),
        ("SESSION.ATTRIBUTE", "SET", "reply_type", "json"),
        ("SESSION.ATTRIBUTE", "SET", "batch", 10),
        ("SESSION.ATTRIBUTE", "SET", "object_id_format", "hex"),
    ]


def test_the_session_namespace_wins_over_the_config_one():
    config = KronotopConfig(namespace="app.prod")
    assert config.to_commands(namespace="app.staging") == [
        ("NAMESPACE", "USE", "app.staging"),
    ]


@pytest.mark.parametrize("batch", [0, -1, True, "5", 1.5])
def test_a_bad_batch_is_refused(batch):
    with pytest.raises(ConfigurationError, match="batch"):
        KronotopConfig(batch=batch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_type", "yaml"),
        ("reply_type", "JSON"),  # the server lowercases, LIST does not
        ("object_id_format", "raw"),
    ],
)
def test_a_bad_value_is_refused(field, value):
    with pytest.raises(ConfigurationError, match=field):
        KronotopConfig(**{field: value})


@pytest.mark.parametrize("namespace", ["", "   "])
def test_a_blank_namespace_is_refused(namespace):
    with pytest.raises(ConfigurationError, match="namespace"):
        KronotopConfig(namespace=namespace)


@pytest.mark.parametrize("namespace", [" app.prod", "app.prod ", " app.prod "])
def test_a_padded_namespace_is_refused(namespace):
    """``to_commands`` sends the value as it is, so the padding would go out."""
    with pytest.raises(ConfigurationError, match="spaces"):
        KronotopConfig(namespace=namespace)


def test_a_configuration_error_is_also_a_value_error():
    with pytest.raises(ValueError):
        KronotopConfig(batch=0)


def test_the_config_is_frozen():
    """A client and every session on it share one instance."""
    config = KronotopConfig(batch=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.batch = 20


def test_replace_derives_a_new_one():
    base = KronotopConfig(namespace="app.prod", reply_type="json")
    derived = dataclasses.replace(base, namespace="app.staging")
    assert derived.namespace == "app.staging"
    assert derived.reply_type == "json"
    assert base.namespace == "app.prod"


def test_a_session_inherits_the_client_config():
    client = Kronotop(config=KronotopConfig(batch=10))
    session = client.session()
    assert session._config is client.config
    assert session._config_pending is True


def test_a_session_config_replaces_the_client_one():
    other = KronotopConfig(batch=20)
    client = Kronotop(config=KronotopConfig(batch=10))
    assert client.session(config=other)._config is other


def test_an_empty_session_config_overrides_the_client_one():
    """An empty config still counts as a config. It means send nothing."""
    client = Kronotop(config=KronotopConfig(batch=10))
    assert (
        client.session(config=KronotopConfig())._config.to_commands(namespace=None)
        == []
    )


def test_a_session_without_a_config_still_reads_the_attributes():
    """No ``SET`` goes out, but the ``LIST`` does. ``insert`` needs it."""
    assert Kronotop().session()._config_pending is True


def test_a_session_namespace_falls_back_to_the_config_one():
    client = Kronotop(config=KronotopConfig(namespace="app.prod"))
    assert client.session()._namespace == "app.prod"
    assert client.session(namespace="app.staging")._namespace == "app.staging"


@pytest.mark.integration
def test_the_config_is_applied_on_the_first_command(kr):
    with kr.session(config=KronotopConfig(batch=7, reply_type="json")) as s:
        attributes = s.attribute_list()
        assert attributes[b"batch"] == 7
        assert attributes[b"reply_type"] == b"json"
        assert attributes[b"input_type"] == b"bson"


@pytest.mark.integration
def test_the_session_namespace_is_applied_on_the_first_command(kr, namespace):
    """``session(namespace=...)`` goes out as ``NAMESPACE USE``."""
    with kr.session(namespace=namespace) as s:
        assert s.namespace.current() == namespace


@pytest.mark.integration
def test_the_session_namespace_beats_the_config_one_on_the_server(kr, namespaces):
    picked = namespaces()
    config = KronotopConfig(namespace=namespaces())
    with kr.session(config=config, namespace=picked) as s:
        assert s.namespace.current() == picked


@pytest.mark.integration
def test_a_missing_session_namespace_keeps_raising(kr):
    """The command the caller asked for must not run in the wrong namespace."""
    with kr.session(namespace="no.such.namespace") as s:
        for _ in range(2):
            with pytest.raises(NoSuchNamespaceError):
                s.ping()
            assert s._config_pending is True


@pytest.mark.integration
def test_a_plain_session_gets_the_server_defaults(session):
    assert session.attribute_list() == {
        b"reply_type": b"bson",
        b"input_type": b"bson",
        b"batch": 100,
        b"object_id_format": b"bytes",
    }


@pytest.mark.integration
def test_the_stored_attributes_are_normalised(kr):
    """``batch`` is an int, the rest are ``str``, whatever the client decodes."""
    with kr.session(config=KronotopConfig(batch=7, reply_type="json")) as s:
        s.ping()
        assert s.attributes == {
            "input_type": "bson",
            "reply_type": "json",
            "batch": 7,
            "object_id_format": "bytes",
        }


@pytest.mark.integration
def test_the_config_goes_out_once(kr):
    with kr.session(config=KronotopConfig(batch=7)) as s:
        sent = []
        wrapped = s._redis.execute_command

        def spy(*args, **options):
            sent.append(args)
            return wrapped(*args, **options)

        s._redis.execute_command = spy
        s.ping()
        s.ping()
        # The ``LIST`` that fills ``_attributes`` goes out once as well, after
        # the ``SET``.
        assert sent == [
            ("SESSION.ATTRIBUTE", "SET", "batch", 7),
            ("SESSION.ATTRIBUTE", "LIST"),
            ("PING",),
            ("PING",),
        ]


@pytest.mark.integration
def test_a_session_without_a_config_sends_only_the_list(kr):
    """No ``SET``, one ``LIST``, and the attributes are filled from it."""
    with kr.session() as s:
        sent = []
        wrapped = s._redis.execute_command

        def spy(*args, **options):
            sent.append(args)
            return wrapped(*args, **options)

        s._redis.execute_command = spy
        s.ping()
        s.ping()
        assert sent == [
            ("SESSION.ATTRIBUTE", "LIST"),
            ("PING",),
            ("PING",),
        ]
        assert s.attributes["input_type"] in {"bson", "json"}
        assert s.attributes["object_id_format"] in {"bytes", "hex"}


@pytest.mark.integration
def test_a_closed_session_has_no_attributes(kr):
    """Reading them is an operation, and a closed session takes none."""
    with kr.session() as s:
        s.ping()
    with pytest.raises(KronotopConnectionError, match="session is closed"):
        _ = s.attributes


@pytest.mark.integration
def test_a_missing_namespace_keeps_raising(kr):
    """The command the caller asked for must not run unconfigured."""
    with kr.session(config=KronotopConfig(namespace="no.such.namespace")) as s:
        for _ in range(2):
            with pytest.raises(NoSuchNamespaceError):
                s.ping()
            assert s._config_pending is True


@pytest.mark.integration
def test_attribute_set_and_list(session):
    assert session.attribute_set("batch", 42) is True
    assert session.attribute_list()[b"batch"] == 42


@pytest.mark.integration
def test_a_set_updates_the_stored_attributes(session):
    """``batch`` is stored as an int, whichever form went out."""
    session.attribute_set("batch", "50")
    assert session.attributes["batch"] == 50
    session.execute_command("SESSION.ATTRIBUTE", "SET", "reply_type", "JSON")
    assert session.attributes["reply_type"] == "json"


@pytest.mark.integration
def test_a_rejected_set_leaves_the_stored_attributes_alone(kr):
    with kr.session(config=KronotopConfig(reply_type="bson")) as s:
        with pytest.raises(ResponseError):
            s.attribute_set("reply_type", "xml")
        assert s.attributes["reply_type"] == "bson"


@pytest.mark.integration
def test_a_closed_session_leaves_an_authenticated_connection(
    kronotop_url, kronotop_password
):
    """``SESSION.CLOSE`` keeps the server side auth flag.

    The flag belongs to the connection, so the pool can hand it to the next
    session as it is. No ``AUTH`` goes out on release.
    """
    client = Kronotop.from_url(kronotop_url, password=kronotop_password)
    try:
        with client.session() as first:
            first.ping()
        with client.session() as second:
            assert second.ping() is True
    finally:
        client.close()
