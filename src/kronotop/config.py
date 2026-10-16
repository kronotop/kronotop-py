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

"""Kronotop-level client settings.

Transport settings are keyword arguments on the client, forwarded unchanged.
This holds what Kronotop adds: the session attributes and the namespace. A
session sends them on its first command, and ``None`` sends nothing.

Imports value types from :mod:`kronotop.commands`. Nothing under ``commands/``
may import it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from kronotop.commands.session import InputType, ObjectIdFormat, ReplyType
from kronotop.exceptions import ConfigurationError

__all__ = ["KronotopConfig"]

#: Field name -> the values the server accepts. ``Literal`` is checked by mypy
#: only, so the same fields are checked again at runtime.
_VALID: Final[dict[str, frozenset[str]]] = {
    "input_type": frozenset({"json", "bson"}),
    "reply_type": frozenset({"json", "bson"}),
    "object_id_format": frozenset({"bytes", "hex"}),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class KronotopConfig:
    """What a session sends before its first command.

    Every field is optional. ``None`` keeps the server default. Each set field
    costs one round trip on the first command.

    Values are lowercase, the form ``SESSION.ATTRIBUTE LIST`` echoes, so a
    comparison against a listed value always matches.

    Frozen, because a client and every session opened on it share one instance.
    Derive a new one with ``dataclasses.replace``.
    """

    #: Dotted path, e.g. ``app.prod``, sent as ``NAMESPACE USE``. It has to
    #: exist, Kronotop does not create a namespace on use.
    namespace: str | None = None
    #: Encoding of the documents the session sends.
    input_type: InputType | None = None
    #: Encoding of the query results the session receives.
    reply_type: ReplyType | None = None
    #: Documents per query when the query names no ``BATCH``. One or more.
    batch: int | None = None
    #: How ObjectIds come back from DELETE and UPDATE.
    object_id_format: ObjectIdFormat | None = None

    def __post_init__(self) -> None:
        """Reject what the server would reject, before any command goes out."""
        for name, allowed in _VALID.items():
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise ConfigurationError(
                    f"{name} must be one of {sorted(allowed)}, got {value!r}"
                )
        if self.namespace is not None:
            if not self.namespace.strip():
                raise ConfigurationError("namespace must not be empty")
            if self.namespace != self.namespace.strip():
                # ``to_commands`` sends the value as it is, and the server looks
                # the name up as it is, so the padding would reach it.
                raise ConfigurationError(
                    "namespace must not have leading or trailing spaces"
                )
        if self.batch is not None:
            # bool is a subclass of int, and batch=True would go out as 1.
            if isinstance(self.batch, bool) or not isinstance(self.batch, int):
                raise ConfigurationError(f"batch must be an int, got {self.batch!r}")
            if self.batch < 1:
                raise ConfigurationError(f"batch must be 1 or more, got {self.batch}")

    def to_commands(self, namespace: str | None) -> list[tuple[str | int, ...]]:
        """The commands that put a session into this configuration.

        ``namespace`` is the one the session was opened with and wins over
        :attr:`namespace`. ``NAMESPACE USE`` goes first because it is the only
        one that can fail on a healthy server, so a missing namespace costs one
        round trip instead of five.
        """
        commands: list[tuple[str | int, ...]] = []
        _namespace: str | None = namespace if namespace else self.namespace
        if _namespace is not None:
            commands.append(("NAMESPACE", "USE", _namespace))
        if self.input_type is not None:
            commands.append(("SESSION.ATTRIBUTE", "SET", "input_type", self.input_type))
        if self.reply_type is not None:
            commands.append(("SESSION.ATTRIBUTE", "SET", "reply_type", self.reply_type))
        if self.batch is not None:
            commands.append(("SESSION.ATTRIBUTE", "SET", "batch", self.batch))
        if self.object_id_format is not None:
            commands.append(
                ("SESSION.ATTRIBUTE", "SET", "object_id_format", self.object_id_format)
            )
        return commands
