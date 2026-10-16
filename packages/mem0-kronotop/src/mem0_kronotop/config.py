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

"""The ``config`` of a mem0 ``vector_store`` whose provider is ``kronotop``.

mem0 reads ``collection_name`` and ``embedding_model_dims`` by name, so the two
are spelled the mem0 way.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["KronotopVectorStoreConfig"]


class KronotopVectorStoreConfig(BaseModel):
    """The keyword arguments of :class:`kronotop.Kronotop`, and where the
    memories live."""

    model_config = ConfigDict(extra="allow")

    #: The members the client starts from, each ``host`` or ``host:port``.
    #: Without it the client starts from ``localhost:5484``.
    bootstrap: list[str] | None = None
    #: Namespace of the bucket. Created when it is missing.
    namespace: str = "mem0"
    #: Name of the bucket. Created when it is missing.
    collection_name: str = "mem0"
    #: Length of each vector. Fixed once the bucket exists.
    embedding_model_dims: int = 1536
