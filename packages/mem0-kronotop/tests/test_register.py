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

"""The ``kronotop`` provider inside mem0, without a server."""

import pytest
from mem0.utils.factory import VectorStoreFactory
from mem0.vector_stores.configs import VectorStoreConfig

from mem0_kronotop import register
from mem0_kronotop.config import KronotopVectorStoreConfig


@pytest.fixture(autouse=True)
def registered():
    register()


def test_the_factory_knows_the_store():
    path = VectorStoreFactory.provider_to_class["kronotop"]
    assert path == "mem0_kronotop.store.KronotopVectorStore"


def test_the_config_is_built_from_a_dict():
    store = VectorStoreConfig(
        provider="kronotop",
        config={"bootstrap": ["a:5484", "b:5484"], "embedding_model_dims": 3},
    )
    assert isinstance(store.config, KronotopVectorStoreConfig)
    assert store.config.bootstrap == ["a:5484", "b:5484"]
    assert store.config.embedding_model_dims == 3
    assert store.config.namespace == "mem0"
    assert store.config.collection_name == "mem0"


def test_the_defaults_name_no_member():
    assert VectorStoreConfig(provider="kronotop").config.bootstrap is None


def test_another_field_is_kept_for_the_client():
    store = VectorStoreConfig(provider="kronotop", config={"password": "secret"})
    # mem0 builds the store from this dict.
    assert store.config.model_dump()["password"] == "secret"


def test_another_provider_still_works():
    store = VectorStoreConfig(provider="qdrant", config={"path": "/tmp/qdrant"})
    assert type(store.config).__name__ == "QdrantConfig"


def test_registering_twice_is_safe():
    register()
    assert VectorStoreConfig(provider="kronotop").provider == "kronotop"
