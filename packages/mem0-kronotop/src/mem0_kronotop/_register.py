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

"""Adds the ``kronotop`` provider to mem0.

mem0 has no hook for a provider that lives outside its own tree. It keeps two
tables of names and imports the config class from a module of its own. So this
writes to the two tables and puts the config module under the name mem0 asks
for.
"""

from __future__ import annotations

import sys

__all__ = ["register"]

PROVIDER = "kronotop"


def register() -> None:
    """Make ``kronotop`` a ``vector_store`` provider of mem0.

    Call it before the first ``Memory``. Calling it twice is safe.
    """
    # Inside the function, so ``import mem0_kronotop`` does not load mem0.
    from mem0.utils.factory import (  # type: ignore[import-untyped]
        VectorStoreFactory,
    )
    from mem0.vector_stores.configs import (  # type: ignore[import-untyped]
        VectorStoreConfig,
    )

    from mem0_kronotop import config

    VectorStoreFactory.provider_to_class[PROVIDER] = (
        "mem0_kronotop.store.KronotopVectorStore"
    )
    # A private attribute of the model. Each instance copies this dict.
    providers = VectorStoreConfig.__private_attributes__["_provider_configs"]
    providers.default[PROVIDER] = config.KronotopVectorStoreConfig.__name__
    sys.modules[f"mem0.configs.vector_stores.{PROVIDER}"] = config
