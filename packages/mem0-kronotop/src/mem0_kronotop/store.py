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

"""The mem0 vector store on a Kronotop bucket.

One memory is one document::

    {"id": "<mem0 id>", "embedding": [...], "payload": {...}}

The mem0 id is any string, and ``_id`` has to be an ObjectId, so the id has a
field of its own, with a unique index. ``_id`` comes from the server.

mem0 calls one store from several threads. Every method borrows a session from
the pool and names the namespace on the command, so a call sends its own
command and leaves nothing on the session.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from mem0.vector_stores.base import (  # type: ignore[import-untyped]
    VectorStoreBase,
)
from pydantic import BaseModel

from kronotop import (
    BarrierNotSatisfiedError,
    Kronotop,
    Session,
    SingleFieldIndex,
    VectorIndex,
)
from mem0_kronotop._filters import to_bql
from mem0_kronotop._pool import SessionPool

__all__ = ["KronotopVectorStore", "OutputData"]

#: Field that holds the vector.
EMBEDDING = "embedding"

#: The vector is large and mem0 never reads it back.
NO_EMBEDDING = {EMBEDDING: 0}

#: Named here because inside the class ``list`` is the method.
_Names = list[str]


class OutputData(BaseModel):
    """One memory, the way mem0 reads it."""

    id: str | None
    score: float | None
    payload: dict[str, Any] | None


def _output(document: dict[str, Any], score: float | None = None) -> OutputData:
    return OutputData(
        id=document.get("id"), score=score, payload=document.get("payload")
    )


class KronotopVectorStore(VectorStoreBase):  # type: ignore[misc]
    """A bucket of memories, in one namespace.

    ``bootstrap`` and ``kwargs`` are the arguments of :class:`kronotop.Kronotop`.
    """

    def __init__(
        self,
        namespace: str,
        collection_name: str,
        embedding_model_dims: int,
        bootstrap: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.namespace = namespace
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        #: mem0 closes the store through this name.
        self.client = Kronotop(bootstrap=bootstrap, **kwargs)
        self.create_col()

    def _open_session(self) -> Session:
        return self.client.session_for(buckets={self.namespace: [self.collection_name]})

    def create_col(self) -> None:
        """Create the namespace and the bucket when they are missing.

        The indexes come with ``BUCKET.CREATE``, so they are ready at once. An
        existing bucket is left as it is.
        """
        # Not from the pool, because a pooled session opens on the member that
        # holds the bucket, and the bucket may not exist yet.
        with self.client.session() as session:
            session.namespace.create(self.namespace, if_not_exists=True)
            session.bucket.create(
                self.collection_name,
                indexes=[
                    SingleFieldIndex("id", "string", unique=True),
                    VectorIndex(EMBEDDING, self.embedding_model_dims, "cosine"),
                ],
                if_not_exists=True,
                namespace=self.namespace,
            )
        # A closed pool does not open again, so ``reset`` needs a new one.
        self._pool = SessionPool(self._open_session)

    def insert(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        payloads = payloads or [{} for _ in vectors]
        ids = ids or [str(uuid.uuid4()) for _ in vectors]
        documents = [
            {"id": id_, EMBEDDING: vector, "payload": payload}
            for id_, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        with self._pool.borrow() as session:
            session.bucket.insert(
                self.collection_name, *documents, namespace=self.namespace
            )

    def search(
        self,
        query: str,
        vectors: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[OutputData]:
        # TODO: the server applies the filter after the similarity ranking and
        # stops at MAX-SCAN-CANDIDATES. In a bucket shared by many users a
        # search for one user can return fewer than ``top_k`` results.
        with self._pool.borrow() as session:
            results = session.bucket.vector(
                self.collection_name,
                EMBEDDING,
                vectors,
                filter_=to_bql(filters) or None,
                projection=NO_EMBEDDING,
                top=top_k,
                namespace=self.namespace,
            )
        return [_output(result.entry, result.score) for result in results]

    def get(self, vector_id: str) -> OutputData | None:
        with (
            self._pool.borrow() as session,
            session.bucket.query(
                self.collection_name,
                {"id": vector_id},
                projection=NO_EMBEDDING,
                limit=1,
                namespace=self.namespace,
            ) as cursor,
        ):
            return _output(cursor.batch[0]) if cursor.batch else None

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        fields: dict[str, Any] = {}
        if vector is not None:
            fields[EMBEDDING] = vector
        if payload is not None:
            fields["payload"] = payload
        if not fields:
            return
        with (
            self._pool.borrow() as session,
            session.bucket.update(
                self.collection_name,
                {"id": vector_id},
                {"$set": fields},
                limit=1,
                namespace=self.namespace,
            ),
        ):
            pass

    def delete(self, vector_id: str) -> None:
        with (
            self._pool.borrow() as session,
            session.bucket.delete(
                self.collection_name,
                {"id": vector_id},
                limit=1,
                namespace=self.namespace,
            ),
        ):
            pass

    def list(
        self, filters: dict[str, Any] | None = None, top_k: int = 100
    ) -> list[OutputData]:
        with (
            self._pool.borrow() as session,
            session.bucket.query(
                self.collection_name,
                to_bql(filters),
                projection=NO_EMBEDDING,
                # One batch holds the whole answer, so nothing is advanced.
                batch=top_k,
                limit=top_k,
                namespace=self.namespace,
            ) as cursor,
        ):
            return [_output(document) for document in cursor.batch]

    def list_cols(self) -> _Names:
        with self._pool.borrow() as session:
            names: _Names = session.bucket.list(namespace=self.namespace)
        return names

    def col_info(self) -> dict[str, Any]:
        with self._pool.borrow() as session:
            indexes = session.bucket.list_indexes(
                self.collection_name, namespace=self.namespace
            )
        return {
            "name": self.collection_name,
            "namespace": self.namespace,
            "indexes": indexes,
        }

    def delete_col(self) -> None:
        """Remove the bucket and purge it, so the name is free again."""
        self._pool.close()
        with self.client.session() as session:
            session.bucket.remove(self.collection_name, namespace=self.namespace)
            try:
                session.bucket.purge(self.collection_name, namespace=self.namespace)
            except BarrierNotSatisfiedError:
                # The failed attempt publishes the removal again, so the
                # second one finds every shard ready.
                session.bucket.purge(self.collection_name, namespace=self.namespace)

    def reset(self) -> None:
        self.delete_col()
        self.create_col()
