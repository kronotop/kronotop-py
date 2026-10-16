# mem0-kronotop

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Kronotop backend for [mem0](https://github.com/mem0ai/mem0), built on [kronotop-py](https://github.com/kronotop/kronotop-py).

## Install

```bash
pip install mem0-kronotop
```

## Usage

Call `register()` before the first `Memory`. It adds the `kronotop` provider to mem0.

```python
from mem0 import Memory

from mem0_kronotop import register

register()

memory = Memory.from_config(
    {
        "vector_store": {
            "provider": "kronotop",
            "config": {
                "bootstrap": ["localhost:5484", "localhost:5485"],
                "namespace": "mem0",
                "collection_name": "mem0",
                "embedding_model_dims": 1536,
            },
        }
    }
)
```

`bootstrap` lists the members the client starts from, and defaults to
`localhost:5484`. The other three values above are the defaults. The store
creates the namespace and the bucket when they are missing.

`config` also takes the keyword arguments of the `Kronotop` client, such as
`password` and `ssl`.

## Example

The program below is a support assistant that remembers what a customer said.
mem0 uses OpenAI for the language model and the embeddings by default, so set
`OPENAI_API_KEY` before you run it.

```python
from mem0 import Memory

from mem0_kronotop import register

register()

memory = Memory.from_config(
    {
        "vector_store": {
            "provider": "kronotop",
            "config": {
                "bootstrap": ["localhost:5484"],
                "namespace": "support",
                "collection_name": "customers",
            },
        }
    }
)

# mem0 reads the conversation and stores the facts it finds.
memory.add(
    [
        {"role": "user", "content": "My order arrived broken. I want a refund."},
        {"role": "assistant", "content": "Sorry about that. I opened a refund."},
        {"role": "user", "content": "Thanks. Please write to me by email only."},
    ],
    user_id="alice",
)

# Later, in another conversation.
found = memory.search("How should we contact her?", filters={"user_id": "alice"})
for result in found["results"]:
    print(result["score"], result["memory"])

# Everything mem0 knows about one customer.
for result in memory.get_all(filters={"user_id": "alice"})["results"]:
    print(result["id"], result["memory"])

memory.delete(found["results"][0]["id"])
```

Kronotop has no full-text search, so mem0 ranks by vector similarity alone.

## License

Apache License 2.0. See [LICENSE](LICENSE).
