# kronotop-py

[![CI](https://github.com/kronotop/kronotop-py/actions/workflows/ci.yml/badge.svg)](https://github.com/kronotop/kronotop-py/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official Python client for [Kronotop](https://github.com/kronotop/kronotop).

Kronotop speaks [RESP2 and RESP3](https://redis.io/docs/latest/develop/reference/protocol-spec/) on the wire. The client is built on [redis-py](https://github.com/redis/redis-py).

## Install

```bash
pip install kronotop-py
# or, with the C parser:
pip install "kronotop-py[hiredis]"
```

## Usage

See Kronotop's [Quickstart]( https://github.com/kronotop/kronotop#quickstart) document first.

```python
from kronotop import Kronotop

client = Kronotop()  # localhost:5484
with client.session() as session:
    session.ping()
```

One TCP connection is one session, and the session holds the open transaction.
The client keeps a pool, so a session pins one connection until it is closed.
Every command is on the session.

```python
with client.session() as session:
    session.zmap.set("greeting", "Hello")  # commits on its own
    session.zmap.get("greeting")  # b'Hello'

    with session.transaction(returning="committed-version") as tx:
        session.zmap.set("a", "1")
        session.zmap.set("b", "2")
    # COMMIT on a clean exit, ROLLBACK when the block raises
    print(tx.committed_version)
```

`KronotopConfig` carries the session attributes and the namespace. Every
session opened on the client starts from it.

```python
from kronotop import KronotopConfig

client = Kronotop(config=KronotopConfig(namespace="app", batch=500))
```

```python
from kronotop.asyncio import Kronotop

client = Kronotop()

async with client.session() as session:
    async with session.transaction():
        ...

await client.aclose()
```

## Examples

The [examples](examples) folder has small programs that show the client in use.

## Packages

This repository is a monorepo. Each directory under `packages/` ships as its own distribution.

- [mem0-kronotop](packages/mem0-kronotop)

## Development

```bash
uv sync --all-extras --all-packages
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Tests

```bash
uv run pytest                        # integration tests skip without a server
uv run pytest -m "not integration"   # never touches the network
uv run pytest -m integration         # only the tests that need a server
```

Integration tests connect to `kronotop://localhost:5484`, and the `Admin`
tests to the admin port of that member, `localhost:3320`. Use
`--kronotop-url` and `--kronotop-admin-host` to point them somewhere else.
The tests that need a password are skipped unless `--kronotop-password`
names the `auth.requirepass` of the server.

```bash
uv run pytest -m integration --kronotop-url kronotop://192.168.1.10:5484
uv run pytest -m integration --kronotop-admin-host 192.168.1.10:3320
uv run pytest -m integration --kronotop-password devpass
```

Start a Kronotop first. `docker/docker-compose.yaml` brings up FoundationDB and one Kronotop node:

```bash
docker compose -f docker/docker-compose.yaml up
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
