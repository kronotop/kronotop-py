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

"""The session pool, against fake sessions instead of a server."""

import pytest

from kronotop import KronotopConnectionError, RejectError
from mem0_kronotop._pool import SessionPool


class FakeSession:
    """The three members of a session the pool reads."""

    def __init__(self):
        self.broken = False
        self.in_transaction = False
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def opened():
    """Every session the pool opened, in order."""
    return []


@pytest.fixture
def pool(opened):
    def open_session():
        opened.append(FakeSession())
        return opened[-1]

    return SessionPool(open_session)


def test_a_session_is_used_again(pool, opened):
    with pool.borrow() as first:
        pass
    with pool.borrow() as second:
        pass
    assert second is first
    assert len(opened) == 1
    assert first.closed is False


def test_two_borrows_at_once_get_two_sessions(pool, opened):
    with pool.borrow() as first, pool.borrow() as second:
        assert second is not first
    assert len(opened) == 2


@pytest.mark.parametrize("flag", ["broken", "in_transaction"])
def test_a_session_with_state_is_closed(pool, opened, flag):
    with pool.borrow() as first:
        setattr(first, flag, True)
    assert first.closed is True
    with pool.borrow() as second:
        pass
    assert second is not first


def test_a_rejected_session_is_closed(pool):
    with pytest.raises(RejectError), pool.borrow() as first:
        raise RejectError("REJECT 1 localhost:5484")
    assert first.closed is True
    with pool.borrow() as second:
        pass
    assert second is not first


def test_another_error_keeps_the_session(pool):
    with pytest.raises(ValueError, match="boom"), pool.borrow() as first:
        raise ValueError("boom")
    assert first.closed is False
    with pool.borrow() as second:
        pass
    assert second is first


def test_close_closes_the_idle_sessions(pool):
    with pool.borrow() as first:
        pass
    pool.close()
    pool.close()
    assert first.closed is True
    with pytest.raises(KronotopConnectionError), pool.borrow():
        pass


def test_a_session_that_is_out_closes_on_return(pool):
    with pool.borrow() as first:
        pool.close()
        assert first.closed is False
    assert first.closed is True
