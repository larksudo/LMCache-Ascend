# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Regression checks for the PD small-prompt hang.

Multi-group KV cache (DSv4) forces ``discard_partial_chunks=True``, which
truncates the save window to a whole number of chunks. A prompt shorter than
``chunk_size`` therefore truncates to zero tokens, the prefiller transfers no
chunk, and the notification that normally rides along with that transfer is
never sent. The proxy's ``wait_decode_kv_ready`` counts one notification per
rank, so the request hangs.

These tests pin the contract that makes that impossible: a last prefill always
reports to the proxy, exactly once, in the wire format the proxy decodes.
"""

# Standard
from collections import OrderedDict
from types import SimpleNamespace
import threading

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.v1.storage_backend.pd_backend import PDMsg, ProxyNotif
import msgspec
import pytest
import zmq

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.storage_backend.pd.sender_mixin import AscendPDSenderMixin


class _Sender(AscendPDSenderMixin):
    """Minimal sender carrying only the state ``notify_prefill_done`` touches."""

    def __init__(self, socket):
        self.proxy_side_channel = socket
        self._prefill_done_notified: "OrderedDict[str, None]" = OrderedDict()
        self._prefill_done_lock = threading.Lock()


@pytest.fixture
def zmq_pair():
    """A real PUSH/PULL pair standing in for prefiller -> proxy."""
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://127.0.0.1:{port}")
    try:
        yield push, pull
    finally:
        push.close(linger=0)
        pull.close(linger=0)
        ctx.term()


def test_notification_decodes_as_proxy_notif(zmq_pair):
    """The proxy must be able to decode what the sender puts on the wire.

    The proxy does ``msgspec.msgpack.decode(msg, type=PDMsg)`` and ignores
    anything that is not a ProxyNotif, so a shape mismatch here would look
    exactly like the hang this fix targets.
    """
    push, pull = zmq_pair
    _Sender(push).notify_prefill_done("req_42")

    pull.setsockopt(zmq.RCVTIMEO, 5000)
    msg = msgspec.msgpack.decode(pull.recv(), type=PDMsg)

    assert isinstance(msg, ProxyNotif)
    assert msg.req_id == "req_42"


def test_notification_sent_exactly_once_per_request(zmq_pair):
    """Transfer path and save-loop fallback may both fire; the proxy sees one.

    A second notification would leave a stale counter in the proxy's
    ``finished_reqs`` for a request it has already popped.
    """
    push, pull = zmq_pair
    sender = _Sender(push)
    sender.notify_prefill_done("req_1")
    sender.notify_prefill_done("req_1")
    sender.notify_prefill_done("req_2")

    pull.setsockopt(zmq.RCVTIMEO, 5000)
    seen = [msgspec.msgpack.decode(pull.recv(), type=PDMsg).req_id for _ in range(2)]

    assert seen == ["req_1", "req_2"]
    pull.setsockopt(zmq.RCVTIMEO, 300)
    with pytest.raises(zmq.Again):
        pull.recv()


def _engine_stub(passive: bool, backend):
    return SimpleNamespace(
        _is_passive=lambda: passive,
        storage_manager=SimpleNamespace(storage_backends={"PDBackend": backend}),
    )


def _spec(is_last_prefill=True, req_id="req_7"):
    return SimpleNamespace(is_last_prefill=is_last_prefill, req_id=req_id)


def test_engine_notifies_when_no_chunk_was_stored():
    """The zero-chunk store path is exactly where the old code went silent."""
    notified = []
    engine = _engine_stub(
        passive=False, backend=SimpleNamespace(notify_prefill_done=notified.append)
    )

    AscendLMCacheEngine.notify_pd_prefill_done(engine, _spec())

    assert notified == ["req_7"]


def test_engine_silent_on_passive_rank():
    """save_only_first_rank: the proxy expects one signal, not one per rank."""
    notified = []
    engine = _engine_stub(
        passive=True, backend=SimpleNamespace(notify_prefill_done=notified.append)
    )

    AscendLMCacheEngine.notify_pd_prefill_done(engine, _spec())

    assert notified == []


@pytest.mark.parametrize(
    "spec",
    [None, _spec(is_last_prefill=False)],
    ids=["no_disagg_spec", "not_last_prefill"],
)
def test_engine_silent_when_proxy_is_not_waiting(spec):
    """Only a last prefill under PD has a proxy blocked on it."""
    notified = []
    engine = _engine_stub(
        passive=False, backend=SimpleNamespace(notify_prefill_done=notified.append)
    )

    AscendLMCacheEngine.notify_pd_prefill_done(engine, spec)

    assert notified == []
