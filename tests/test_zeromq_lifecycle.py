"""ZeroMQ receiver lifecycle (0.3.0 review, 2026-10-14): stop() leaves the poll loop before closing the sockets,
and RTAB-Map map id 0 names a session like any other map id."""
from __future__ import annotations

import json
import time

import pytest

from rtsm.io.ingest_queue import IngestQueue


def _sub(**kw):
    pytest.importorskip("zmq")
    from rtsm.io.zeromq import ZeroMQSubscriber
    return ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                            ingest_queue=IngestQueue(maxsize=8), **kw)


def test_stop_joins_the_poll_loop_before_closing():
    sub = _sub()
    sub.start()
    time.sleep(0.3)
    assert sub._thread is not None and sub._thread.is_alive()
    sub.stop(timeout=3.0)
    assert not sub._thread.is_alive()
    assert sub._closed is True
    sub.stop()                                   # idempotent: a second stop (the runner's finally) does nothing
    sub.close()


def test_stop_without_a_started_thread_just_closes():
    sub = _sub()
    sub.stop()
    assert sub._closed is True


def test_map_id_zero_names_a_session():
    seen = []
    sub = _sub(on_session_change=lambda sid, epoch: seen.append((sid, epoch)))
    try:
        msg = [b"rtabmap.map_correction", json.dumps({"T_w_odom": {"t": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]},
                                                      "map_id": 0, "ts_ns": 1}).encode()]
        sub._handle_map_correction(msg)
        assert seen == [("rtabmap-map-0", 1)]
        assert sub.stats()["map_correction"]["map_id"] == 0
        sub._handle_map_correction(msg)          # the same map again: no new session
        assert seen == [("rtabmap-map-0", 1)]
    finally:
        sub.close()
