"""E2E integration test for embedded MCP SSE server."""
import asyncio
import json
import threading
import time

import numpy as np
import uvicorn
import httpx
from prometheus_client import CollectorRegistry

from rtsm.stores.working_memory import WorkingMemory, ObjectState
from rtsm.stores.proximity_index import ProximityIndex, GridSpec
from rtsm.api.server import create_app


def build_test_app():
    grid = GridSpec(cell_m=0.25, use_3d=False, up_axis="y")
    pi = ProximityIndex(grid, per_cell_cap=64, neighbors_max=128)
    cfg = {
        "object": {"stability_hit": 0.05, "confirm_stability": 0.5,
                    "confirm_views": 2, "promo_min_hits": 3,
                    "proto_ttl_s": 30.0, "max_gallery": 5,
                    "gallery_cos_thresh": 0.85, "max_crops": 8},
        "vectors": {"flush_period_s": 10.0, "min_delta_cos": 0.05,
                    "min_delta_m": 0.1},
    }
    wm = WorkingMemory(cfg, index=pi)

    for oid, xyz, label in [("obj_mug", [1, 0, 1], "mug"), ("obj_book", [1.2, 0, 1.1], "book")]:
        o = ObjectState(
            id=oid, xyz_world=np.array(xyz, dtype=np.float32),
            cov_world=np.array([0.01, 0.01, 0.01], dtype=np.float32),
            emb_mean=np.zeros(512, dtype=np.float32),
            emb_gallery=np.zeros((0, 512), dtype=np.float16),
            view_bins={}, label_scores={label: 1.0}, label_primary=label,
            stability=0.8, hits=5, confirmed=True,
            created_mono=1000.0, created_wall_utc=1000.0,
            last_seen_mono=1000.0, last_seen_wall_utc=1000.0,
            last_seen_px=None, last_upsert_wall_utc=0.0, last_upsert_mono=0.0,
            last_upsert_emb=None, last_upsert_xyz=None,
            image_crops=[], last_update_frame_id=None, _dim=512,
        )
        wm._map[oid] = o
        pi.insert(oid, o.xyz_world)

    return create_app(working_memory=wm, registry=CollectorRegistry(), mcp_enabled=True)


async def _next_data(lines) -> dict:
    """Read the next SSE 'data:' line from the stream and parse as JSON."""
    async for line in lines:
        if line.startswith("data:"):
            return json.loads(line[5:])
    raise RuntimeError("SSE stream ended without data")


async def mcp_protocol_test(base_url: str):
    async with httpx.AsyncClient(base_url=base_url) as client:
        async with client.stream("GET", "/mcp/sse") as sse_resp:
            print(f"  SSE status: {sse_resp.status_code}")
            assert sse_resp.status_code == 200

            # httpx only allows one aiter_lines() call per stream — keep a single iterator
            lines = sse_resp.aiter_lines()

            # Read endpoint event (first data line is the POST URL)
            endpoint_url = None
            async for line in lines:
                if line.startswith("data:"):
                    endpoint_url = line[5:].strip()
                    break
            print(f"  Endpoint: {endpoint_url}")
            assert endpoint_url and "session_id" in endpoint_url

            # Initialize
            r = await client.post(endpoint_url, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "test", "version": "1.0"}}
            })
            assert r.status_code == 202
            resp = await _next_data(lines)
            info = resp["result"]["serverInfo"]
            print(f"  Server: {info}")
            assert info["name"] == "rtsm"

            # Initialized notification
            await client.post(endpoint_url, json={
                "jsonrpc": "2.0", "method": "notifications/initialized"
            })

            # tools/list
            await client.post(endpoint_url, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
            })
            resp = await _next_data(lines)
            tools = resp["result"]["tools"]
            names = [t["name"] for t in tools]
            print(f"  Tools ({len(tools)}): {names}")
            assert len(tools) == 6
            assert "rtsm.semantic_query" in names
            assert "rtsm.spatial_query" in names

            # tools/call: rtsm.list_objects
            await client.post(endpoint_url, json={
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "rtsm.list_objects", "arguments": {}}
            })
            resp = await _next_data(lines)
            result = json.loads(resp["result"]["content"][0]["text"])
            print(f"  list_objects: count={result['count']}, ids={[o['id'] for o in result['objects']]}")
            assert result["count"] == 2

            # tools/call: rtsm.spatial_query
            await client.post(endpoint_url, json={
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "rtsm.spatial_query",
                           "arguments": {"x": 1.0, "y": 0.0, "z": 1.0, "radius_m": 0.3}}
            })
            resp = await _next_data(lines)
            result = json.loads(resp["result"]["content"][0]["text"])
            ids = [o["id"] for o in result["results"]]
            print(f"  spatial_query(1,0,1 r=0.3): {ids}")
            assert "obj_mug" in ids
            assert "obj_book" in ids

            # tools/call: rtsm.status
            await client.post(endpoint_url, json={
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "rtsm.status", "arguments": {}}
            })
            resp = await _next_data(lines)
            result = json.loads(resp["result"]["content"][0]["text"])
            print(f"  status: health={result['health']}, objects={result['stats'].get('objects')}")
            assert result["health"]["status"] == "ok"


def main():
    PORT = 18768
    app = build_test_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    srv = uvicorn.Server(config)
    srv.install_signal_handlers = lambda: None
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    time.sleep(1.5)

    print("=== TEST: REST endpoints ===")
    r = httpx.get(f"http://127.0.0.1:{PORT}/search/spatial", params={"x": 1, "y": 0, "z": 1, "radius_m": 0.5})
    print(f"  /search/spatial: {r.status_code}, results={len(r.json()['results'])}")
    assert r.status_code == 200

    r = httpx.get(f"http://127.0.0.1:{PORT}/objects")
    print(f"  /objects: {r.status_code}, count={r.json()['count']}")
    assert r.status_code == 200

    print()
    print("=== TEST: MCP SSE full protocol ===")
    asyncio.run(asyncio.wait_for(mcp_protocol_test(f"http://127.0.0.1:{PORT}"), timeout=10))

    print()
    print("=== ALL E2E TESTS PASSED ===")
    srv.should_exit = True


if __name__ == "__main__":
    main()
