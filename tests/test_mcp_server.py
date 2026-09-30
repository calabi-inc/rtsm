"""Unit tests for the RTSM MCP server tool handlers."""
from __future__ import annotations

import json
import pytest
import pytest_asyncio
import httpx
import respx

from rtsm.io.mcp_server import _dispatch, RTSM_API_URL


@pytest_asyncio.fixture(autouse=True)
async def _reset_client():
    """Reset the module-level httpx client between tests. Async so the client
    is closed in the test's own event loop: pytest-asyncio >= 1.4 leaves no
    current loop at a sync teardown (asyncio.get_event_loop() raises there)."""
    import rtsm.io.mcp_server as mod
    mod._client = None
    yield
    if mod._client and not mod._client.is_closed:
        await mod._client.aclose()
    mod._client = None


# ── semantic_query ──

@respx.mock
@pytest.mark.asyncio
async def test_semantic_query_success():
    respx.get(f"{RTSM_API_URL}/search/semantic").mock(
        return_value=httpx.Response(200, json={
            "query": "coffee mug",
            "results": [
                {"id": "obj_1", "score": 0.92, "label_hint": "mug", "confirmed": True, "xyz_world": [1, 0, 2]},
                {"id": "obj_2", "score": 0.75, "label_hint": "cup", "confirmed": True, "xyz_world": [3, 0, 1]},
            ],
        })
    )
    result = await _dispatch("rtsm.semantic_query", {"query": "coffee mug", "top_k": 5})
    assert result["query"] == "coffee mug"
    assert len(result["results"]) == 2


@respx.mock
@pytest.mark.asyncio
async def test_semantic_query_empty():
    respx.get(f"{RTSM_API_URL}/search/semantic").mock(
        return_value=httpx.Response(200, json={"query": "piano", "results": []})
    )
    result = await _dispatch("rtsm.semantic_query", {"query": "piano"})
    assert result["results"] == []


# ── spatial_query ──

@respx.mock
@pytest.mark.asyncio
async def test_spatial_query_success():
    respx.get(f"{RTSM_API_URL}/search/spatial").mock(
        return_value=httpx.Response(200, json={
            "center": [1.0, 0.0, 2.0],
            "radius_m": 1.0,
            "results": [
                {"id": "obj_a", "distance_m": 0.12, "label_primary": "mug"},
                {"id": "obj_b", "distance_m": 0.45, "label_primary": "book"},
                {"id": "obj_c", "distance_m": 0.89, "label_primary": "pen"},
            ],
        })
    )
    result = await _dispatch("rtsm.spatial_query", {"x": 1.0, "y": 0.0, "z": 2.0, "radius_m": 1.0})
    assert len(result["results"]) == 3
    distances = [r["distance_m"] for r in result["results"]]
    assert distances == sorted(distances)


# ── relational_query ──

@respx.mock
@pytest.mark.asyncio
async def test_relational_query_composition():
    # Step 1: semantic search finds the laptop
    respx.get(f"{RTSM_API_URL}/search/semantic").mock(
        return_value=httpx.Response(200, json={
            "query": "laptop",
            "results": [{"id": "obj_laptop", "score": 0.95, "label_hint": "laptop",
                         "confirmed": True, "xyz_world": [2.0, 0.0, 1.0]}],
        })
    )
    # Step 2: spatial search around the laptop
    respx.get(f"{RTSM_API_URL}/search/spatial").mock(
        return_value=httpx.Response(200, json={
            "center": [2.0, 0.0, 1.0],
            "radius_m": 1.0,
            "results": [
                {"id": "obj_laptop", "distance_m": 0.0, "label_primary": "laptop"},
                {"id": "obj_mouse", "distance_m": 0.25, "label_primary": "mouse"},
                {"id": "obj_pen", "distance_m": 0.40, "label_primary": "pen"},
            ],
        })
    )
    result = await _dispatch("rtsm.relational_query", {"query": "laptop", "radius_m": 1.0})
    assert result["reference"]["id"] == "obj_laptop"
    nearby_ids = [o["id"] for o in result["nearby"]]
    assert "obj_laptop" not in nearby_ids  # reference excluded
    assert "obj_mouse" in nearby_ids
    assert "obj_pen" in nearby_ids


@respx.mock
@pytest.mark.asyncio
async def test_relational_query_no_reference():
    respx.get(f"{RTSM_API_URL}/search/semantic").mock(
        return_value=httpx.Response(200, json={"query": "piano", "results": []})
    )
    result = await _dispatch("rtsm.relational_query", {"query": "piano"})
    assert result["error"] == "not_found"


# ── list_objects ──

@respx.mock
@pytest.mark.asyncio
async def test_list_objects():
    respx.get(f"{RTSM_API_URL}/objects").mock(
        return_value=httpx.Response(200, json={
            "count": 2,
            "objects": [
                {"id": "obj_1", "label_primary": "mug", "confirmed": True},
                {"id": "obj_2", "label_primary": "book", "confirmed": False},
            ],
        })
    )
    result = await _dispatch("rtsm.list_objects", {})
    assert result["count"] == 2
    assert len(result["objects"]) == 2


# ── get_object ──

@respx.mock
@pytest.mark.asyncio
async def test_get_object_found():
    respx.get(f"{RTSM_API_URL}/objects/obj_abc").mock(
        return_value=httpx.Response(200, json={
            "id": "obj_abc", "xyz_world": [1.0, 0.0, 2.0],
            "label_primary": "mug", "confirmed": True, "stability": 0.85,
        })
    )
    result = await _dispatch("rtsm.get_object", {"object_id": "obj_abc"})
    assert result["id"] == "obj_abc"
    assert result["label_primary"] == "mug"


@respx.mock
@pytest.mark.asyncio
async def test_get_object_not_found():
    respx.get(f"{RTSM_API_URL}/objects/nonexistent").mock(
        return_value=httpx.Response(200, json={"error": "not_found", "id": "nonexistent"})
    )
    result = await _dispatch("rtsm.get_object", {"object_id": "nonexistent"})
    assert result["error"] == "not_found"


# ── status ──

@respx.mock
@pytest.mark.asyncio
async def test_status_healthy():
    respx.get(f"{RTSM_API_URL}/healthz").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.get(f"{RTSM_API_URL}/stats").mock(
        return_value=httpx.Response(200, json={"objects": 12, "confirmed": 8})
    )
    result = await _dispatch("rtsm.status", {})
    assert result["health"]["status"] == "ok"
    assert result["stats"]["objects"] == 12


# ── error handling ──

@pytest.mark.asyncio
async def test_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        await _dispatch("rtsm.nonexistent", {})


@respx.mock
@pytest.mark.asyncio
async def test_api_http_error():
    respx.get(f"{RTSM_API_URL}/search/semantic").mock(
        return_value=httpx.Response(503, text="Semantic search not available")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await _dispatch("rtsm.semantic_query", {"query": "test"})
