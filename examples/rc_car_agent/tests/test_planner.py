"""planner tests — fake Anthropic clients (zero network to the LLM),
mock RTSM server for /search/semantic + /objects/{id}."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from config import load_config
from planner import extract_query, plan
from rtsm_client import RtsmClient

POSE = {"xyz": [0.0, 0.3, 0.0], "quaternion_xyzw": [0, 0, 0, 1],
        "timestamp": 1751000000.0}

SEMANTIC = {
    "query": "red mug",
    "robot_pose": POSE,
    "results": [
        # ranked by score; obj_2 outranks but is UNCONFIRMED
        {"id": "obj_2", "score": 0.20, "confirmed": False,
         "stability": 0.25, "xyz_world": [2.0, 0.1, 2.0]},
        {"id": "obj_7", "score": 0.14, "confirmed": True,
         "stability": 0.71, "xyz_world": [0.5, 0.8, 1.5]},
        {"id": "obj_9", "score": 0.09, "confirmed": True,
         "stability": 0.60, "xyz_world": None},          # not navigable
    ],
}

OBJECTS = {
    "/objects/obj_2": {"id": "obj_2", "label_primary": "cup"},
    "/objects/obj_7": {"id": "obj_7", "label_primary": "mug"},
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        raw = getattr(self.server, "raw_routes", {}).get(path)
        if raw is not None:                    # bytes route (snapshot JPEGs)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        payload = self.server.routes.get(path)
        if payload is None:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def rtsm_env():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.routes = {"/search/semantic": json.loads(json.dumps(SEMANTIC)), **OBJECTS}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = load_config()
    client = RtsmClient(f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv, client, cfg
    srv.shutdown()


class FakeLLM:
    """Duck-typed anthropic client: .messages.create -> canned tool_use."""

    def __init__(self, target_id, reason="matches goal", raise_exc=False,
                 no_match=False):
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                if raise_exc:
                    raise RuntimeError("api down")
                inp = ({"no_match": True, "reason": reason} if no_match
                       else {"target_id": target_id, "reason": reason})
                block = SimpleNamespace(type="tool_use", input=inp)
                return SimpleNamespace(content=[block])

        self.messages = _Messages()


# ── extract_query ────────────────────────────────────────────────────────


def test_extract_query_variants():
    assert extract_query("go to the red mug") == "red mug"
    assert extract_query("Go To the Red Mug") == "Red Mug"   # prefix case-insensitive
    assert extract_query("navigate to blue backpack") == "blue backpack"
    assert extract_query("red mug") == "red mug"              # passthrough


# ── selection paths ──────────────────────────────────────────────────────


def test_haiku_pick_valid_id(rtsm_env):
    _, client, cfg = rtsm_env
    llm = FakeLLM("obj_7")
    r = plan("go to the red mug", client, cfg, anthropic_client=llm)
    assert (r.status, r.planner_path, r.target_id) == ("ok", "haiku", "obj_7")
    assert r.label == "mug"                    # enriched via /objects/{id}
    assert r.xyz_world == [0.5, 0.8, 1.5]
    assert r.plan_pose is not None             # snapshot from the same response
    # the forced-tool contract actually went out
    kw = llm.calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "select_target"}
    texts = " ".join(b["text"] for b in kw["messages"][0]["content"]
                     if b.get("type") == "text")
    assert "obj_7" in texts


def test_hallucinated_id_falls_back(rtsm_env):
    _, client, cfg = rtsm_env
    r = plan("go to the red mug", client, cfg, anthropic_client=FakeLLM("obj_999"))
    assert r.planner_path == "top1_fallback"
    assert r.target_id == "obj_7"              # confirmed-first ranking wins


def test_llm_exception_falls_back(rtsm_env):
    _, client, cfg = rtsm_env
    r = plan("go to the red mug", client, cfg,
             anthropic_client=FakeLLM("obj_7", raise_exc=True))
    assert (r.planner_path, r.target_id) == ("top1_fallback", "obj_7")


def test_snapshots_attached_as_image_blocks(rtsm_env):
    """Candidates' camera crops ride along as base64 image blocks so the
    LLM judges by the image, not the captioner's label (which called the
    real teddy bear 'audio' and got it rejected, t20260811-200758-001)."""
    import base64
    srv, client, cfg = rtsm_env
    srv.raw_routes = {
        "/objects/obj_7/snapshots/0/image": b"\xff\xd8mugjpeg",
        "/objects/obj_2/snapshots/0/image": b"\xff\xd8cupjpeg",
    }
    llm = FakeLLM("obj_7")
    r = plan("go to the red mug", client, cfg, anthropic_client=llm)
    assert r.status == "ok"
    content = llm.calls[0]["messages"][0]["content"]
    imgs = [b for b in content if b.get("type") == "image"]
    assert len(imgs) == 2                      # obj_9 has no xyz -> not eligible
    # confirmed-first ordering: obj_7's crop is the first image block
    assert imgs[0]["source"]["data"] == base64.standard_b64encode(
        b"\xff\xd8mugjpeg").decode("ascii")
    assert imgs[0]["source"]["media_type"] == "image/jpeg"


def test_missing_snapshots_degrade_to_text_only(rtsm_env):
    """Snapshot endpoint 404s (or RTSM hiccups) -> the pick still goes out
    with text-only candidates. Crops are evidence, never a dependency."""
    _, client, cfg = rtsm_env                  # no raw_routes -> all 404
    llm = FakeLLM("obj_7")
    r = plan("go to the red mug", client, cfg, anthropic_client=llm)
    assert (r.status, r.target_id) == ("ok", "obj_7")
    content = llm.calls[0]["messages"][0]["content"]
    assert not [b for b in content if b.get("type") == "image"]


def test_haiku_no_match_is_respected_not_overridden(rtsm_env):
    """The LLM declaring 'nothing plausibly matches' must produce a clean
    not_found — NEVER the ranked-top-1 fallback. The fallback overriding
    an explicit no-match is exactly the settle bug that drove the baseline
    at a smartphone for 160 s on 'teddy bear' (2026-08-11)."""
    _, client, cfg = rtsm_env
    llm = FakeLLM(None, reason="all candidates are furniture", no_match=True)
    r = plan("go to the teddy bear", client, cfg, anthropic_client=llm)
    assert r.status == "not_found"
    assert r.planner_path == "haiku_no_match"
    assert r.target_id is None
    assert r.reason == "all candidates are furniture"


def test_no_llm_path(rtsm_env, monkeypatch):
    _, client, cfg = rtsm_env
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = plan("go to the red mug", client, cfg, use_llm=False)
    assert (r.planner_path, r.target_id) == ("top1_no_llm", "obj_7")


def test_confirmed_partition_beats_raw_score(rtsm_env):
    # obj_2 has the higher score but is unconfirmed → top-1 is confirmed obj_7.
    _, client, cfg = rtsm_env
    r = plan("go to the red mug", client, cfg, use_llm=False)
    assert r.target_id == "obj_7"
    assert r.confirmed is True


# ── not-found ────────────────────────────────────────────────────────────


def test_not_found_empty_results(rtsm_env):
    srv, client, cfg = rtsm_env
    srv.routes["/search/semantic"] = {"query": "piano", "robot_pose": POSE,
                                      "results": []}
    r = plan("go to the piano", client, cfg, use_llm=False)
    assert (r.status, r.target_id) == ("not_found", None)
    assert r.reason == "no results"


def test_not_found_when_no_candidate_has_xyz(rtsm_env):
    srv, client, cfg = rtsm_env
    srv.routes["/search/semantic"]["results"] = [
        {"id": "obj_9", "score": 0.2, "confirmed": True,
         "stability": 0.6, "xyz_world": None},
    ]
    r = plan("go to the red mug", client, cfg, use_llm=False)
    assert r.status == "not_found"
    assert "3D position" in (r.reason or "")


# ── label enrichment robustness ─────────────────────────────────────────


def test_missing_object_detail_tolerated(rtsm_env):
    srv, client, cfg = rtsm_env
    del srv.routes["/objects/obj_7"]           # detail endpoint 404s
    r = plan("go to the red mug", client, cfg, use_llm=False)
    assert (r.status, r.target_id) == ("ok", "obj_7")
    assert r.label is None                     # enrichment optional, never fatal
