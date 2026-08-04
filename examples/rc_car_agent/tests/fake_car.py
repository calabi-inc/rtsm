"""
Kinematic fake car — the closed-loop test fixture Gate F requires.

A differential-drive simulator whose pose ADVANCES in response to the
/drive commands the agent sends, exposed through the same two HTTP mocks
the rest of the suite uses:

    FakeRtsmHandler  — /healthz, /stats, /search/semantic, /objects/{id};
                       robot_pose is generated live from the sim (fresh,
                       advancing timestamps + frame_epoch), so the nav
                       loop's re-aim actually closes the loop.
    FakeCarEsp32Handler — records every request like _RecordingHandler AND
                       feeds /drive commands into the sim; /stop zeroes it.

Sim conventions MIRROR the production stack (geometry.py facts 1-4):
world Y-up, ground = X-Z, yaw = atan2(x, z) with 0 = +Z and positive =
CCW/left, camera quaternion = rotation about +Y by yaw (xyzw), so
forward = R(q)@[0,0,1] = (sin yaw, 0, cos yaw). Differential drive:
right wheel faster -> CCW (positive yaw rate) — the firmware convention.

Test controls:
    car.freeze()        timestamps stop advancing (stale/frozen tests)
    car.epoch = N       served frame_epoch (bump mid-drive for reset tests)
    car.teleport(x, z)  discontinuity injection
"""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler

_TS_BASE = 1751000000.0


class FakeCar:
    def __init__(self, x: float = 0.0, z: float = 0.0, yaw: float = 0.0,
                 speed_scale_mps: float = 2.0, turn_scale_rps: float = 3.0,
                 epoch: int = 7,
                 cam_yaw_offset: float = 0.0,
                 cam_lever_arm_rf: tuple = (0.0, 0.0)):
        """(x, z, yaw) is the DRIVE CENTER pose. The served pose is the
        CAMERA's — displaced by the mount model so calibration routines can
        be tested by planting constants and asserting recovery:
          cam_yaw_offset: car_yaw = cam_yaw + offset (geometry.camera_to_car
              convention), i.e. the camera is skewed CW-of-car by +offset.
          cam_lever_arm_rf: (right, forward) from CAMERA to DRIVE CENTER in
              the car body frame — same tuple calibrate.py must recover.
        Defaults (0, (0,0)) collapse to camera == drive center (all
        pre-existing tests unchanged)."""
        self._lock = threading.Lock()
        self.x, self.z, self.yaw = float(x), float(z), float(yaw)
        self._speed = float(speed_scale_mps)
        self._turn = float(turn_scale_rps)
        self.epoch = int(epoch)
        self._cam_yaw_offset = float(cam_yaw_offset)
        self._cam_lever_r = float(cam_lever_arm_rf[0])
        self._cam_lever_f = float(cam_lever_arm_rf[1])
        self._left = 0.0
        self._right = 0.0
        self._t0 = time.monotonic()
        self._last_advance = self._t0
        self._frozen_ts: float | None = None

    # ── controls (called from HTTP handlers / tests) ─────────────────────

    def set_drive(self, left: float, right: float) -> None:
        with self._lock:
            self._advance_locked()
            self._left, self._right = float(left), float(right)

    def stop(self) -> None:
        self.set_drive(0.0, 0.0)

    def freeze(self) -> None:
        """Sensor timestamps stop advancing (pose feed 'frozen')."""
        with self._lock:
            self._frozen_ts = _TS_BASE + (time.monotonic() - self._t0)

    def teleport(self, x: float, z: float) -> None:
        with self._lock:
            self._advance_locked()
            self.x, self.z = float(x), float(z)

    # ── sim ──────────────────────────────────────────────────────────────

    def _advance_locked(self) -> None:
        now = time.monotonic()
        dt = now - self._last_advance
        self._last_advance = now
        if dt <= 0:
            return
        v = 0.5 * (self._left + self._right) * self._speed
        w = 0.5 * (self._right - self._left) * self._turn   # right faster -> CCW
        self.yaw = math.atan2(math.sin(self.yaw + w * dt),
                              math.cos(self.yaw + w * dt))
        self.x += v * math.sin(self.yaw) * dt
        self.z += v * math.cos(self.yaw) * dt

    def pose_payload(self) -> dict:
        with self._lock:
            self._advance_locked()
            ts = (self._frozen_ts if self._frozen_ts is not None
                  else _TS_BASE + (time.monotonic() - self._t0))
            # Camera pose from the drive-center pose via the mount model:
            # car = cam + f*fwd(yaw) + r*right(yaw)  ->  cam = car - f*fwd - r*right
            # with fwd = (sin, cos), right = (-cos, sin) at CAR yaw.
            s, c = math.sin(self.yaw), math.cos(self.yaw)
            cam_x = self.x - self._cam_lever_f * s + self._cam_lever_r * c
            cam_z = self.z - self._cam_lever_f * c - self._cam_lever_r * s
            cam_yaw = self.yaw - self._cam_yaw_offset
            half = 0.5 * cam_yaw
            return {
                "xyz": [cam_x, 0.3, cam_z],
                "quaternion_xyzw": [0.0, math.sin(half), 0.0, math.cos(half)],
                "timestamp": ts,
                "frame_epoch": self.epoch,
            }

    def ground_dist_to(self, target_xyz) -> float:
        with self._lock:
            self._advance_locked()
            return math.hypot(float(target_xyz[0]) - self.x,
                              float(target_xyz[2]) - self.z)

    def camera_xz_yaw(self):
        """(cam_x, cam_z, cam_yaw) — for the handler's visibility model."""
        with self._lock:
            self._advance_locked()
            s, c = math.sin(self.yaw), math.cos(self.yaw)
            return (self.x - self._cam_lever_f * s + self._cam_lever_r * c,
                    self.z - self._cam_lever_f * c - self._cam_lever_r * s,
                    self.yaw - self._cam_yaw_offset)


class FakeRtsmHandler(BaseHTTPRequestHandler):
    """Serves server.car (FakeCar) + server.semantic_results (list of hit
    dicts) + server.objects (id -> detail dict). Attach all three to the
    ThreadingHTTPServer instance before starting."""

    def do_GET(self):
        path = self.path.split("?")[0]
        car: FakeCar = self.server.car
        if path == "/healthz":
            return self._json({"status": "ok"})
        if path == "/stats":
            return self._json({
                "objects": len(self.server.semantic_results),
                "confirmed": len(self.server.semantic_results),
                "robot_pose": car.pose_payload(),
            })
        if path == "/search/semantic":
            return self._json({
                "query": "q",
                "robot_pose": car.pose_payload(),
                "results": self._results_with_freshness(car),
            })
        if path.startswith("/objects/"):
            detail = self.server.objects.get(path.rsplit("/", 1)[1])
            if detail is not None:
                return self._json(detail)
        self.send_response(404)
        self.end_headers()

    def _results_with_freshness(self, car: FakeCar) -> list:
        """Stamp last_seen_wall_utc on each result. Without a visibility
        model (server.visibility is None/absent): always fresh — memory-
        condition tests behave exactly as before. WITH one ({fov_deg,
        range_m}): an object's last-seen only advances while the camera is
        actually pointed at it — which is what makes the baseline sweep
        meaningful in tests."""
        vis = getattr(self.server, "visibility", None)
        now = time.time()
        out = []
        if not hasattr(self.server, "obj_last_seen"):
            self.server.obj_last_seen = {}
        for entry in self.server.semantic_results:
            e = dict(entry)
            if vis is None:
                e["last_seen_wall_utc"] = now
            else:
                cx, cz, cyaw = car.camera_xz_yaw()
                tx, tz = e["xyz_world"][0], e["xyz_world"][2]
                bearing = math.atan2(tx - cx, tz - cz)
                off = math.atan2(math.sin(bearing - cyaw),
                                 math.cos(bearing - cyaw))
                dist = math.hypot(tx - cx, tz - cz)
                if (abs(off) <= math.radians(vis["fov_deg"]) / 2
                        and dist <= vis["range_m"]):
                    self.server.obj_last_seen[e["id"]] = now
                e["last_seen_wall_utc"] = self.server.obj_last_seen.get(
                    e["id"], 0.0)
            out.append(e)
        return out

    def _json(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FakeCarEsp32Handler(BaseHTTPRequestHandler):
    """_RecordingHandler semantics + drives server.car."""

    def _respond(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else None
        self.server.recorded.append(
            {"path": self.path, "body": body,
             "connection": self.headers.get("Connection"),
             "t": time.monotonic()}
        )
        car: FakeCar = self.server.car
        if self.path == "/drive" and body is not None:
            car.set_drive(body.get("left", 0.0), body.get("right", 0.0))
        elif self.path == "/stop":
            car.stop()
        self._respond({"ok": True})

    def do_GET(self):
        self.server.recorded.append(
            {"path": self.path, "body": None,
             "connection": self.headers.get("Connection"),
             "t": time.monotonic()}
        )
        if self.path == "/battery":
            self._respond({"mv": 7400})
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ESP32 fake car online\n")

    def log_message(self, *args):
        pass
