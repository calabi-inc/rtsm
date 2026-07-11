"""
Agent server — the persistent entry point (Phase E).

Boot once; command over HTTP; no rerun between goals:

    .venv/Scripts/python.exe server.py [--bench] [--host 127.0.0.1] [--port 8010]

    POST /command  {"goal": "go to the red mug", "condition": "rtsm"|"baseline"}
                   -> {task_id, accepted}   (immediately; worker runs it)
                   A new command PREEMPTS the current drive (safe stop first).
    GET  /status   -> full state (READY/NOT_READY/RUNNING/ESTOPPED, task, estop)
    POST /cancel   -> cancel current task -> idle
    POST /stop     -> convenience soft stop (bridge.stop) — NOT the hard e-stop
    POST /reset_estop -> operator re-arm after an e-stop (explicit two-step:
                   clears monitor AND bridge latch)

Safety model (locked):
  * The HARD e-stop is the EstopMonitor thread (gamepad-X) + Ctrl-C path +
    the ESP32 300 ms firmware watchdog — never an HTTP endpoint.
  * E-stop ABANDONS the mission; /command returns 503 until /reset_estop.
  * NOT_READY (preflight failed) -> /command 503 with the reasons; the
    server stays up and re-probes preflight on each /command attempt.

Phase-E scope: the worker is a STUB (updates status ticks; exercises
accept/preempt/cancel/estop). Phase F replaces the stub body with
planner -> nav -> monitor. `--bench` additionally allows the special goal
"__bench_dummy_drive__" (drives 0.2/0.2 through the gated bridge, car on
blocks, 45 s cap) — the Gate-E hardware e-stop test vehicle; it bypasses
preflight BY DESIGN and must only be used on the bench.

RTSM lifecycle: attach if :8002/healthz answers; else spawn
cfg.rtsm.spawn_cmd (the GPU env's python, NOT this venv) and wait for
healthz; RTSM is left running on exit (condition-(a) memory must survive
agent restarts) unless --kill-rtsm-on-exit.
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import Config, load_config
from esp32_bridge import Esp32Bridge
from estop import EstopMonitor
from rtsm_client import RtsmClient

BENCH_DUMMY_GOAL = "__bench_dummy_drive__"
_BENCH_SPEED = 0.2
_BENCH_CAP_S = 45.0
_TICK_S = 0.02


class CommandBody(BaseModel):
    goal: str
    condition: Optional[str] = None


class AgentServer:
    """All mutable state + the single-slot worker. FastAPI routes are a
    thin shell over this object (dependency-injected for tests)."""

    def __init__(self, cfg: Config, rtsm: RtsmClient, bridge: Esp32Bridge,
                 bench: bool = False, kill_rtsm_on_exit: Optional[bool] = None):
        self.cfg = cfg
        self.rtsm = rtsm
        self.bridge = bridge
        self.bench = bool(bench)
        self._kill_rtsm = cfg.rtsm.kill_on_exit if kill_rtsm_on_exit is None else kill_rtsm_on_exit

        self.state = "NOT_READY"          # NOT_READY | READY | RUNNING | ESTOPPED
        self.not_ready_reasons: list = []
        self._ready = False

        self.stop_event = threading.Event()
        self.monitor = EstopMonitor(bridge, self.stop_event)

        self._lock = threading.RLock()
        self._pending: Optional[Dict[str, Any]] = None
        self.current: Optional[Dict[str, Any]] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self._preempt = threading.Event()
        self._cancel = threading.Event()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._rtsm_proc: Optional[subprocess.Popen] = None
        self._seq = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def startup(self) -> None:
        self._rtsm_lifecycle()
        self.run_preflight()
        self.monitor.start()
        self._worker = threading.Thread(
            target=self._worker_loop, name="agent-worker", daemon=True
        )
        self._worker.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        self.monitor.shutdown()
        # Hygiene stop on exit (skip if e-stop already latched things safe).
        if not self.bridge.estopped:
            self.bridge.stop()
        if self._rtsm_proc is not None and self._kill_rtsm:
            self._rtsm_proc.terminate()

    def _rtsm_lifecycle(self) -> None:
        mode = self.cfg.rtsm.lifecycle
        if mode == "off":
            return
        if self.rtsm.healthz():
            return                                   # attach
        if mode != "spawn":
            return                                   # attach-only: preflight reports it
        try:
            self._rtsm_proc = subprocess.Popen(self.cfg.rtsm.spawn_cmd.split())
        except OSError:
            return
        deadline = time.monotonic() + 90.0           # GPU model loads are slow
        while time.monotonic() < deadline:
            if self.rtsm.healthz():
                return
            time.sleep(1.0)

    # ── preflight (§5.4 — re-checkable; never run X–Z math on None) ──────

    def run_preflight(self) -> list:
        reasons = []
        if self.bridge.ping() is None:
            reasons.append("ESP32 unreachable — car powered? same WiFi?")
        else:
            mv = self.bridge.battery_mv()
            if mv is None:
                reasons.append("ESP32 battery unreadable")
            elif mv < self.cfg.esp32.battery_min_mv:
                reasons.append(f"battery {mv} mV < {self.cfg.esp32.battery_min_mv} mV — charge first")
        if not self.rtsm.healthz():
            reasons.append("RTSM unreachable — start it (python -m rtsm) in the GPU env")
        else:
            try:
                stats = self.rtsm.stats()
                if not stats.get("robot_pose"):
                    reasons.append("no robot_pose yet — is the iPhone (Calabi Lens) streaming?")
                if int(stats.get("objects", 0)) == 0:
                    reasons.append("RTSM map empty — scan the room before goals")
            except Exception as e:  # noqa: BLE001
                reasons.append(f"RTSM /stats failed: {e}")
        with self._lock:
            self.not_ready_reasons = reasons
            self._ready = not reasons
            if self.state not in ("RUNNING", "ESTOPPED"):
                self.state = "READY" if self._ready else "NOT_READY"
        return reasons

    # ── command intake ───────────────────────────────────────────────────

    def submit(self, goal: str, condition: Optional[str]) -> Dict[str, Any]:
        cond = condition or self.cfg.server.default_condition
        if cond not in ("rtsm", "baseline"):
            raise HTTPException(400, f"condition must be rtsm|baseline, got {cond!r}")
        if self.monitor.triggered or self.bridge.estopped:
            raise HTTPException(503, "ESTOPPED — POST /reset_estop to re-arm (operator action)")

        is_bench_goal = goal == BENCH_DUMMY_GOAL
        if is_bench_goal and not self.bench:
            raise HTTPException(403, "bench goal requires the server to run with --bench")

        # Bench dummy-drive bypasses preflight BY DESIGN (its whole point is
        # testing the e-stop without the full stack). Everything else gates.
        if not is_bench_goal:
            with self._lock:
                ready = self._ready
            if not ready:
                reasons = self.run_preflight()       # re-probe: maybe the phone just connected
                if reasons:
                    raise HTTPException(503, {"not_ready": reasons})

        with self._lock:
            self._seq += 1
            task = {
                "task_id": f"t{datetime.now():%Y%m%d}-{self._seq:03d}",
                "goal": goal,
                "condition": cond,
                "phase": "queued",
                "stub_ticks": 0,
                "result": None,
                "accepted_at": time.time(),
            }
            if self.current is not None:
                self._preempt.set()                  # safe-stop, then swap
            self._pending = task
            self._wake.set()
        return {"task_id": task["task_id"], "accepted": True, "condition": cond}

    def cancel(self) -> Dict[str, Any]:
        with self._lock:
            running = self.current is not None or self._pending is not None
            self._pending = None
        if running:
            self._cancel.set()
        return {"cancelled": running}

    def soft_stop(self) -> Dict[str, Any]:
        """Convenience stop — NOT the hard e-stop (no latch)."""
        self.cancel()
        ok = self.bridge.stop()
        return {"stopped": ok}

    def reset_estop(self) -> Dict[str, Any]:
        self.monitor.reset()
        self.bridge.reset_estop()
        with self._lock:
            # The operator ack is the ONLY path allowed to leave ESTOPPED;
            # preflight's own guard refuses to touch that state.
            if self.state == "ESTOPPED":
                self.state = "NOT_READY"          # provisional; preflight decides
        self.run_preflight()
        return {"state": self.state}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "not_ready_reasons": list(self.not_ready_reasons),
                "task": dict(self.current) if self.current else None,
                "last_result": dict(self.last_result) if self.last_result else None,
                "estop": self.monitor.status(),
                "bench": self.bench,
            }

    # ── single-slot worker ───────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            self._wake.wait(timeout=0.05)
            with self._lock:
                task = self._pending
                self._pending = None
                if task is None:
                    self._wake.clear()
            if task is not None:
                self._run_task(task)

    def _run_task(self, task: Dict[str, Any]) -> None:
        """PHASE-E STUB: ticks status until preempted/cancelled/estopped.
        Phase F replaces this body with planner -> nav -> monitor.
        The bench dummy goal streams a gentle /drive so the hardware
        e-stop test has real motion to kill."""
        with self._lock:
            self.current = task
            task["phase"] = "stub_running"
            prev_state = self.state
            self.state = "RUNNING"

        is_bench = task["goal"] == BENCH_DUMMY_GOAL
        t0 = time.monotonic()
        result = "stub_ended"
        while not self._shutdown.is_set():
            if self.stop_event.is_set():
                result = "estopped"                  # monitor already stopped the car
                break
            if self._preempt.is_set():
                self._preempt.clear()
                self.bridge.stop()                   # safe stop before swap
                result = "preempted"
                break
            if self._cancel.is_set():
                self._cancel.clear()
                self.bridge.stop()
                result = "cancelled"
                break
            task["stub_ticks"] += 1
            if is_bench:
                self.bridge.drive(_BENCH_SPEED, _BENCH_SPEED)   # gated internally
                if time.monotonic() - t0 > _BENCH_CAP_S:
                    self.bridge.stop()
                    result = "bench_timeout"
                    break
            time.sleep(_TICK_S)

        with self._lock:
            task["result"] = result
            task["phase"] = "finished"
            self.last_result = dict(task)
            self.current = None
            if result == "estopped":
                self.state = "ESTOPPED"
            else:
                self.state = "READY" if self._ready else (
                    prev_state if prev_state == "NOT_READY" else "NOT_READY"
                )


# ── FastAPI shell ────────────────────────────────────────────────────────


def create_app(cfg: Optional[Config] = None,
               rtsm: Optional[RtsmClient] = None,
               bridge: Optional[Esp32Bridge] = None,
               bench: bool = False,
               kill_rtsm_on_exit: Optional[bool] = None) -> FastAPI:
    cfg = cfg or load_config()
    rtsm = rtsm or RtsmClient(cfg.rtsm.url)
    bridge = bridge or Esp32Bridge(
        cfg.esp32.url,
        drive_rate_hz=cfg.esp32.drive_rate_hz,
        heartbeat_s=cfg.esp32.heartbeat_s,
        change_epsilon=cfg.esp32.change_epsilon,
        http_timeout_s=cfg.esp32.http_timeout_s,
    )
    srv = AgentServer(cfg, rtsm, bridge, bench=bench,
                      kill_rtsm_on_exit=kill_rtsm_on_exit)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        srv.startup()
        try:
            yield
        finally:
            srv.shutdown()

    app = FastAPI(title="rc_car_agent", lifespan=lifespan)
    app.state.srv = srv                              # test access

    @app.post("/command")
    def command(body: CommandBody):
        return srv.submit(body.goal, body.condition)

    @app.get("/status")
    def status():
        return srv.status()

    @app.post("/cancel")
    def cancel():
        return srv.cancel()

    @app.post("/stop")
    def stop():
        return srv.soft_stop()

    @app.post("/reset_estop")
    def reset_estop():
        return srv.reset_estop()

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="RC car agent server (Demo 2)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--bench", action="store_true",
                   help="enable __bench_dummy_drive__ (car on blocks ONLY)")
    p.add_argument("--kill-rtsm-on-exit", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    app = create_app(cfg, bench=args.bench,
                     kill_rtsm_on_exit=args.kill_rtsm_on_exit or None)
    try:
        uvicorn.run(app,
                    host=args.host or cfg.server.host,
                    port=args.port or cfg.server.port)
    except KeyboardInterrupt:
        pass
    finally:
        # Belt-and-braces: lifespan shutdown already stops the car; the
        # ESP32 watchdog covers even a hard process kill.
        srv = app.state.srv
        if not srv.bridge.estopped:
            srv.bridge.stop()


if __name__ == "__main__":
    main()
