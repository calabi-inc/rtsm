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
    POST /preflight -> force a preflight re-probe (console "Re-check" button)
    GET  /trial_log?tail=N -> parsed tail of the current/last trial's JSONL
    GET  /ui (also /) -> operator console: single-file web page (ui.html)
                   for a laptop on the same WiFi. Requires --host 0.0.0.0
                   (config default stays 127.0.0.1). The console gets the
                   SOFT stop only — the hard e-stop is never HTTP.

Safety model (locked):
  * The HARD e-stop is the EstopMonitor thread (gamepad-X) + Ctrl-C path +
    the ESP32 300 ms firmware watchdog — never an HTTP endpoint.
  * E-stop ABANDONS the mission; /command returns 503 until /reset_estop.
  * server.require_verified_estop (default true): EVERY motion goal —
    bench included — 503s until the pad is bound AND live-fire verified
    on the CURRENT binding. Remote-console gate: nobody is at the PC to
    notice a dead pad.
  * server.api_token (optional): POST endpoints require X-Auth-Token when
    set — mandatory hygiene for --host 0.0.0.0 (any device on the LAN can
    otherwise command motion).
  * NOT_READY (preflight failed) -> /command 503 with the reasons; the
    server stays up and re-probes preflight on each /command attempt.

Worker (Phase F): planner -> nav -> monitor. One /command = one trial:
plan() picks the target (Haiku forced-tool, top-1 fallback), NavRunner
closed-loop drives against live RTSM pose, MissionMonitor is the sole
arrival/abort authority, TrialLogger writes the E1 JSONL. `--bench`
additionally allows the special goal "__bench_dummy_drive__" (drives
0.2/0.2 through the gated bridge, car on blocks, 45 s cap) — the Gate-E
hardware e-stop test vehicle; it bypasses preflight BY DESIGN and must
only be used on the bench.

RTSM lifecycle: attach if :8002/healthz answers; else spawn
cfg.rtsm.spawn_cmd (the GPU env's python, NOT this venv) and wait for
healthz; RTSM is left running on exit (condition-(a) memory must survive
agent restarts) unless --kill-rtsm-on-exit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from baseline_search import BaselineSearcher, derive_seed
from config import Config, load_config
from esp32_bridge import Esp32Bridge
from estop import EstopMonitor
from nav import NavRunner
from planner import (PlanResult, extract_query, plan as plan_target,
                     select_target_from_hits)
from rtsm_client import RtsmClient
from trial_logger import TrialLogger

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
        # Monitor BEFORE preflight: preflight reports the kill-switch
        # binding, which only exists once the monitor has run its
        # main-thread SDL init.
        self.monitor.start()
        self.run_preflight()
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
        if self.cfg.server.require_verified_estop and not self._estop_live():
            reasons.append(
                "kill switch not live-fire verified — plug in the wired pad "
                "and press a button (the protocol's X check) before missions")
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

    def _estop_live(self) -> bool:
        """Kill switch bound AND live-fire verified on the CURRENT binding.
        Read fresh from the monitor every time — never cached, so a pad
        that slept mid-session (binding_verified reset) blocks the next
        command even though preflight's _ready is still True."""
        return bool(self.monitor.gamepad_available
                    and self.monitor.binding_verified)

    def submit(self, goal: str, condition: Optional[str]) -> Dict[str, Any]:
        cond = condition or self.cfg.server.default_condition
        if cond not in ("rtsm", "baseline"):
            raise HTTPException(400, f"condition must be rtsm|baseline, got {cond!r}")
        if self.monitor.triggered or self.bridge.estopped:
            raise HTTPException(503, "ESTOPPED — POST /reset_estop to re-arm (operator action)")
        # EVERY motion goal — bench included, its whole purpose is e-stop
        # testing — requires a working, press-proven kill switch. This is
        # the remote-operator gate: with the console on another laptop,
        # nobody is at the PC to notice a dead pad.
        if self.cfg.server.require_verified_estop and not self._estop_live():
            raise HTTPException(503, {
                "kill_switch": "not live-fire verified — plug in the wired "
                               "pad and press a button (protocol X check); "
                               "required again after any pad sleep/reconnect"})

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
            # ALL keys pre-seeded: the worker/nav thread only ever updates
            # values, so status()'s dict() copy never races a size change.
            task = {
                # Timestamped to the second: ids stay unique across server
                # restarts (the seq counter resets), so trial JSONLs never
                # merge two trials into one file.
                "task_id": f"t{datetime.now():%Y%m%d-%H%M%S}-{self._seq:03d}",
                "goal": goal,
                "condition": cond,
                "phase": "queued",
                "stub_ticks": 0,
                "ticks": 0,
                "ground_dist_m": None,
                "planner_path": None,
                "target_id": None,
                "target_label": None,
                "target_score": None,
                "planner_reason": None,
                "trial_log": None,
                "result": None,
                "detail": None,
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
        with self._lock:
            # Re-arm is an IDLE-only operation. Clearing the latch and
            # stop_event while a mission is still live (nav possibly blocked
            # in an HTTP call, not yet having observed the event) would
            # silently un-fire the e-stop and the car would resume driving.
            if self.current is not None or self._pending is not None:
                raise HTTPException(
                    409, "mission still active/queued — the e-stop can only "
                         "be re-armed from idle; wait for the mission to "
                         "finalize (state ESTOPPED), then retry")
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
            # The latch is authoritative for display: an e-stop triggered
            # while IDLE never passes through a mission finalize, so
            # self.state alone would keep saying READY while /command 503s.
            state = self.state
            if state != "RUNNING" and (self.monitor.triggered
                                       or self.bridge.estopped):
                state = "ESTOPPED"
            return {
                "state": state,
                "not_ready_reasons": list(self.not_ready_reasons),
                "task": dict(self.current) if self.current else None,
                "last_result": dict(self.last_result) if self.last_result else None,
                "estop": self.monitor.status(),
                "bench": self.bench,
                # Server wall clock: clients compute elapsed = now -
                # task.accepted_at without trusting their OWN clock (the
                # console laptop may be skewed vs this PC).
                "now": time.time(),
                # Config truths the console renders (budget bar, arrival
                # line) — served, never hardcoded client-side.
                "limits": {
                    "timeout_rtsm_s": self.cfg.nav.timeout_rtsm_s,
                    "timeout_baseline_s": self.cfg.nav.timeout_baseline_s,
                    "arrival_threshold_m": self.cfg.nav.arrival_threshold_m,
                },
            }

    def preflight_recheck(self) -> Dict[str, Any]:
        """Console "Re-check" — refused while a mission is live/queued:
        the ESP32 probes (2 s-budget GETs to a single-connection Arduino
        WebServer) contend with the drive heartbeat that feeds the 300 ms
        firmware watchdog, and a probe hiccup mid-drive would strand the
        server NOT_READY after a successful trial. Same pattern as
        reset_estop's idle-only guard."""
        with self._lock:
            if self.current is not None or self._pending is not None:
                raise HTTPException(
                    409, "mission live/queued — preflight probes contend "
                         "with the drive heartbeat; retry when idle")
        reasons = self.run_preflight()
        # Derived state, not raw: an IDLE e-stop latch never passes
        # through a mission finalize, so raw state would read READY here
        # while /command 503s ESTOPPED.
        return {"reasons": reasons, "state": self.status()["state"]}

    def trial_log_tail(self, tail: int = 200) -> Dict[str, Any]:
        """Parsed tail of the current (else last) trial's JSONL — the
        console's feedback-loop view (plan record, per-tick pose/dist/
        freshness/commands, verdict). Read-only; only ever serves the path
        the worker itself recorded, never a client-supplied one."""
        with self._lock:
            # First of current/last that HAS a log: a fresh task whose
            # logger isn't up yet (or failed on OSError, or a bench run)
            # must not 404 away the previous trial's perfectly good file.
            task = next((t for t in (self.current, self.last_result)
                         if t and t.get("trial_log")), None)
            path = task.get("trial_log") if task else None
            task_id = task.get("task_id") if task else None
        if not path:
            raise HTTPException(404, "no trial log yet — run a mission first")
        p = Path(path)
        if not p.exists():
            raise HTTPException(404, f"trial log missing on disk: {p.name}")
        tail = max(1, min(int(tail), 2000))
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            raise HTTPException(500, f"trial log unreadable: {e}")
        records = []
        for line in lines[-tail:]:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue                 # in-flight partial line — skip, not fail
        return {"task_id": task_id, "file": p.name,
                "total_lines": len(lines), "records": records}

    # ── single-slot worker ───────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            self._wake.wait(timeout=0.05)
            with self._lock:
                task = self._pending
                self._pending = None
                if task is None:
                    self._wake.clear()
                else:
                    # Atomic handoff: pending -> current under ONE lock hold
                    # (no window where a preempting submit() can miss both),
                    # and stale edge-triggered preempt/cancel signals die at
                    # the task boundary — they always targeted the PREVIOUS
                    # task, and must never insta-kill this one.
                    self.current = task
                    task["phase"] = "planning"
                    prev_state = self.state
                    self.state = "RUNNING"
                    self._preempt.clear()
                    self._cancel.clear()
            if task is not None:
                # E-stop latched between submit-time check and claim (X
                # during the previous task's preempt window keeps _pending):
                # finalize as estopped IMMEDIATELY — no logger, no planner.
                # Otherwise status() shows RUNNING mid-e-stop (hiding the
                # console's ESTOPPED banner) and a phantom trial JSONL for
                # a mission that never ran lands in the paper data.
                if (self.stop_event.is_set() or self.monitor.triggered
                        or self.bridge.estopped):
                    self._finalize_task(task, "estopped",
                                        "e-stop latched before mission start",
                                        prev_state)
                    continue
                # Exception barrier: the single worker thread must survive
                # ANYTHING a task throws — a dead worker would wedge the
                # server in RUNNING while still accepting commands.
                try:
                    self._run_task(task, prev_state)
                except Exception as e:  # noqa: BLE001
                    try:
                        self.bridge.stop()
                    except Exception:  # noqa: BLE001
                        pass
                    self._finalize_task(task, "worker_error",
                                        f"{type(e).__name__}: {e}", prev_state)

    def _run_task(self, task: Dict[str, Any], prev_state: str) -> None:
        """One task = one trial: bench dummy drive, or the real mission
        (planner -> nav -> monitor, trial-logged)."""
        if task["goal"] == BENCH_DUMMY_GOAL:
            result, detail = self._run_bench(task), ""
        elif task["condition"] == "baseline":
            result, detail = self._run_baseline_mission(task)
        else:
            result, detail = self._run_mission(task)
        self._finalize_task(task, result, detail, prev_state)

    def _finalize_task(self, task: Dict[str, Any], result: str, detail: str,
                       prev_state: str) -> None:
        with self._lock:
            task["result"] = result
            task["detail"] = detail
            task["phase"] = "finished"
            self.last_result = dict(task)
            self.current = None
            if result == "estopped":
                self.state = "ESTOPPED"
                # A goal queued BEFORE the e-stop must never auto-drive the
                # car after re-arm — the e-stop abandons EVERYTHING; every
                # post-e-stop motion needs a fresh operator command.
                self._pending = None
            else:
                self.state = "READY" if self._ready else (
                    prev_state if prev_state == "NOT_READY" else "NOT_READY"
                )

    def _run_bench(self, task: Dict[str, Any]) -> str:
        """Gentle constant /drive so the hardware e-stop test has real
        motion to kill (Gate E). Bench-only; preflight bypassed by design."""
        task["phase"] = "bench_running"
        t0 = time.monotonic()
        while not self._shutdown.is_set():
            if self.stop_event.is_set():
                return "estopped"                    # monitor already stopped the car
            if self._preempt.is_set():
                self._preempt.clear()
                self.bridge.stop()                   # safe stop before swap
                return "preempted"
            if self._cancel.is_set():
                self._cancel.clear()
                self.bridge.stop()
                return "cancelled"
            task["stub_ticks"] += 1
            self.bridge.drive(_BENCH_SPEED, _BENCH_SPEED)       # gated internally
            if time.monotonic() - t0 > _BENCH_CAP_S:
                self.bridge.stop()
                return "bench_timeout"
            time.sleep(_TICK_S)
        return "shutdown"

    def _run_mission(self, task: Dict[str, Any]) -> tuple:
        """The Phase-F body: plan once, then closed-loop drive until the
        monitor (or an interrupt) ends the trial. Always trial-logged."""
        t0 = time.monotonic()
        logger: Optional[TrialLogger] = None
        try:
            logger = TrialLogger(self._trials_dir(), task["task_id"],
                                 task["goal"], task["condition"], self.cfg)
            task["trial_log"] = str(logger.path)
        except OSError:
            logger = None                            # never block a mission on disk

        try:
            pr = plan_target(task["goal"], self.rtsm, self.cfg)
        except Exception as e:  # noqa: BLE001 — an RTSM/HTTP hiccup at plan
            # time is a failed TRIAL, never a dead worker thread.
            detail = f"planning failed: {type(e).__name__}: {e}"
            if logger is not None:
                logger.log_end("plan_error", detail,
                               elapsed_s=time.monotonic() - t0)
            return "plan_error", detail
        task["planner_path"] = pr.planner_path
        task["target_id"] = pr.target_id
        task["target_label"] = pr.label
        task["target_score"] = pr.score
        task["planner_reason"] = pr.reason           # Haiku's one-liner (or
        if logger is not None:                       # the not_found detail)
            logger.log_plan(pr, rtsm_stats=self._safe_stats())

        if pr.status != "ok":
            detail = pr.reason or "target not found"
            if logger is not None:
                logger.log_end("not_found", detail,
                               elapsed_s=time.monotonic() - t0)
            return "not_found", detail

        # Symmetric budget semantics: 60 s is a hard TOTAL clock from
        # command receipt — planning/selection time counts here exactly as
        # search time counts in the baseline's 180 s (audited 2026-08-06).
        remaining = self.cfg.nav.timeout_rtsm_s - (time.monotonic() - t0)
        if remaining <= 0:
            detail = "budget exhausted at planning"
            if logger is not None:
                logger.log_end("timeout", detail,
                               elapsed_s=time.monotonic() - t0)
            return "timeout", detail
        task["phase"] = "driving"
        runner = NavRunner(
            self.cfg, self.bridge, self.rtsm, pr, task["condition"],
            stop_event=self.stop_event, preempt_event=self._preempt,
            cancel_event=self._cancel, shutdown_event=self._shutdown,
            logger=logger, progress=task, log_t0_mono=t0,
            timeout_s_override=remaining,
        )
        try:
            result, detail = runner.run()
        except Exception as e:  # noqa: BLE001 — a nav crash must still stop the car
            self.bridge.stop()
            result, detail = "nav_error", f"{type(e).__name__}: {e}"
        if logger is not None:
            logger.log_end(result, detail, elapsed_s=time.monotonic() - t0,
                           rtsm_stats=self._safe_stats())
        return result, detail

    def _run_baseline_mission(self, task: Dict[str, Any]) -> tuple:
        """E1 condition (b): freshness-gated search until the target is
        CURRENTLY visible, then the same closed-loop drive with whatever
        remains of the 180 s budget. One hard total clock — search time is
        the cost of memorylessness and is meant to show up in TTA."""
        t0 = time.monotonic()
        logger: Optional[TrialLogger] = None
        try:
            logger = TrialLogger(self._trials_dir(), task["task_id"],
                                 task["goal"], task["condition"], self.cfg)
            task["trial_log"] = str(logger.path)
        except OSError:
            logger = None

        query = extract_query(task["goal"])
        seed = derive_seed(self.cfg.baseline.rng_seed, task["task_id"])
        task["planner_path"] = "baseline_fresh"
        if logger is not None:
            logger.log_plan(PlanResult(status="searching", goal=task["goal"],
                                       query=query, planner_path="baseline_fresh"),
                            rng_seed=seed, rtsm_stats=self._safe_stats())

        budget = self.cfg.nav.timeout_baseline_s
        # Search cap (2026-08-28): the acquisition phase gets AT MOST
        # search_cap_s of the trial budget; exhausting it concludes NOT
        # FOUND — an explicit, analyzable outcome (every standpoint the
        # observe-then-confirm agent reached was judged; none contained
        # the target) instead of a generic timeout — and the remainder
        # stays reserved for a drive the ~0.04 m/s rig could actually
        # complete. Formally the cap bounds the acquisition phase of BOTH
        # conditions; condition (a)'s acquisition is a single query +
        # selection call, so it only ever binds here.
        cap = budget
        if self.cfg.baseline.search_cap_s > 0:
            cap = min(budget, self.cfg.baseline.search_cap_s)
        task["phase"] = "searching"
        searcher = BaselineSearcher(
            self.cfg, self.bridge, self.rtsm,
            stop_event=self.stop_event, preempt_event=self._preempt,
            cancel_event=self._cancel, shutdown_event=self._shutdown,
            logger=logger, progress=task,
        )
        # Acquire -> select LOOP: when the shared selection rule declares
        # the fresh set contains no plausible match, the search RESUMES
        # (rejected ids masked) instead of settling or ending the trial —
        # a memoryless agent that sees only wrong objects must keep
        # looking. Observed live 2026-08-11 (t20260811-195409-002): the
        # forced pick drove at a smartphone for 160 s on 'teddy bear'.
        rejected: set = set()
        picked = None
        while True:
            try:
                # The searcher's deadline must coincide with the mission
                # clock: logger/plan/selection overhead already spent budget.
                acq = searcher.acquire(query, task["task_id"],
                                       max(0.0, cap - (time.monotonic() - t0)),
                                       exclude_ids=frozenset(rejected))
            except Exception as e:  # noqa: BLE001 — a search crash must stop the car
                self.bridge.stop()
                result, detail = "search_error", f"{type(e).__name__}: {e}"
                if logger is not None:
                    logger.log_end(result, detail,
                                   elapsed_s=time.monotonic() - t0)
                return result, detail

            if acq.status != "acquired":
                result = acq.status
                detail = acq.detail or f"search ended after {acq.sweeps} sweeps"
                # NOT FOUND is a substantive conclusion — it requires that
                # standpoints were actually queried/judged. A retrieval
                # outage that produced zero successful round queries must
                # stay a plain timeout (round_query_failed events in the
                # trial log carry the fault), or the aggregate would code
                # an infrastructure failure as "target absent".
                if (acq.status == "timeout" and cap < budget
                        and (acq.rounds_queried > 0 or rejected)):
                    result = "not_found"
                    detail = (f"search cap {cap:.0f}s exhausted -> concluded "
                              f"not found ({acq.sweeps} sweeps, "
                              f"{acq.rounds_queried} rounds queried, "
                              f"{acq.query_failures} query failures)")
                break
            # SAME selection rule as condition (a), applied over the
            # freshness-gated (currently visible) set — the comparison
            # masks persistence, never target-selection intelligence.
            sel = select_target_from_hits(list(acq.hits), task["goal"],
                                          self.rtsm, self.cfg)
            if sel is None:                          # defensive: gate always
                result = "not_found"                 # requires xyz_world
                detail = "no eligible fresh candidate"
                if logger is not None:
                    logger.log_end(result, detail,
                                   elapsed_s=time.monotonic() - t0)
                return result, detail
            picked, sel_path, sel_reason = sel
            if picked is None:                       # LLM: nothing plausible
                ids = [h.id for h in acq.hits]
                rejected.update(ids)
                if logger is not None:
                    logger.log_event(
                        "baseline_no_match", time.monotonic() - t0,
                        rejected_ids=ids, reason=sel_reason,
                        sweeps=acq.sweeps,
                        search_time_s=round(acq.elapsed_s, 3))
                if time.monotonic() - t0 >= cap:
                    if cap < budget:
                        result, detail = "not_found", (
                            f"search cap {cap:.0f}s exhausted -> concluded "
                            "not found (last standpoint judged no-match)")
                    else:
                        result, detail = "timeout", "budget exhausted in search"
                    break
                continue                             # relocate + next round
            break                                    # real pick -> drive

        if picked is not None:
            planner_path = ("baseline_fresh_haiku" if sel_path == "haiku"
                            else "baseline_fresh_top1")
            task["planner_path"] = planner_path
            task["target_id"] = picked.id
            task["target_label"] = picked.label
            task["target_score"] = picked.score
            task["planner_reason"] = sel_reason
            if logger is not None:
                logger.log_event(
                    "baseline_acquired", time.monotonic() - t0,
                    target_id=picked.id, label=picked.label,
                    xyz_world=picked.xyz_world,
                    pose=TrialLogger._pose_dict(acq.pose),
                    hit_age_s=(round(acq.hit_age_s, 3)
                               if acq.hit_age_s is not None else None),
                    # The staleness audit must describe the PICKED
                    # candidate, not the top-ranked one — under the round
                    # window they can differ by minutes (2026-08-28).
                    target_last_seen_age_s=(
                        round(time.time() - picked.last_seen_wall_utc, 3)
                        if picked.last_seen_wall_utc is not None else None),
                    n_fresh=len(acq.hits), planner_path=planner_path,
                    reason=sel_reason, sweeps=acq.sweeps,
                    search_time_s=round(acq.elapsed_s, 3))
            remaining = budget - (time.monotonic() - t0)
            if remaining <= 0:
                result, detail = "timeout", "budget exhausted at acquisition"
            else:
                pr = PlanResult(
                    status="ok", goal=task["goal"], query=query,
                    target_id=picked.id, label=picked.label,
                    xyz_world=picked.xyz_world,
                    score=picked.score, confirmed=picked.confirmed,
                    stability=picked.stability, planner_path=planner_path,
                    plan_pose=acq.pose,
                    frame_epoch=acq.pose.frame_epoch if acq.pose else None,
                )
                task["phase"] = "driving"
                runner = NavRunner(
                    self.cfg, self.bridge, self.rtsm, pr, "baseline",
                    stop_event=self.stop_event, preempt_event=self._preempt,
                    cancel_event=self._cancel, shutdown_event=self._shutdown,
                    logger=logger, progress=task,
                    timeout_s_override=remaining, log_t0_mono=t0,
                )
                try:
                    result, detail = runner.run()
                except Exception as e:  # noqa: BLE001
                    self.bridge.stop()
                    result, detail = "nav_error", f"{type(e).__name__}: {e}"

        if logger is not None:
            logger.log_end(result, detail, elapsed_s=time.monotonic() - t0,
                           rtsm_stats=self._safe_stats())
        return result, detail

    def _trials_dir(self) -> Path:
        p = Path(self.cfg.trials_output_dir)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        return p

    def _safe_stats(self) -> Optional[Dict[str, Any]]:
        """Best-effort RTSM /stats snapshot for trial-log throughput
        auditing (upserts delta start->end per trial). Never raises."""
        try:
            s = self.rtsm.stats()
            return {"objects": s.get("objects"), "confirmed": s.get("confirmed"),
                    "upserts_total": s.get("upserts_total")}
        except Exception:  # noqa: BLE001
            return None


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
        import asyncio

        srv.startup()
        # MAIN-THREAD SDL pump: Windows DirectInput delivers joystick
        # input only to the thread that initialized SDL (found live
        # 2026-08-07). startup() bound the pads on THIS thread; this task
        # keeps input flowing so the e-stop poll thread reads real state.
        async def _sdl_pump():
            while True:
                srv.monitor.pump_once()
                await asyncio.sleep(0.05)

        pump_task = asyncio.create_task(_sdl_pump())
        try:
            yield
        finally:
            pump_task.cancel()
            srv.shutdown()

    app = FastAPI(title="rc_car_agent", lifespan=lifespan)
    app.state.srv = srv                              # test access

    def _auth(request: Request) -> None:
        """Shared-secret gate for every POST when serving the LAN. GETs
        stay open (read-only telemetry); anything that commands, stops,
        re-arms, or probes hardware needs the token."""
        token = cfg.server.api_token
        if token and request.headers.get("x-auth-token") != token:
            raise HTTPException(401, "missing/invalid X-Auth-Token")

    @app.post("/command")
    def command(body: CommandBody, request: Request):
        _auth(request)
        return srv.submit(body.goal, body.condition)

    @app.get("/status")
    def status():
        return srv.status()

    @app.post("/cancel")
    def cancel(request: Request):
        _auth(request)
        return srv.cancel()

    @app.post("/stop")
    def stop(request: Request):
        _auth(request)
        return srv.soft_stop()

    @app.post("/reset_estop")
    def reset_estop(request: Request):
        _auth(request)
        return srv.reset_estop()

    @app.post("/preflight")
    def preflight(request: Request):
        _auth(request)
        return srv.preflight_recheck()

    @app.get("/trial_log")
    def trial_log(tail: int = 200):
        return srv.trial_log_tail(tail)

    ui_path = Path(__file__).resolve().parent / "ui.html"

    @app.get("/", include_in_schema=False)
    @app.get("/ui")
    def ui():
        if not ui_path.exists():
            raise HTTPException(404, "ui.html missing next to server.py")
        return FileResponse(ui_path, media_type="text/html")

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
