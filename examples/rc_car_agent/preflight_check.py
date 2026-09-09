"""
Standalone liveness / preflight check — the embryo of server.py's preflight.

Run from the agent venv:
    .venv/Scripts/python.exe preflight_check.py

Checks (read-only, sends NO motion commands):
  RTSM : /healthz alive · /stats parses · robot_pose present+fresh · map non-empty
  ESP32: ping (GET /) · battery voltage above threshold

Exit code 0 = READY, 1 = NOT_READY (any check failed).
"""

from __future__ import annotations

import sys
import time

from config import load_config
from esp32_bridge import Esp32Bridge
from rtsm_client import RtsmClient

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def main() -> int:
    cfg = load_config()
    rtsm = RtsmClient(cfg.rtsm.url)
    bridge = Esp32Bridge(cfg.esp32.url, http_timeout_s=cfg.esp32.http_timeout_s)
    failures = 0

    print(f"— RTSM @ {cfg.rtsm.url}")
    if rtsm.healthz():
        print(f"  {OK} /healthz alive")
        try:
            stats = rtsm.stats()
            n_obj = int(stats.get("objects", 0))
            print(f"  {OK} /stats parses (objects={n_obj}, confirmed={stats.get('confirmed', 0)})")
            pose = rtsm.get_robot_pose()
            if pose is None:
                print(f"  {WARN} robot_pose=None — no frames received yet (is the iPhone streaming?)")
                failures += 1
            else:
                age_hint = time.time() - pose.timestamp
                print(f"  {OK} robot_pose present: xyz={[round(v, 3) for v in pose.xyz]} "
                      f"(sender-ts age ≈ {age_hint:.1f}s — cross-clock, indicative only)")
            # Receive-time clearance is opt-in on the RTSM side
            # (io.clearance.enable). Key absent = RTSM started without it:
            # nav's drive guard is fail-open and the relocation walk is
            # fail-closed, so the mission would run blind -- refuse.
            if "forward_clearance" not in stats:
                print(f"  {BAD} /stats has no forward_clearance -- restart RTSM with "
                      f"`--set io.clearance.enable=true` (drive guard + relocation walk need it)")
                failures += 1
            elif stats.get("forward_clearance") is None:
                print(f"  {WARN} forward_clearance=None -- no depth frames received yet")
            else:
                c = stats["forward_clearance"]
                print(f"  {OK} forward_clearance present: {float(c.get('clearance_m', 0.0)):.2f} m "
                      f"(valid_frac {float(c.get('valid_frac', 0.0)):.2f})")
            if n_obj == 0:
                print(f"  {WARN} map empty — scan the room before issuing goals")
        except Exception as e:  # noqa: BLE001 — preflight reports, never crashes
            print(f"  {BAD} /stats failed: {e}")
            failures += 1
    else:
        print(f"  {BAD} /healthz unreachable — start RTSM "
              f"(`python -m rtsm --set io.clearance.enable=true` in the GPU env)")
        failures += 1

    print(f"— ESP32 @ {cfg.esp32.url}")
    banner = bridge.ping()
    if banner is None:
        print(f"  {BAD} ping failed — is the car powered on and on this WiFi?")
        failures += 1
    else:
        print(f"  {OK} ping: {banner.strip().splitlines()[0]}")
        mv = bridge.battery_mv()
        if mv is None:
            print(f"  {BAD} /battery unreadable")
            failures += 1
        elif mv < cfg.esp32.battery_min_mv:
            print(f"  {BAD} battery {mv} mV < minimum {cfg.esp32.battery_min_mv} mV — charge first")
            failures += 1
        else:
            print(f"  {OK} battery {mv} mV")

    verdict = "READY" if failures == 0 else f"NOT_READY ({failures} check(s) failed)"
    print(f"\n=> {verdict}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
