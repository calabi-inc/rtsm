"""
Gate-C bench test — CAR ON BLOCKS, WHEELS OFF THE GROUND.

Drives the wheels through the real Esp32Bridge (send-gating + heartbeat,
exactly like nav will), so this doubles as the bridge's first hardware run.

    .venv/Scripts/python.exe spin_test.py

Sequence:
  1. SPIN     ~2 s forward at 0.5, then MID-MOTION stop()  -> instant halt
  2. WATCHDOG ~1.5 s forward at 0.5, then SILENCE          -> self-stop <=0.3 s
  3. DIRECTION ~1.2 s left-forward / right-backward, stop  -> CW pivot pattern
"""

from __future__ import annotations

import time

from config import load_config
from esp32_bridge import Esp32Bridge


def stream(bridge: Esp32Bridge, left: float, right: float, seconds: float) -> int:
    """Call drive() every 50 ms (bridge gates the wire itself). Returns sends."""
    sends = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if bridge.drive(left, right):
            sends += 1
        time.sleep(0.05)
    return sends


def main() -> int:
    cfg = load_config()
    b = Esp32Bridge(
        cfg.esp32.url,
        drive_rate_hz=cfg.esp32.drive_rate_hz,
        heartbeat_s=cfg.esp32.heartbeat_s,
        change_epsilon=cfg.esp32.change_epsilon,
        http_timeout_s=cfg.esp32.http_timeout_s,
    )
    banner = b.ping()
    if banner is None:
        print("FAIL: ESP32 unreachable")
        return 1
    print(f"ESP32: {banner.strip().splitlines()[0]}  battery={b.battery_mv()} mV")
    input(">> Car on blocks, wheels free? Press Enter to start test 1 (or Ctrl-C)...")

    print("\n=== TEST 1: SPIN ~2 s at 0.5, then MID-MOTION stop() ===")
    n = stream(b, 0.5, 0.5, 2.0)
    ok = b.stop()
    print(f"  wire sends={n}, mid-motion stop ok={ok} -> wheels must have halted instantly")
    time.sleep(1.5)

    print("=== TEST 2: WATCHDOG - ~1.5 s at 0.5, then SILENCE (no stop) ===")
    n = stream(b, 0.5, 0.5, 1.5)
    t = time.strftime("%H:%M:%S")
    print(f"  wire sends={n}, last command ~{t} -> SILENCE; wheels must self-stop within ~0.3 s")
    time.sleep(2.5)

    print("=== TEST 3: DIRECTION - left FWD / right BACK ~1.2 s, then stop ===")
    n = stream(b, 0.5, -0.5, 1.2)
    b.stop()
    print(f"  wire sends={n}, stopped. Expected: left wheels forward, right wheels backward (CW pivot)")

    print(f"\nalive: battery={b.battery_mv()} mV, failures={b.consecutive_failures}")
    print("Report what you saw for tests 1/2/3.")
    return 0


if __name__ == "__main__":
    main()
