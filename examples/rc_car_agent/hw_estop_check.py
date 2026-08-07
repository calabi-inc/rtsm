"""
Live-fire hardware validation of the e-stop gamepad channel.

Run on the rig desktop with the real PS4 controller after ANY estop.py
change, and whenever the E1 protocol calls for a live-fire check:

    .venv/Scripts/python.exe hw_estop_check.py

Runs the REAL EstopMonitor against a no-op bridge — no ESP32 traffic, no
motion, safe with the car powered on or off. Run it standalone (agent
server stopped) so the binding under test is this process's own.

Walk the sequence the 2026-08-07 incident proved can go deaf:

    1. pad connected & awake        -> expect available=True
    2. PRESS X                      -> expect TRIGGERED + verified=True
    3. put the pad to sleep (hold PS ~10 s) or power it off
                                    -> expect available=False within ~1 s
    4. wake / reconnect the pad     -> expect available=True, verified=False
    5. PRESS X again                -> expect TRIGGERED + verified=True
                                       (THE step the old retry-loop
                                        binding failed silently)

PASS = every expectation observed. A mocked unit test cannot substitute:
39d7ec7's mocked test was green while the live button was dead. Ctrl-C
to exit.
"""

from __future__ import annotations

import sys
import threading
import time

from estop import EstopMonitor


class NoopBridge:
    """Absorbs the latch calls; nothing on the wire, nothing moves."""

    def __init__(self):
        self.stops = 0

    def stop(self, estop=False):
        self.stops += 1
        return True


def main() -> int:
    bridge = NoopBridge()
    stop_event = threading.Event()
    mon = EstopMonitor(bridge, stop_event).start()

    print(__doc__)
    print("Watching (5 Hz). Status line updates in place; transitions and "
          "triggers print on their own lines.\n")

    triggers = 0
    last = None
    try:
        while True:
            s = mon.status()
            key = (s["gamepad_available"], s["binding_verified"],
                   s["triggered"])
            if s["triggered"]:
                triggers += 1
                print(f"\n>>> TRIGGERED #{triggers}  source={s['trigger_source']}"
                      f"  verified={s['binding_verified']}  — re-arming")
                mon.reset()
                stop_event.clear()
            elif key != last:
                print(f"\n--- available={s['gamepad_available']}"
                      f"  name={s['gamepad_name']}"
                      f"  verified={s['binding_verified']}")
            last = key
            age = ("never" if s["last_button_press_mono"] is None else
                   f"{time.monotonic() - s['last_button_press_mono']:.1f}s ago")
            print(f"\ravailable={s['gamepad_available']!s:5}  "
                  f"verified={s['binding_verified']!s:5}  "
                  f"last_press={age:10}  triggers={triggers}   ",
                  end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print(f"\n\nDone. triggers={triggers}, bridge latch calls={bridge.stops}.")
        print("PASS requires: trigger+verified BOTH before AND after a "
              "sleep/reconnect cycle (steps 2 and 5).")
        mon.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
