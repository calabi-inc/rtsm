"""
Gamepad teleop for the RTSM RC car.

Reads a gamepad (PS4, Xbox, etc.) via pygame, streams left/right speed
commands to the ESP32's /drive endpoint at ~20 Hz. The ESP32 has a
300 ms watchdog — if this script crashes or you walk out of WiFi range,
the car stops on its own.

Default control scheme (arcade-style with left stick):
    Left stick Y     →  throttle (forward / reverse)
    Left stick X     →  steering (mix into left / right wheels)
    X button         →  emergency stop, then exit
    Options / Start  →  exit gracefully

Setup:
    pip install pygame requests

Usage:
    python teleop_gamepad.py
    python teleop_gamepad.py --url http://192.168.1.189 --rate 20

Tested with a Sony DualShock 4 over Bluetooth on Windows 11.
Should work with Xbox / Switch Pro / 8BitDo controllers too — only the
button indices may need tweaking (see PS4 mapping notes at bottom).
"""

from __future__ import annotations

import argparse
import sys
import time

import pygame
import requests

# ===== Defaults =====
DEFAULT_URL  = "http://192.168.1.189"
DEFAULT_RATE = 10          # Hz — must be > 1000/TELEOP_WATCHDOG_MS in firmware (~3 Hz min)
                           # Kept low because Arduino's WebServer has no HTTP keep-alive;
                           # every request opens a new TCP connection. 20 Hz overwhelms
                           # the ESP32's connection pool with sockets in TIME_WAIT.
DEFAULT_TIMEOUT = 0.6      # seconds — generous enough to ride out WiFi retransmits
DEADZONE     = 0.15        # ignore stick drift below this magnitude
# ===================


def apply_deadzone(value: float, dz: float = DEADZONE) -> float:
    """Smoothly zero out small stick deflections around the rest position."""
    if abs(value) < dz:
        return 0.0
    # Scale the remaining range so output goes 0..1 instead of dz..1
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def arcade_mix(throttle: float, steer: float) -> tuple[float, float]:
    """
    Map throttle (forward/back) + steer (left/right) to left/right wheel speeds.
    Both inputs are -1.0..1.0. Returns (left, right) in the same range.

    Arcade mixing is the classic RC car / tank control: forward stick goes
    straight, sideways stick rotates in place, diagonals do the obvious thing.
    """
    left  = throttle + steer
    right = throttle - steer
    # Normalize so a hard diagonal (1,1) doesn't ask for 2.0
    peak = max(abs(left), abs(right), 1.0)
    return (left / peak, right / peak)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gamepad teleop for the RTSM RC car")
    parser.add_argument("--url",  default=DEFAULT_URL,  help="ESP32 base URL")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Polling rate in Hz")
    parser.add_argument("--invert-y", action="store_true",
                        help="Invert throttle if your stick reads opposite")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Pair your controller first (see README).",
              file=sys.stderr)
        return 1

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Using controller: {joy.get_name()}")
    print(f"ESP32 URL:        {args.url}")
    print(f"Polling rate:     {args.rate} Hz")

    # Pre-flight: confirm ESP32 is reachable BEFORE entering the joystick loop.
    # Fail fast with a clear message instead of running into hundreds of
    # silent timeouts (which is what happens if the ESP32 is rebooting, the
    # battery is too low, or we're on the wrong WiFi).
    print(f"\nPinging ESP32...", end=" ", flush=True)
    try:
        r = requests.get(args.url + "/", timeout=2.0,
                         headers={"Connection": "close"})
        print(f"OK\n{r.text.strip()}\n")
    except requests.RequestException as e:
        print(f"FAILED")
        print(f"  {str(e).split(chr(10), 1)[0]}")
        print(f"\nThe ESP32 isn't responding. Check:")
        print(f"  1. Is the ESP32 powered? (USB-C plugged in?)")
        print(f"  2. Is the IP correct? (Check Serial Monitor for 'Connected! IP: ...')")
        print(f"  3. Is the battery healthy? (Boot output should show 7000+ mV)")
        print(f"  4. Are PC and ESP32 on the same WiFi network?")
        return 1

    print(f"Press X to stop and exit, Options to exit gracefully.\n")

    # Deliberately NOT using requests.Session() — Arduino WebServer closes
    # each TCP connection after the response, so caching connections in a
    # pool just leads to stale sockets that fail mysteriously on the next
    # request. Open-and-close per request is the right model for this stack.
    HEADERS = {"Connection": "close"}

    period = 1.0 / args.rate
    last_print = 0.0
    last_sent_left = None
    last_sent_right = None
    last_sent_time = 0.0
    consecutive_failures = 0

    # Don't spam the ESP32. Only send /drive when:
    #   (a) the commanded speed changed by more than CHANGE_EPS, or
    #   (b) it's been more than HEARTBEAT_S since the last send
    #       (so the watchdog stays fed when holding the stick steady).
    CHANGE_EPS  = 0.04   # ~4 % of full speed
    HEARTBEAT_S = 0.25   # well under firmware's 0.3 s watchdog window

    try:
        while True:
            loop_start = time.monotonic()
            pygame.event.pump()

            # Left stick: axis 0 = X (left/right), axis 1 = Y (up/down, -1 = up)
            raw_steer    = joy.get_axis(0)
            raw_throttle = -joy.get_axis(1)  # invert so up = forward
            if args.invert_y:
                raw_throttle = -raw_throttle

            throttle = apply_deadzone(raw_throttle)
            steer    = apply_deadzone(raw_steer)
            left, right = arcade_mix(throttle, steer)

            # Decide whether to actually send this tick.
            now = time.monotonic()
            speeds_changed = (
                last_sent_left is None
                or abs(left  - last_sent_left)  > CHANGE_EPS
                or abs(right - last_sent_right) > CHANGE_EPS
            )
            heartbeat_due = (now - last_sent_time) > HEARTBEAT_S

            if speeds_changed or heartbeat_due:
                try:
                    requests.post(f"{args.url}/drive",
                                  json={"left": left, "right": right},
                                  timeout=DEFAULT_TIMEOUT,
                                  headers=HEADERS)
                    last_sent_left  = left
                    last_sent_right = right
                    last_sent_time  = now
                    consecutive_failures = 0
                except requests.RequestException as e:
                    # Drop the frame, try next tick. Watchdog will stop the car
                    # if these failures continue.
                    consecutive_failures += 1
                    # Compact one-line error (the full stack trace is noise).
                    err = str(e).split("\n", 1)[0][:120]
                    print(f"  (drop frame #{consecutive_failures}: {err})")

                    if consecutive_failures >= 10:
                        print(f"  >> {consecutive_failures} failures in a row.")
                        print(f"  >> The ESP32 may have crashed or browned out.")
                        print(f"  >> Check Serial Monitor and battery voltage.")
                        print(f"  >> Pausing 2 s then continuing...")
                        time.sleep(2.0)
                        consecutive_failures = 0

            # Lightweight terminal feedback once per second
            now = time.monotonic()
            if now - last_print > 1.0:
                print(f"  throttle={throttle:+.2f}  steer={steer:+.2f}  "
                      f"L={left:+.2f}  R={right:+.2f}")
                last_print = now

            # Button handling — exit conditions
            # PS4 button indices on Windows (pygame): 0=X, 1=Circle, 2=Square,
            # 3=Triangle, 6=Share, 7=Options, 8=PS, 9=L3, 10=R3
            if joy.get_button(0):
                print("\nX pressed — stopping and exiting.")
                try:
                    requests.post(f"{args.url}/stop", timeout=1, headers=HEADERS)
                except requests.RequestException:
                    pass
                return 0
            if joy.get_button(7):
                print("\nOptions pressed — exiting (motors will stop via watchdog).")
                try:
                    requests.post(f"{args.url}/stop", timeout=1, headers=HEADERS)
                except requests.RequestException:
                    pass
                return 0

            # Tick pacing
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nCtrl-C — stopping and exiting.")
        try:
            requests.post(f"{args.url}/stop", timeout=1, headers=HEADERS)
        except requests.RequestException:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
