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
DEFAULT_RATE = 20          # Hz — must be > 1000/TELEOP_WATCHDOG_MS in firmware (~3 Hz min)
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
    print(f"Press X to stop and exit, Options to exit gracefully.\n")

    session = requests.Session()
    period = 1.0 / args.rate
    last_print = 0.0

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

            # Send /drive command. Even when stick is centered we send (0, 0)
            # so the watchdog doesn't trigger a stop mid-stream — keeps motors
            # smoothly responsive when you nudge the stick again.
            try:
                session.post(f"{args.url}/drive",
                             json={"left": left, "right": right},
                             timeout=0.2)
            except requests.RequestException as e:
                # Drop the frame, try next tick. Watchdog will stop the car
                # if these failures continue.
                print(f"  (drop frame: {e})")

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
                session.post(f"{args.url}/stop", timeout=1)
                return 0
            if joy.get_button(7):
                print("\nOptions pressed — exiting (motors will stop via watchdog).")
                session.post(f"{args.url}/stop", timeout=1)
                return 0

            # Tick pacing
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nCtrl-C — stopping and exiting.")
        try:
            session.post(f"{args.url}/stop", timeout=1)
        except requests.RequestException:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
