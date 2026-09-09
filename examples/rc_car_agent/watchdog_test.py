"""Isolated Gate-C watchdog test: 1.5 s forward, then 5 s SILENCE.
The wheels' behavior in the silent window IS the result:
  PASS: stop on their own within ~0.3 s of silence, stay stopped.
  FAIL: keep spinning into the silent window.
Hygiene stop is sent only AFTER the window (announced)."""

import time

from config import load_config
from esp32_bridge import Esp32Bridge

cfg = load_config()
b = Esp32Bridge(cfg.esp32.url, drive_rate_hz=cfg.esp32.drive_rate_hz,
                heartbeat_s=cfg.esp32.heartbeat_s,
                change_epsilon=cfg.esp32.change_epsilon,
                http_timeout_s=cfg.esp32.http_timeout_s)
print("battery:", b.battery_mv(), "mV")
print("[%s] FORWARD at 0.5 for 1.5 s..." % time.strftime("%H:%M:%S"))
t0 = time.monotonic()
n = 0
while time.monotonic() - t0 < 1.5:
    if b.drive(0.5, 0.5):
        n += 1
    time.sleep(0.05)
print("[%s] SILENCE NOW (sent %d cmds). Watch: wheels must stop within ~0.3 s "
      "and STAY stopped for the whole 5 s window." % (time.strftime("%H:%M:%S"), n))
time.sleep(5.0)
print("[%s] window over -> hygiene stop (should change nothing if watchdog worked)"
      % time.strftime("%H:%M:%S"))
b.stop()
print("battery:", b.battery_mv(), "mV  failures:", b.consecutive_failures)
