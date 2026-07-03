# RTSM RC Car Agent

Python code that drives the Hiwonder 4WD chassis from a PC.
The ESP32 firmware in `examples/esp32_firmware/motor_controller/` is the
actuator side; everything here is the brains.

Two entry points so far:

| Script | What it does |
|---|---|
| `teleop_gamepad.py` | Manual control via PS4/Xbox gamepad over Bluetooth. Used for the "without RTSM" half of the demo recording and for room scanning. |
| `run.py` (TODO) | Autonomous: queries RTSM for a target object, plans heading + distance, drives the car there. |

---

## Quick start — gamepad teleop

```bash
# One-time setup
pip install pygame requests

# Run (uses the ESP32 IP from your boot log)
python teleop_gamepad.py --url http://192.168.1.189
```

Controls (PS4 layout):

| Input | Action |
|---|---|
| Left stick up/down | Throttle (forward/back) |
| Left stick left/right | Steering (mixed into wheel speeds) |
| X (cross) button | Emergency stop, then exit |
| Options button | Graceful exit |
| Ctrl-C in terminal | Graceful exit |

The script streams `{"left": -1..1, "right": -1..1}` to `/drive` at 20 Hz.
The ESP32 has a 300 ms watchdog — if this script dies, you walk out of WiFi
range, or the PC freezes, the car stops on its own within a third of a second.

---

## Pairing a PS4 controller to Windows over Bluetooth

The DualShock 4 enumerates as a generic Windows gamepad (HID), no driver
install required. Pygame reads it directly.

### Steps

1. **Put the controller in pairing mode.**
   Hold the **PS button** and the **Share button** simultaneously for about
   3 seconds. The lightbar on the back of the controller will start flashing
   white quickly (rapid double-flash).
2. **Windows: Settings → Bluetooth & devices → Add device.**
3. Click **Bluetooth** in the popup.
4. After a few seconds, **Wireless Controller** appears in the list. Click it.
5. Windows pairs and the lightbar goes solid (blue on a fresh controller).
   Done.

### Verifying

Open the Windows Game Controllers utility:

```powershell
joy.cpl
```

You should see **Wireless Controller** with status **OK**. Click **Properties**
to confirm sticks and buttons respond. If buttons look right but axes look
wrong (e.g., stick rests at 0.3 instead of 0), it's drift — `teleop_gamepad.py`
applies a deadzone of 0.15 to compensate.

### Troubleshooting

| Problem | Fix |
|---|---|
| Lightbar never flashes | Battery dead — plug in USB cable for 10 minutes to charge |
| Windows finds the controller but pairing fails | Forget any old "Wireless Controller" entries in Bluetooth settings, then re-pair |
| Pygame says "No gamepad detected" but Windows shows it works | Restart the script; pygame only enumerates joysticks at startup |
| Controller pairs, drops every few seconds | Move closer to PC, or check for 2.4 GHz interference (your ESP32 is also 2.4 GHz — but they shouldn't fight) |
| Sticks read full deflection at rest | Recalibrate in `joy.cpl` → Properties → Settings → Calibrate |
| You want USB instead | Plug the controller in with a USB-A or USB-C data cable (not a power-only cable). Same script works unchanged. |

### Battery & disconnect

- Charging: USB-A on the PC (or any USB-C 5 V source). Lightbar pulses yellow while charging.
- Disconnect cleanly: hold **PS button** for ~10 seconds. Otherwise it auto-sleeps after 10 min idle.

### Reconnecting after first pair

After the initial pair, the controller and Windows remember each other.
To reconnect later:
1. Power on the controller with a single quick press of **PS button**.
2. Lightbar pulses, then goes solid blue — connected.

No need to re-pair unless you hit "Forget" in Windows settings.

---

## Button indices reference (pygame on Windows, DualShock 4)

If you want to customize the script:

```
button 0  = X (cross)
button 1  = Circle
button 2  = Square
button 3  = Triangle
button 4  = L1
button 5  = R1
button 6  = Share
button 7  = Options
button 8  = PS
button 9  = L3 (left stick click)
button 10 = R3 (right stick click)

axis 0 = Left stick X    (-1 = left, +1 = right)
axis 1 = Left stick Y    (-1 = up,   +1 = down)
axis 2 = Right stick X
axis 3 = Right stick Y
axis 4 = L2 (-1 released, +1 fully pressed)
axis 5 = R2 (-1 released, +1 fully pressed)
```

**Xbox controllers** use the same axis layout but button indices differ
slightly (A=0, B=1, X=2, Y=3). The script's stick reading works unchanged;
only the exit buttons would shift.

---

## Architecture

```
DualShock 4 ──Bluetooth──► PC ──WiFi──► ESP32 ──I²C──► Motors
              (pygame)       (HTTP)      (driveLeftRight)
```

PC handles all input processing and arcade mixing. ESP32 just receives
`left`/`right` speed commands (-1..1) and applies them with the
calibrated `TELEOP_MAX_SPEED` constant.

**Why not pair PS4 directly to ESP32?**
ESP32 does support Bluetooth Classic and there's a [Bluepad32] library
that pairs DualShock controllers, but:

* It adds ~200 lines of firmware and ~50 KB of RAM.
* WiFi + Bluetooth simultaneously on the ESP32 shares one radio — works but
  occasionally hiccups.
* The PC is already running for RTSM perception during the demo, so the
  Bluetooth-to-PC hop costs nothing.

If you want pure-mobile teleop with no PC in the loop, see [Bluepad32].

[Bluepad32]: https://github.com/ricardoquesada/bluepad32

---

## Files

The autonomous agent is a **persistent server** (`POST /command {"goal": "go to the red mug"}`),
built in audited phases per `code-plan-demo2-rc-car.md`. It consumes RTSM via REST only.

```
rc_car_agent/
├── teleop_gamepad.py     # gamepad → PC → /drive (manual teleop; baseline video + e-stop pattern)
├── config.yaml           # ✅ every tunable: URLs, thresholds, timeouts, calibration constants
├── config.py             # ✅ typed, validated loader
├── geometry.py           # ✅ Tier-1 math: X–Z ground plane, yaw-from-quat, camera→car transform
├── tests/                # ✅ 18 unit tests (geometry = the sign/axis bug zone + config smoke)
├── rtsm_client.py        # Phase B: /search/semantic + /stats pose poll
├── esp32_bridge.py       # Phase B: send-gated /drive + /stop client
├── planner.py            # Phase D: Haiku forced-tool target pick (top-1 fallback)
├── estop.py              # Phase E: independent e-stop thread (gamepad + keyboard)
├── server.py             # Phase E: FastAPI agent server + single-slot worker (entry point)
├── nav.py                # Phase F: closed-loop X–Z re-aim @ pose freshness
├── monitor.py            # Phase F: sole arrival authority + staleness/frozen/frame-epoch gates
├── calibrate.py          # Phase G: auto-derive camera→car constants + acceptance gate
├── baseline_search.py    # Phase H: memoryless E1 condition (freshness-gated)
└── trial_logger.py       # Phase F: per-trial JSONL → paper/demo2_data/
```

Run the tests: `python -m pytest examples/rc_car_agent/tests/ -v`

**Geometry conventions (pinned — see `geometry.py` docstring):** ARKit world is **Y-up**
→ ground plane is **X–Z**; RTSM serves the camera quaternion in **OpenCV convention**
(forward = `R(q)·[0,0,1]`); yaw = `atan2(x, z)`, **positive = CCW from above = left**,
matching the firmware's `+angle = CCW`.
