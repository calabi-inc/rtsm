# ESP32 Motor Controller — RTSM Demo 2 RC Car

ESP32 firmware that drives a **Hiwonder Large Metal 4WD chassis** over WiFi.
The PC streams wheel velocities (`/drive`) — the ESP32 talks I²C to Hiwonder's
4-channel encoder motor driver and stops the motors on its own if commands
cease for 300 ms (watchdog).

**Single-mode firmware (2026-07):** the old open-loop `/forward` + `/turn`
timed-move endpoints were removed — they blocked `loop()` and suspended the
watchdog exactly while the car was moving. The desktop agent closes the loop
against live ARKit pose and speaks continuous wheel velocities only.

This is the actuator half of the RTSM Demo 2 stack. The agent (Python) lives
in `examples/rc_car_agent/` and is responsible for vision + planning.

---

## Hardware

| Part | Notes |
|---|---|
| ESP32 dev board (ESP-WROOM-32) | Any 30-pin variant. Tested with ELEGOO USB-C and generic micro-USB boards. |
| Hiwonder Large Metal 4WD chassis | Includes 4-ch encoder motor driver board, 4× JGB37 motors, mecanum wheels |
| 4-pin Molex 5264 → Dupont cable | I²C link from controller board to ESP32 (DIY or buy ready-made) |
| Battery, 6–8 V | Powers the motors via the Hiwonder driver. **Not** the ESP32. |
| USB-C / micro-USB cable | Programming and power for ESP32 during dev |

### Wiring

ESP32 ↔ Hiwonder controller board (4 wires):

| Hiwonder pin | ESP32 pin |
|---|---|
| SDA | GPIO 21 |
| SCL | GPIO 22 |
| 5V  | VIN (or 5V) |
| GND | GND |

Motors plug into M1–M4 ports on the Hiwonder board with the included 6-pin
cables. Motor battery connects to the controller's separate power input —
**do not power motors from the ESP32's 5V pin.**

---

## Setup

### 1. Arduino IDE

Install **Arduino IDE 2.x**, then add ESP32 board support:

1. **File → Preferences → Additional Boards Manager URLs:**
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
2. **Tools → Board → Boards Manager → search "esp32" → install** (Espressif Systems package, ~300 MB)
3. **Tools → Board → ESP32 Dev Module**
4. **Tools → Port → COM_x_** (whichever your ESP32 enumerates as)

### 2. USB driver (Windows)

If Device Manager shows the ESP32 with a yellow warning under "Other devices":

- **CP2102 chip:** [Silicon Labs CP210x driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
- **CH340 chip:** [WCH CH341SER driver](http://www.wch-ic.com/downloads/CH341SER_EXE.html)

After install, unplug → wait 10 s → replug. Device should appear under
"Ports (COM & LPT)" with no warning.

### 3. Secrets

Copy the template and fill in your WiFi credentials:

```bash
cp secrets.h.example secrets.h
# edit secrets.h, set WIFI_SSID and WIFI_PASSWORD
```

`secrets.h` is gitignored — your password never enters git history.

> **ESP32 only supports 2.4 GHz WiFi.** Connect to your router's 2.4 GHz SSID,
> not the 5 GHz one.

### 4. Upload

1. Open `motor_controller.ino` in Arduino IDE
2. You should see two tabs at the top: `motor_controller.ino` and `secrets.h`
3. Click **Upload** (→ arrow icon)
4. After flash completes, open **Tools → Serial Monitor** at **115200 baud**
5. Press the **EN / RST** button on the ESP32

Expected boot output:
```
Booting (single-mode firmware)...
Motor driver initialized (JGB37, encoder polarity 0)
Battery: 7800 mV
Connecting to your-ssid
......
Connected! IP: 192.168.1.189
HTTP server ready (single-mode: /drive + /stop)
```

Note the IP address — that's the ESP32's address for all the HTTP commands below.

---

## HTTP API

All endpoints accept POST with JSON body. Replace `192.168.1.189` with your
ESP32's IP from the boot log.

### `GET /`

Returns status text including battery voltage and WiFi signal strength.

```powershell
Invoke-RestMethod -Uri "http://192.168.1.189/" -Method Get
```

### `POST /drive`

**The control interface.** Continuous wheel velocities, held until replaced.
The firmware's 300 ms watchdog stops the motors if no fresh `/drive` arrives —
so callers must stream (change-or-heartbeat < 0.3 s; see
`examples/rc_car_agent/esp32_bridge.py` for the proven client discipline).

```powershell
Invoke-WebRequest -Uri "http://192.168.1.189/drive" -Method Post `
  -Body '{"left": 0.5, "right": 0.5}' -ContentType "application/json" -DisableKeepAlive
```

| Field | Type | Range | Notes |
|---|---|---|---|
| `left`  | float | −1..1 | fraction of `TELEOP_MAX_SPEED`; + = forward |
| `right` | float | −1..1 | `left > right` turns right (CW); `right > left` turns left (CCW) |

A single command moves the car for at most ~0.3 s — that's the watchdog
working, not a bug. Keep-alive is not supported (Arduino WebServer);
send `Connection: close` / `-DisableKeepAlive` and stay at ≤ ~10 req/s.

### `POST /stop`

Stop all motors immediately.

```powershell
Invoke-RestMethod -Uri "http://192.168.1.189/stop" -Method Post
```

### `POST /test_m1` … `/test_m4`

Spin one motor for 1 second at gentle speed. Used during wheels-in-air
testing to confirm which physical wheel each M_n_ port drives.

```powershell
Invoke-RestMethod -Uri "http://192.168.1.189/test_m1" -Method Post
```

### `GET /battery`

Returns motor battery voltage in millivolts as JSON.

---

## Bring-up & calibration

The firmware itself has only two tunables — `TELEOP_MAX_SPEED` (±50 max) and
`TELEOP_WATCHDOG_MS` (keep 300). **There is no timing calibration anymore**:
the old `MS_PER_*` open-loop procedure is gone because the desktop agent
closes the loop against live pose (`examples/rc_car_agent/calibrate.py`
derives the camera→car constants agent-side).

### Bring-up procedure (wheels in air)

1. **Identify each motor**: run `/test_m1` through `/test_m4`. Note which
   physical wheel spins for each port; if the mapping differs from yours,
   fix the `M1_REVERSED..M4_REVERSED` constants.
2. **Direction check**: stream `/drive {"left": 0.3, "right": 0.3}` (use
   `spin_test.py` in `examples/rc_car_agent/`) — all four wheels must roll
   the car forward.
3. **Watchdog check (safety gate)**: stream `/drive`, then stop sending —
   wheels must halt on their own within ~0.3 s.

### Common issues

- **Car drifts left or right** when both sides get equal speed: motors are
  unbalanced or a wheel is misaligned — the agent's closed loop corrects
  gentle drift automatically, but check the chassis if it's severe.
- **A single `/drive` only moves the car briefly**: that's the watchdog —
  clients must stream (heartbeat < 0.3 s), see `esp32_bridge.py`.

---

## Troubleshooting

### Upload fails: "Could not open COM port — Access denied"

Close Serial Monitor before uploading. Arduino IDE 2.x usually auto-closes
it, but occasionally fails. If still failing:

```powershell
Get-Process esptool, python, arduino* -ErrorAction SilentlyContinue | Stop-Process -Force
```

Unplug → wait 10 s → replug ESP32.

### Boot output shows `Battery: 0 mV`

Motor power battery is not connected to the Hiwonder controller. The ESP32
talks to the controller fine (logic power from ESP32's 5V), but motors won't
spin without the separate 6–8 V battery.

### Motors don't spin, or spin at full speed and won't stop

Encoder polarity is wrong. In `setup()`:

```cpp
writeMotorReg(MOTOR_ENCODER_POL_ADDR, 0);  // change 0 → 1
```

Re-flash. This is documented in Hiwonder's reference Python as the #1 quirk.

### I²C scanner finds no device

- Check the 4 wires: SDA↔21, SCL↔22, 5V↔VIN, GND↔GND
- Make sure ESP32 is on USB power (Hiwonder controller doesn't power the ESP32)
- Confirm Molex 5264 connector seats fully — these latch with a click
- Logic level: ESP32 is 3.3 V, but Hiwonder's I²C pulls up to 3.3 V internally
  on this board — no level shifter needed

### `Connecting to ...` hangs with endless dots

- ESP32 only does 2.4 GHz WiFi — not 5 GHz
- Check `WIFI_SSID` and `WIFI_PASSWORD` in `secrets.h` exactly match your network
- Some routers have AP isolation enabled — disable, or pick a different network

---

## Reference

- **Hiwonder 4-ch motor driver protocol**: PDF "01 Raspberry Pi Development"
  (Hiwonder customer docs). Register map, motor types, polarity behavior
  are all derived from there.
- **Chassis product page**: Hiwonder Large Metal 4WD Vehicle Chassis with
  8 V Encoder Geared Motor
- **Motor model**: JGB37_520_12V_110RPM (44 pulses/rev × 90:1 reduction)

---

## File map

```
motor_controller/
├── motor_controller.ino     # main firmware
├── secrets.h.example        # template (committed)
├── secrets.h                # YOUR credentials (gitignored)
└── README.md                # this file
```
