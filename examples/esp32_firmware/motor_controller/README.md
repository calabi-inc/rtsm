# ESP32 Motor Controller — RTSM Demo 2 RC Car

ESP32 firmware that drives a **Hiwonder Large Metal 4WD chassis** over WiFi.
The PC sends HTTP commands (`/forward`, `/turn`, `/stop`); the ESP32 talks I²C
to Hiwonder's 4-channel encoder motor driver to spin the wheels.

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
Booting...
Motor driver initialized (JGB37, encoder polarity 0)
Battery: 7800 mV
Connecting to your-ssid
......
Connected! IP: 192.168.1.189
HTTP server ready
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

### `POST /forward`

Drive forward (or backward) a distance in meters.

```powershell
Invoke-WebRequest -Uri "http://192.168.1.189/forward" -Method Post `
  -Body '{"distance": 1.0}' -ContentType "application/json"
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `distance` | float | 0.5 | Negative = reverse |

### `POST /turn`

Rotate in place by angle in radians.

```powershell
Invoke-WebRequest -Uri "http://192.168.1.189/turn" -Method Post `
  -Body '{"angle": 1.5708}' -ContentType "application/json"
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `angle` | float | 0.0 | Positive = CCW (left). 1.5708 ≈ 90° |

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

## Calibration

After flashing, three constants in `motor_controller.ino` need tuning to your
specific car, battery, and floor surface:

```cpp
const int8_t DRIVE_SPEED      = 30;     // ±50 max — speed for forward/back
const int8_t TURN_SPEED       = 25;     // speed for in-place rotation
const unsigned long MS_PER_METER  = 3000;  // ms to travel 1 m
const unsigned long MS_PER_RADIAN = 800;   // ms to rotate 1 rad
```

### Procedure

1. **Wheels in air**: lift the chassis on books so wheels don't touch anything.
2. **Identify each motor**: run `/test_m1` through `/test_m4`. Note which
   physical wheel spins for each port. Label them with masking tape.
3. **Direction check**: run `/forward` with `distance: 0.5`. All four wheels
   should spin to drive the car forward. If any spin the wrong way, swap that
   motor's connector orientation or flip its sign in code.
4. **Floor test, 1 meter**: place the car on the floor, mark the starting
   position. Run `/forward` with `distance: 1.0`. Measure how far it actually
   travelled.
5. **Adjust `MS_PER_METER`** proportionally:
   ```
   new_MS_PER_METER = old_MS_PER_METER × (1.0 / measured_distance_m)
   ```
6. **Repeat for `MS_PER_RADIAN`** using `/turn` with `angle: 3.14159` (180°)
   and measuring the actual rotation.

### Common calibration issues

- **Car drifts left or right** during `/forward`: motors are unbalanced. Adjust
  individual motor speeds in `setMotorSpeed()`, or check wheel alignment.
- **Different distances on carpet vs hardwood**: friction matters. Calibrate
  on the surface used for the demo recording.
- **Battery drop changes distance**: switch from PWM (open-loop) to fixed
  speed (closed-loop) mode — already the default in this firmware.

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
