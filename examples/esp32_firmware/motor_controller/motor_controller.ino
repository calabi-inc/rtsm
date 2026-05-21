// motor_controller.ino
//
// RTSM Demo 2 — RC car ESP32 firmware
// Target: Hiwonder Large Metal 4WD chassis with 4-ch encoder motor driver
// Driver I2C address: 0x34 (Hiwonder STM32 motor controller)
// Motors: JGB37_520_12V_110RPM (4× mecanum)
//
// HTTP endpoints (POST):
//   /forward  {"distance": 1.5}  — drive distance in meters
//   /turn     {"angle": 1.57}    — turn angle in radians (positive = CCW/left)
//   /stop                        — emergency stop
//   /test_m1, /test_m2, /test_m3, /test_m4  — spin one motor 1 sec
//   /battery                     — read battery voltage
//
// Wiring (ESP32 → Hiwonder driver):
//   GPIO 21 → SDA
//   GPIO 22 → SCL
//   VIN/5V  → 5V
//   GND     → GND
//
// Protocol reference: Hiwonder "01 Raspberry Pi Development" PDF

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include "secrets.h"   // local file, gitignored — defines WIFI_SSID and WIFI_PASSWORD

const char* ssid     = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// I2C config
#define MOTOR_ADDR                 0x34
#define ADC_BAT_ADDR               0x00
#define MOTOR_TYPE_ADDR            0x14
#define MOTOR_ENCODER_POL_ADDR     0x15
#define MOTOR_FIXED_PWM_ADDR       0x1F
#define MOTOR_FIXED_SPEED_ADDR     0x33
#define MOTOR_ENCODER_TOTAL_ADDR   0x3C

// Motor types
#define MOTOR_TYPE_WITHOUT_ENCODER 0
#define MOTOR_TYPE_TT              1
#define MOTOR_TYPE_N20             2
#define MOTOR_TYPE_JGB37_520       3   // matches the 110RPM 8V geared motor

// Configuration — adjust during calibration.
// Using FIXED_SPEED mode (closed-loop with encoders), so speed is pulses per 10ms,
// range roughly ±50. Per Hiwonder reference Python: speed 50 = forward, -50 = reverse.
const int8_t DRIVE_SPEED      = 30;      // ±50 max — start gentle, calibrate up
const int8_t TURN_SPEED       = 25;
const unsigned long MS_PER_METER  = 6200; // ms to drive 1 m at DRIVE_SPEED — CALIBRATE
                                          // Calibration log (carpet, DRIVE_SPEED=30):
                                          //   3000 → 0.42 m (huge undershoot, startup lag)
                                          //   7150 → 1.15 m (15% overshoot)
                                          //   6200 → expected ~1.0 m (target)
const unsigned long MS_PER_RADIAN = 1600; // ms to rotate 1 rad at TURN_SPEED — CALIBRATE
                                          // Bumped 800→1600 after first test: 1.57 rad
                                          // commanded produced ~0.78 rad observed.
                                          // Last test: ~83° vs 90° commanded (8% short,
                                          // acceptable for demo).

// === Motor port → physical wheel mapping ===
// Determined empirically by running /test_m1 ... /test_m4, then verified
// against /drive joystick behavior (stick right -> physically turn right).
//
//   M1 = front-LEFT,  spins BACKWARD with +speed (wires reversed)
//   M2 = rear-LEFT,   spins forward with +speed
//   M3 = front-RIGHT, spins forward with +speed
//   M4 = rear-RIGHT,  spins BACKWARD with +speed (wires reversed)
//
// (The first bring-up mapping used "top/bottom" from a chassis-flipped
// viewpoint which had left/right inverted — the joystick test exposed it.)
//
// To make a wheel physically roll forward, we invert the I2C speed value
// for the two reversed ports.
const bool M1_REVERSED = true;   // front-left
const bool M2_REVERSED = false;  // rear-left
const bool M3_REVERSED = false;  // front-right
const bool M4_REVERSED = true;   // rear-right

// === Teleop state (used by /drive endpoint for joystick control) ===
// Latest commanded speeds. teleopTick() applies them each loop iteration and
// stops the motors if no fresh /drive command arrives within the watchdog
// window — prevents runaway if WiFi or the controlling PC drops.
const int8_t  TELEOP_MAX_SPEED     = 40;
const unsigned long TELEOP_WATCHDOG_MS = 300;
volatile int8_t teleopLeft  = 0;
volatile int8_t teleopRight = 0;
volatile unsigned long teleopLastCmdMs = 0;
int8_t teleopAppliedLeft  = 0;
int8_t teleopAppliedRight = 0;

WebServer server(80);

// === Low-level I2C helpers ===

void writeMotorReg(byte reg, byte value) {
    Wire.beginTransmission(MOTOR_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

// FIXED_SPEED mode (register 0x33) — closed-loop. Set once, controller maintains.
// This matches the Hiwonder reference Python's approach.
void setMotorSpeed(int8_t m1, int8_t m2, int8_t m3, int8_t m4) {
    Wire.beginTransmission(MOTOR_ADDR);
    Wire.write(MOTOR_FIXED_SPEED_ADDR);
    Wire.write((byte)m1);
    Wire.write((byte)m2);
    Wire.write((byte)m3);
    Wire.write((byte)m4);
    Wire.endTransmission();
}

void stopMotors() {
    setMotorSpeed(0, 0, 0, 0);
}

// Higher-level: drive left-side wheels and right-side wheels at signed physical speeds.
// Positive = wheel rolls forward, negative = rolls backward.
// Handles per-motor polarity reversal so the caller can think in physical terms.
void driveLeftRight(int8_t leftPhysical, int8_t rightPhysical) {
    // Left side: M1 (front-left), M2 (rear-left)
    int8_t m1 = M1_REVERSED ? -leftPhysical  : leftPhysical;
    int8_t m2 = M2_REVERSED ? -leftPhysical  : leftPhysical;
    // Right side: M3 (front-right), M4 (rear-right)
    int8_t m3 = M3_REVERSED ? -rightPhysical : rightPhysical;
    int8_t m4 = M4_REVERSED ? -rightPhysical : rightPhysical;
    setMotorSpeed(m1, m2, m3, m4);
}

// Drive at given LEFT/RIGHT physical speeds for duration_ms, then stop.
// No resending needed in fixed-speed (closed-loop) mode — STM32 maintains
// the speed automatically.
void driveTimed(int8_t leftPhysical, int8_t rightPhysical, unsigned long duration_ms) {
    driveLeftRight(leftPhysical, rightPhysical);
    // Yield to HTTP server during the move so we still respond to /stop
    unsigned long start = millis();
    while (millis() - start < duration_ms) {
        server.handleClient();
        delay(10);
    }
    stopMotors();
}

// Read battery voltage (returns mV)
uint16_t readBatteryMV() {
    Wire.beginTransmission(MOTOR_ADDR);
    Wire.write(ADC_BAT_ADDR);
    Wire.endTransmission(false);
    Wire.requestFrom(MOTOR_ADDR, (uint8_t)2);
    if (Wire.available() < 2) return 0;
    uint8_t lo = Wire.read();
    uint8_t hi = Wire.read();
    return (hi << 8) | lo;
}

// Parse "key": number from a JSON-ish body. Returns defaultVal if not found.
float parseFloatArg(const String& body, const char* key, float defaultVal) {
    int keyIdx = body.indexOf(key);
    if (keyIdx < 0) return defaultVal;
    int colonIdx = body.indexOf(':', keyIdx);
    if (colonIdx < 0) return defaultVal;
    return body.substring(colonIdx + 1).toFloat();
}

// === HTTP handlers ===

void handleRoot() {
    String s = "ESP32 RC car controller online\n";
    s += "Battery: "; s += readBatteryMV(); s += " mV\n";
    s += "WiFi RSSI: "; s += WiFi.RSSI(); s += " dBm\n";
    server.send(200, "text/plain", s);
}

void handleForward() {
    float distance = 0.5;
    if (server.hasArg("plain")) distance = parseFloatArg(server.arg("plain"), "distance", 0.5);

    Serial.printf("Forward: %.3f m\n", distance);
    unsigned long ms = (unsigned long)(fabs(distance) * MS_PER_METER);
    int8_t sp = (distance >= 0) ? DRIVE_SPEED : -DRIVE_SPEED;
    // Forward drive: both sides physically forward at the same speed.
    driveTimed(sp, sp, ms);

    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"forward\"}");
}

void handleTurn() {
    float angle = 0.0;
    if (server.hasArg("plain")) angle = parseFloatArg(server.arg("plain"), "angle", 0.0);

    Serial.printf("Turn: %.3f rad\n", angle);
    unsigned long ms = (unsigned long)(fabs(angle) * MS_PER_RADIAN);

    // In-place rotation. POSITIVE angle = CCW (left turn): left wheels physically
    // backward, right wheels physically forward.
    // (The earlier sign-flip workaround was removed once driveLeftRight got
    // the M1/M2 vs M3/M4 labels in the right places — see the mapping
    // section above.)
    int8_t left  = (angle >= 0) ? -TURN_SPEED :  TURN_SPEED;
    int8_t right = (angle >= 0) ?  TURN_SPEED : -TURN_SPEED;
    driveTimed(left, right, ms);

    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"turn\"}");
}

void handleStop() {
    // Cancel any active teleop drive AND stop the motors.
    teleopLeft = 0;
    teleopRight = 0;
    teleopLastCmdMs = millis();
    stopMotors();
    Serial.println("Stop");
    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"stop\"}");
}

// /drive — non-blocking velocity command for joystick / PS4 teleop.
// Body: {"left": <-1.0..1.0>, "right": <-1.0..1.0>}
// Values are fractions of TELEOP_MAX_SPEED. Motors keep running at the
// commanded speeds until a new /drive (or /stop) arrives, or until the
// watchdog times out after TELEOP_WATCHDOG_MS — whichever comes first.
void handleDrive() {
    float leftPct = 0.0;
    float rightPct = 0.0;
    if (server.hasArg("plain")) {
        String body = server.arg("plain");
        leftPct  = parseFloatArg(body, "left",  0.0);
        rightPct = parseFloatArg(body, "right", 0.0);
    }
    if (leftPct  >  1.0) leftPct  =  1.0;
    if (leftPct  < -1.0) leftPct  = -1.0;
    if (rightPct >  1.0) rightPct =  1.0;
    if (rightPct < -1.0) rightPct = -1.0;

    teleopLeft  = (int8_t)(leftPct  * TELEOP_MAX_SPEED);
    teleopRight = (int8_t)(rightPct * TELEOP_MAX_SPEED);
    teleopLastCmdMs = millis();

    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"drive\"}");
}

// Called every loop() iteration. Applies the latest teleop speeds, or
// stops the motors if no fresh /drive command has arrived recently.
void teleopTick() {
    int8_t targetLeft, targetRight;
    if (millis() - teleopLastCmdMs > TELEOP_WATCHDOG_MS) {
        targetLeft = 0;
        targetRight = 0;
    } else {
        targetLeft = teleopLeft;
        targetRight = teleopRight;
    }
    // Only push to I2C when the speeds change — avoids spamming the bus.
    if (targetLeft != teleopAppliedLeft || targetRight != teleopAppliedRight) {
        driveLeftRight(targetLeft, targetRight);
        teleopAppliedLeft  = targetLeft;
        teleopAppliedRight = targetRight;
    }
}

void handleTestMotor(int motorNum) {
    // Spin one motor at raw +20 for 1 sec — does NOT apply polarity correction.
    // Used purely for the bring-up mapping step (figure out which port = which wheel).
    int8_t s[4] = {0, 0, 0, 0};
    s[motorNum - 1] = 20;
    Serial.printf("Test M%d for 1 sec (raw, no polarity correction)\n", motorNum);
    setMotorSpeed(s[0], s[1], s[2], s[3]);
    unsigned long start = millis();
    while (millis() - start < 1000) {
        server.handleClient();
        delay(10);
    }
    stopMotors();
    String resp = "{\"ok\":true,\"motor\":" + String(motorNum) + "}";
    server.send(200, "application/json", resp);
}

void handleBattery() {
    uint16_t mv = readBatteryMV();
    String resp = "{\"mv\":" + String(mv) + "}";
    server.send(200, "application/json", resp);
}

// === Setup ===

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println();
    Serial.println("Booting...");

    // I2C init
    Wire.begin();   // default SDA=21, SCL=22
    delay(50);

    // Initialize motor driver — delays match Hiwonder reference Python (0.5 sec).
    writeMotorReg(MOTOR_TYPE_ADDR, MOTOR_TYPE_JGB37_520);
    delay(500);
    writeMotorReg(MOTOR_ENCODER_POL_ADDR, 0);
    delay(50);
    stopMotors();
    Serial.println("Motor driver initialized (JGB37, encoder polarity 0)");

    uint16_t batt = readBatteryMV();
    Serial.printf("Battery: %u mV\n", batt);

    // WiFi
    Serial.printf("Connecting to %s\n", ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());

    // HTTP routes
    server.on("/",         HTTP_GET,  handleRoot);
    server.on("/forward",  HTTP_POST, handleForward);
    server.on("/turn",     HTTP_POST, handleTurn);
    server.on("/stop",     HTTP_POST, handleStop);
    server.on("/drive",    HTTP_POST, handleDrive);    // joystick teleop
    server.on("/battery",  HTTP_GET,  handleBattery);
    server.on("/test_m1",  HTTP_POST, []() { handleTestMotor(1); });
    server.on("/test_m2",  HTTP_POST, []() { handleTestMotor(2); });
    server.on("/test_m3",  HTTP_POST, []() { handleTestMotor(3); });
    server.on("/test_m4",  HTTP_POST, []() { handleTestMotor(4); });

    server.begin();
    Serial.println("HTTP server ready");
}

void loop() {
    server.handleClient();
    teleopTick();
}
