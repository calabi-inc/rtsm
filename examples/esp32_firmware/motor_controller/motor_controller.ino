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
// Determined empirically by running /test_m1 ... /test_m4 and observing:
//   M1 = front-right, spins BACKWARD with +speed (wires reversed)
//   M2 = rear-right,  spins forward with +speed
//   M3 = front-left,  spins forward with +speed
//   M4 = rear-left,   spins BACKWARD with +speed (wires reversed)
//
// To make a wheel physically roll forward, we invert the I2C speed value
// for the two reversed ports.
const bool M1_REVERSED = true;   // front-right
const bool M2_REVERSED = false;  // rear-right
const bool M3_REVERSED = false;  // front-left
const bool M4_REVERSED = true;   // rear-left

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
    // Left side: M3 (front-left), M4 (rear-left)
    int8_t m3 = M3_REVERSED ? -leftPhysical  : leftPhysical;
    int8_t m4 = M4_REVERSED ? -leftPhysical  : leftPhysical;
    // Right side: M1 (front-right), M2 (rear-right)
    int8_t m1 = M1_REVERSED ? -rightPhysical : rightPhysical;
    int8_t m2 = M2_REVERSED ? -rightPhysical : rightPhysical;
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

    // In-place rotation. Sign convention determined empirically:
    //   First test: positive angle made the car rotate CW (right), not CCW.
    //   That means our M3/M4 == "left side" assumption was inverted relative
    //   to the car's physical forward direction. Flipping signs here makes
    //   POSITIVE angle = CCW (left turn) as desired.
    int8_t left  = (angle >= 0) ?  TURN_SPEED : -TURN_SPEED;
    int8_t right = (angle >= 0) ? -TURN_SPEED :  TURN_SPEED;
    driveTimed(left, right, ms);

    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"turn\"}");
}

void handleStop() {
    stopMotors();
    Serial.println("Stop");
    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"stop\"}");
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
}
