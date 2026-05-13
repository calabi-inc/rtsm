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
const unsigned long MS_PER_METER  = 3000; // ms to drive 1 m at DRIVE_SPEED — CALIBRATE
const unsigned long MS_PER_RADIAN = 800;  // ms to rotate 1 rad at TURN_SPEED — CALIBRATE

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

// Drive at given speeds for duration_ms, then stop. No resending needed in
// fixed-speed (closed-loop) mode — STM32 maintains the speed automatically.
void driveTimed(int8_t m1, int8_t m2, int8_t m3, int8_t m4, unsigned long duration_ms) {
    setMotorSpeed(m1, m2, m3, m4);
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
    driveTimed(sp, sp, sp, sp, ms);

    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"forward\"}");
}

void handleTurn() {
    float angle = 0.0;
    if (server.hasArg("plain")) angle = parseFloatArg(server.arg("plain"), "angle", 0.0);

    Serial.printf("Turn: %.3f rad\n", angle);
    unsigned long ms = (unsigned long)(fabs(angle) * MS_PER_RADIAN);

    // In-place rotation: left motors and right motors opposite signs.
    // Assumption: M1/M2 = left side, M3/M4 = right side.
    // POSITIVE angle = CCW (left turn) → left wheels reverse, right wheels forward.
    // If car turns the wrong way during test, swap the signs below.
    int8_t left  = (angle >= 0) ? -TURN_SPEED : TURN_SPEED;
    int8_t right = (angle >= 0) ?  TURN_SPEED : -TURN_SPEED;
    driveTimed(left, left, right, right, ms);

    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"turn\"}");
}

void handleStop() {
    stopMotors();
    Serial.println("Stop");
    server.send(200, "application/json", "{\"ok\":true,\"cmd\":\"stop\"}");
}

void handleTestMotor(int motorNum) {
    int8_t s[4] = {0, 0, 0, 0};
    s[motorNum - 1] = 20;  // gentle test speed (closed-loop units, ±50 max)
    Serial.printf("Test M%d for 1 sec\n", motorNum);
    driveTimed(s[0], s[1], s[2], s[3], 1000);
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
