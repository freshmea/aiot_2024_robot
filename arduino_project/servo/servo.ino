#include <Arduino.h>
#include <Servo.h>
#include <string.h>

Servo myServo;
int angle = 0;
int SERVO_PIN = 6;

void setup()
{
    myServo.attach(SERVO_PIN);
    Serial.begin(115200);
}

void loop()
{
    static String buffer;
    if (Serial.available() > 0)
    {
        buffer = Serial.readStringUntil('\n');
        Serial.print("Echo : ");
        Serial.print(buffer.substring(0, 4));
        Serial.print(buffer.substring(4, 7));
        if (buffer.substring(0, 4) == "move")
        {
            Serial.print("move command received");
            int pos = buffer.substring(4, 7).toInt();
            myServo.write(pos);
        }
    }
}