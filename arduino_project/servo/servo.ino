#include <Arduino.h>
#include <Servo.h>
#include <string.h>

Servo myServo;
int angle = 0;
int SERVO_PIN = 9;

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
        Serial.print(buffer.substring(0, 4));
        if (buffer.substring(0, 4) == "move")
        {
            int pos = buffer.substring(4, 7).toInt();
            myServo.write(pos);
        }
        Serial.flush();
    }
    // for (angle = 0; angle < 180; angle++)
    // {
    //     myServo.write(angle);
    //     delay(15);
    // }
    // for (angle = 180; angle > 0; angle--)
    // {
    //     myServo.write(angle);
    //     delay(15);
    // }
}