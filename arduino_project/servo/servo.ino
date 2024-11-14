#include <Servo.h>

Servo myServo;
int angle = 0;
int SERVO_PIN = 6;

void setup()
{
    myServo.attach(SERVO_PIN);
}

void loop()
{
    for (angle = 0; angle < 180; angle += 1)
    {
        myServo.write(angle);
        delay(15);
    }
    for (angle = 180; angle >= 0; angle -= 1)
    {
        myServo.write(angle);
        delay(15);
    }
}