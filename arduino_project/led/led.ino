#include <Arduino.h>
#include <string.h>

void setup()
{
    pinMode(13, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
    String buffer;
    if (Serial.available() > 0)
    {
        buffer = Serial.readStringUntil('\n');
        Serial.print("Echo : ");
        Serial.println(buffer);
    }
    delay(100);
    // digitalWrite(13, HIGH);
    // delay(1000);
    // digitalWrite(13, LOW);
    // delay(1000);
}