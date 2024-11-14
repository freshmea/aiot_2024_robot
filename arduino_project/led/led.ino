#include <Arduino.h>
#include <string.h>

void setup()
{
    pinMode(13, OUTPUT);
    Serial.begin(115200);
}

void loop()
{
    static String buffer;
    if (Serial.available() > 0)
    {
        buffer = Serial.readStringUntil('\n');
        if (buffer == "high")
            digitalWrite(13, HIGH);
        if (buffer == "low")
            digitalWrite(13, LOW);
        Serial.print("Echo : ");
        Serial.println(buffer);
    }
    delay(100);
    // delay(1000);
    // digitalWrite(13, LOW);
    // delay(1000);
}