// YWROBOT
// Compatible with the Arduino IDE 1.0
// Library version:1.1
#include <LiquidCrystal_I2C.h>
#include <Wire.h>

LiquidCrystal_I2C lcd(0x27, 16, 2); // set the LCD address to 0x27 for a 16 chars and 2 line display

void setup()
{
    lcd.init(); // initialize the lcd
    lcd.backlight();
    lcd.setCursor(3, 0);
    lcd.print("Hello, world!");
    lcd.setCursor(2, 1);
    lcd.print("Ywrobot Arduino!");
    lcd.setCursor(0, 2);
    lcd.print("Arduino LCM IIC 2004");
    lcd.setCursor(2, 3);
    lcd.print("Power By Ec-yuan!");
    Serial.begin(119200);
}

void loop()
{
    static String buffer;
    if (Serial.available() > 0)
    {
        buffer = Serial.readStringUntil('\n');
        Serial.print(buffer.substring(0, 4));
        if (buffer.substring(0, 4) == "Alcd")
        {
            lcd.setCursor(0, 0);
            lcd.print(buffer.substring(4, 17));
        }
        Serial.flush();
    }
}
