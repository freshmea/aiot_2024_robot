const int buttonPin = 7;
int buttonState = 0;
bool flag = false;

void setup()
{
    Serial.begin(115200);
    pinMode(buttonPin, INPUT);
}

void loop()
{
    static unsigned long prev_time = millis();
    buttonState = digitalRead(buttonPin);
    if (buttonState == LOW)
    {
        if (!flag && (millis() - prev_time) > 100)
        {
            prev_time = millis();
            Serial.println("Button Pressed!!");
            flag = true;
        }
    }
    else if (flag)
    {
        flag = false;
        Serial.println("Button Reales!!");
        prev_time = millis();
    }
}