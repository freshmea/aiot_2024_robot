const int buttonPin = 2;
int buttonState = 0;

void setup()
{
    Serial.begin(115200);
    pinMode(buttonPin, INPUT);
}

void loop()
{
    buttonState = digitalRead(buttonPin);
    if (buttonState == LOW)
    {
        Serial.println("Button Pressed!!");
    }
    else
    {
        Serial.println("Button Released!!");
    }
}