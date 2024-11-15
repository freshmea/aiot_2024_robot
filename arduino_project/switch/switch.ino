#include <string.h>
class Switch
{

public:
    Switch(int buttonPin, bool pullup = true)
        : _buttonPin(buttonPin), _prev_time(millis()), _buttonState(HIGH), _flag(false)
    {
        if (pullup)
        {
            pinMode(_buttonPin, INPUT_PULLUP);
        }
        else
        {
            pinMode(_buttonPin, INPUT);
        }
    }
    String read()
    {
        _buttonState = digitalRead(_buttonPin);
        if (_buttonState == LOW)
        {
            if (!_flag && (millis() - _prev_time) > 100)
            {
                _prev_time = millis();
                _flag = true;
                return "button_on" + String(_buttonPin);
            }
        }
        else if (_flag)
        {
            _flag = false;
            _prev_time = millis();
            return "button_off" + String(_buttonPin);
        }
    }

private:
    int _buttonPin;
    int _buttonState;
    bool _flag;
    unsigned long _prev_time;
};

Switch mySwitch1(7);
Switch mySwitch2(6);
Switch mySwitch3(5);
void setup()
{
    Serial.begin(115200);
}

void loop()
{
    String output1 = mySwitch1.read();
    String output2 = mySwitch2.read();
    String output3 = mySwitch3.read();
    if (output1 != "")
    {
        Serial.println(output1);
    }
    if (output2 != "")
    {
        Serial.println(output2);
    }
    if (output3 != "")
    {
        Serial.println(output3);
    }
}