""" this module is about circle, rectangle"""
PI = 3.1415926535

def number_input():
    """ input number function"""
    output = input("숫자 입력> ")
    return float(output)

def get_circumference(radius):
    """ this is calculate circle ..."""
    return 2 * PI * radius

def get_circle_area(radius):
    """ this is calculate circle area function"""
    return PI * radius ** 2

class Rectangle:
    """ this is Rectangle class, initial args is two"""
    def __init__(self, w, h):
        self.width = w
        self.height = h
        print()
    def get_area(self):
        """this method is return area value"""
        return self.width * self.height
    
def main():
    print("test_module.py 의 __name__")
    print(__name__)

if __name__ == "__main__":
    main()