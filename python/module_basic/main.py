import test_module as test


def main():
    radius = test.number_input()
    print(test.get_circumference(radius))
    print(test.get_circle_area(radius))

    print("main.py 의 __name__")
    print(__name__)

if __name__ == "__main__":
    main()
