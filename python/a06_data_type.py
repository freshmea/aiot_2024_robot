
class Abc:
    pass

def main():
    print(type(10))
    print(type(object()))
    print(issubclass(str, int))
    print(issubclass(int, object))
    print(issubclass(Abc, int))
    print(issubclass(Abc, object))

if __name__ == "__main__":
    main()
