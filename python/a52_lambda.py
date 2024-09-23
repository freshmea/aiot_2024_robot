def call_10_times(func):
    for i in range(10):
        func()

def print_hello():
    print("안녕하세요")

def power(item):
    return item * item

def under_3(item):
    return item < 3

def main():
    print(print_hello)
    # ph = print_hello
    ph = lambda : print("안녕하세요")
    print(ph)
    call_10_times(ph)
    li1 = [1, 2, 3, 4, 5]
    print(list(map(power, li1)))
    print(list(filter(under_3, li1)))

    print(list(map(lambda x : x * x, li1)))
    print(list(filter(lambda x : x < 3, li1)))

if __name__ == "__main__":
    main()
