from functools import lru_cache


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# f(0) = 0, f(1) = 1, f(2) = 1, f(3) = 2, f(4) = 3 , ....
counter = 0

@lru_cache(maxsize=None)
def fibonacci(n):
    global counter
    counter += 1
    if n == 1:
        return 1
    if n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

dictionary = {
    1 : 1,
    2 : 1
}

def fibonacci_memo(n):
    global counter # , dictionary
    counter += 1
    if n in dictionary:
        return dictionary[n]
    else:
        output = fibonacci_memo(n-1) + fibonacci_memo(n-2)
        dictionary[n] = output
        return output

def main():
    global counter
    print(f"1!: {factorial(1)}")
    print(f"2!: {factorial(2)}")
    print(f"3!: {factorial(3)}")
    print(f"4!: {factorial(4)}")
    print(f"5!: {factorial(5)}")
    print(f"fibonacci(40) : {fibonacci(40)}")
    # print(f"fibonacci_memo(40) : {fibonacci_memo(40)}")
    print(f"counter : {counter}")

if __name__ == "__main__":
    main()
