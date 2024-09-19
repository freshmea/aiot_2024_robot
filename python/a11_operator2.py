import math


def main():
    pi = 3.141592
    print(pi)
    print(math.pi)
    pi = math.pi
    r = 10
    # print("원주율 =", pi)
    # print("반지름 =", r)
    # print("원의 둘레 =", 2*pi*r)
    # print("원의 넓이 =", pi*r**2)
    print(f"원주율 = {pi}\n반지름 = {r}")
    print(f"원의 둘레 = {2*pi*r}\n원의 넓이 = {pi*r**2}")
    
    # 복합 대입 연산자
    # i ++
    i = 0
    i += 1
    str1 = "이것은"
    str1 += " 파이썬입니다."
    print(str1)

if __name__ == "__main__":
    main()
