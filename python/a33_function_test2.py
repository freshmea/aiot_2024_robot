
def print_n_times(*value, n = 2, **kwargs):
    print(type(value))
    print(type(kwargs))
    # a,b,c = value
    for _ in range(n):
        for ele in value:
            print(ele , end=" ")
        print()
    # print("함수 작동 중")
    for key, value in kwargs.items():
        print(f"{key} : {value}")

def abc_print(a, b = "요", c = ".", *d):
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")

def main():
    print_n_times("안녕하세요.", "파이썬", "방갑습니다.", n=3)
    # abc_print( "안녕", c ="하세")
    abc_print( "안녕", "하세", "입니다.", "a", "b")
    
    # 가변 키워드 매개변수
    print_n_times("안녕하세요.", "파이썬", "방갑습니다.", n=3, m = 4, l = 2)
    

if __name__ == "__main__":
    main()
