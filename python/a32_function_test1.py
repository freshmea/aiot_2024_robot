
def print_n_times(n : int, *value):
    print(type(value))
    # a,b,c = value
    for _ in range(n):
        for ele in value:
            print(ele , end=" ")
        print()
    # print("함수 작동 중")

def main():
    print_n_times(3, "안녕하세요.", "파이썬", "방갑습니다.")

if __name__ == "__main__":
    main()
