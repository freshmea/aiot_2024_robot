def test():
    print("함수가 호출되었습니다.")
    print("C 지점")
    yield "first"
    print("함수가 호출되었습니다.")
    print("D 지점")
    yield "second"

def main():
    # print("A 지점")
    # generator = test()
    # text = next(generator)
    # print(text)
    # print("B 지점")
    # text = next(generator)
    # print(text)
    for text in test():
        print(text)

if __name__ == "__main__":
    main()
