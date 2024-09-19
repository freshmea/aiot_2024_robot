class Abc:
    def __getitem__(self, idx):
        return idx
    

def main():
    print("문자 선택 연산자에 대해 알아볼까요?")
    print("안녕하세요"[0])
    print("안녕하세요"[1])
    print("안녕하세요"[2])
    print("안녕하세요"[3])
    print("안녕하세요"[4])
    # print("안녕하세요"[5]) out of range
    a = Abc()
    print(a[3:4])
    print("안녕하세요"[-1])
    print("안녕하세요"[-2])
    print("안녕하세요"[-3])
    print("안녕하세요"[-4])
    print("안녕하세요"[-5])
    # print("안녕하세요"[-6]) out of range
    print("안녕하세요"[1:4])
    print("안녕하세요"[1:4:1])
    print("안녕하세요"[1:4:2])
    print("안녕하세요"[::-1])
    

if __name__ == "__main__":
    main()
