def main():
    for i in range(10):
        print(f"{i}번째 반복 출력입니다.")
    li1 = [0,1,2,3,4,5,6,7,8,9]

    for i in li1:
        print(f"{i}번째 반복 출력입니다.")

    for i in range(10,1,-1):
        print(f"{i}번째 반복 출력입니다.")

    li2 = ['banana', 'apple', 'grape', 'mango', 'dragonfruit']
    for ele in li2:
        print(ele)

if __name__ == "__main__":
    main()
