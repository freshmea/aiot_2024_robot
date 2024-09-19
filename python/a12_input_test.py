def main():
    str_input = input("숫자를 넣으시오 100을 더해서 출력합니다>\n")
    # print(str_input)
    # print(type(str_input))
    if str_input.isdigit():
        print(int(str_input) + 100)
    else:
        print("숫자가 아닙니다.")
if __name__ == "__main__":
    main()
