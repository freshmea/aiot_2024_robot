# 사용자가 여러 숫자를 쉽포로 구분하여 입력을 받습니다.input
# ex) 10, 20, 30, abc
# 합계 계산, 평균 계산, 최대값 계산, 최소값 계산
# 만약에 문자가 있으면 무시하고 계산을 진행하고 어떤 값이 무시되었는지 출력!
# 유효한 숫자가 없습니다. ->
# format 을 써서 자리수를 유효숫자 소수점 3번째 자리까지 출력.

def main():
    user_input = input("숫자를 쉼표로 구분하여 입력하세요.")

    split_values = user_input.split(',')

    # 초기 변수
    index = 0
    total = 0
    count = 0
    maximum = None
    minimum = None

    while index < len(split_values):
        current_number = split_values[index].strip()

        try:
            number = float(current_number)
            total += number
            count += 1
            if maximum is None or number > maximum:
                maximum = number
            if minimum is None or number < minimum:
                minimum = number
        except ValueError:
            print(f"무시된 값: {current_number}")

        index +=1

    if count > 0:
        average = total /count
        print(f"합계 : {total:.3f}")
        print(f"평균 : {average:.3f}")
        print(f"최대값 : {maximum}")
        print(f"최소값 : {minimum}")
    else:
        print("유효한 숫자가 없습니다.")

if __name__ == "__main__":
    main()
