# datetime을 활용해서 month 데이터를 얻어와서 
# 조건문으로 3 ~ 5 포함 봄
# 6 ~ 8 포함 까지 여름
# 9 ~ 11 포함 까지 가을
# 나머지 겨울 을 출력하는 프로그램 작성.
import datetime


def main():
    month = datetime.datetime.now().month
    month = 4
    if month < 3:
        print(f"이번달은 {month}달로 겨울입니다.")
    elif month < 6:
        print(f"이번달은 {month}달로 봄입니다.")
    elif month < 9:
        print(f"이번달은 {month}달로 여름입니다.")
    elif month < 12:
        print(f"이번달은 {month}달로 가을입니다.")
    else:
        print(f"이번달은 {month}달로 겨울입니다.")
    # if 3 <= month  <= 5:
    #     print(f"이번달은 {month}달로 봄입니다.")

if __name__ == "__main__":
    main()
