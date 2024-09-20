class Myerror(Exception):
    def __init__(self, *args):
        super().__init__(*args)
        self.args = ['이것은 내가 만든 에러 입니다']

def check_negative_number(number):
    if number < 0 :
        raise Myerror

def main():
    li1 = [1,2,3,4,5,6]
    try:
        number = int(input("정수 입력>"))
        check_negative_number(number)
        print(li1[number])
    except ValueError as e:
        print(type(e), e)
    except IndexError as e:
        print(type(e), e)
    except Myerror as e:
        print(type(e), e)
    finally:
        print("프로그램 종료")

if __name__ == "__main__":
    main()
