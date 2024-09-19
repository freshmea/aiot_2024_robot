def main():
    format_a = "{}만 원".format(500)
    format_b = "파이썬 열공해서 첫 연봉 {}만 원 만들기".format(5000)
    format_c = "{2:5.2f}\n{1:10.2f}\n{0:015d}".format(3000, 4000, 5000)
    format_d = "{} {} {}".format(1, "문자열", True)
    format_e = "{:05d}".format(-52)

    print(format_a)
    print(format_b)
    print(format_c)
    print(format_d)
    print(format_e)


if __name__ == "__main__":
    main()
