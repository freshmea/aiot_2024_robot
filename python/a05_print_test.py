def main():
    print("# 하나만 출력합니다.")
    print()
    print("# 여러 개를 출력합니다.", end="")
    print(10, 20, '삼십', 40.0, sep='\t')
    print()
    print("""
첫 번째 줄
두 번째 줄
세 번째 줄""")

if __name__ == "__main__":
    main()
