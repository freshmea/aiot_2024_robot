def main():
    output_a = "안녕안녕하세안녕요11111안녕 111 안녕.".find("안녕")
    print(output_a)
    while output_a != -1:
        output_a = "안녕안녕하세안녕요11111안녕 111 안녕.".find("안녕", output_a+1)
        print(output_a)

    print("안녕11" in "안녕하세요.")
    
    a = "10 20 30 40 50".split()
    print(a)
    str1 = " ".join(a)
    print(str1)

if __name__ == "__main__":
    main()
