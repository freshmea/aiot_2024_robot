def main():
    number = [103, 52, 273, 32, 77]
    print(min(number))
    print(max(number))
    print(sum(number))
    reversed_list = reversed(number)
    li1 = [ele for ele in reversed_list]
    # li1 = []
    # for ele in reversed_list:
    #     print(ele, end=" ")
    #     li1.append(ele)
    # print()
    print(li1)
    number.reverse()
    print(number)

if __name__ == "__main__":
    main()
