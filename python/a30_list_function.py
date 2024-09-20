def main():
    number = [103, 52, 273, 32, 77]
    
    # min, max, sum
    print(min(number))
    print(max(number))
    print(sum(number))
    
    # reversed
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
    
    # list_comprehension
    array = ["사과", "자두", "초콜릿", "바나나", "체리"]
    output = [fruit for fruit in array if fruit != "초콜릿"]
    print(output)
    
    # 삼항 연산자 C condition ? true : false
    print( "True" if False else "False")

if __name__ == "__main__":
    main()
