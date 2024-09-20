def main():
    list_a = [1, 2, 3]
    print(list_a)

    # append
    list_a.append(4)
    list_a.append(5)
    print(list_a)

    # insert
    list_a.insert(0, 0)
    list_a.insert(3, 2.5)
    print(list_a)
    
    # del
    del list_a[0]
    print(list_a)
    a = "a variable"
    list_a.insert(0, a)
    print(list_a)
    del list_a[0]
    print(list_a)
    print(a)
    # del a
    # print(a)
    
    # pop
    list_a.append("last element")
    re = list_a.pop(2)
    print(list_a, re)
    
    #remove by value
    list_a.remove("last element")
    print(list_a)

if __name__ == "__main__":
    main()
