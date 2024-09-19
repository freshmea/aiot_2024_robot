def main():
    bo = bool()
    bo1 = True
    bo1 = [] # no element is False
    bo1 = 0.0000001
    bo1 = ""
    if bo1:
        print("bo1 is True")
    else:
        print("bo1 is False")
        
    # c (!true) -> false
    # python (not True) -> False
    # c ( ACondition && BCondition )
    # python ( ACondition and BCondition )
    # c ( ACondition || BCondition)
    # python (ACondition or BCondition)

if __name__ == "__main__":
    main()
