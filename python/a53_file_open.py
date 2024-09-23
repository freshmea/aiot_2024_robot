import os
import sys


def main():
    print(os.path)
    print(__file__)
    print(os.getcwd())
    os.chdir("/home/aa/aiot_2024_robot/python/data")
    if os.path.exists("basic.txt"):
        print("파일이 이미 있습니다.")
        with open("basic.txt", 'a', encoding='utf-8') as file:
            file.write("this is file save data! add\n")
    else:
        with open('basic.txt', 'w', encoding='utf-8') as file:
            file.write("this is file save data! 1\n")
            file.write("this is file save data! 2\n")

if __name__ == "__main__":
    main()
