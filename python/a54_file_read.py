import os


def main():
    print(os.getcwd())
    os.chdir("/home/aa/aiot_2024_robot/python/data")
    if os.path.exists("basic.txt"):
        with open('basic.txt', 'r', encoding='utf-8') as file:
            # data = file.read()
            # print(data)
            while(data := file.readline()):
                print(data, end="")
    else:
        print("파일이 없습니다.")

if __name__ == "__main__":
    main()
