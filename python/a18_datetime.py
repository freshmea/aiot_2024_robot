import datetime
import time


def main():
    now = datetime.datetime.now()
    print(f"""{now.year} 년
{now.month} 월
{now.day} 일
{now.hour} 시간
{now.minute} 분
{now.second} 초""")
    
    print(time.time())

if __name__ == "__main__":
    main()
