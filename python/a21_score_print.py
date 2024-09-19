def main():
    score = input("학점 입력>")
    try:
        if not score.isdigit():
            raise
        score = float(score)
        if score > 4.5:
            raise
        if score < 0:
            raise
    except:
        exit()
    
    if score  ==  4.5:
        print(f"신")
    elif score  >=  4.2:
        print(f"교수님의 사랑")
    elif score  >=  3.5:
        print(f"현 체제의 수호자")
    elif score  >=  2.8:
        print(f"일반인")
    elif score  >=  2.3:
        print(f"일탈을 꿈꾸는 소시민")
    elif score  >= 1.75:
        print(f"오락문화의 선구자")
    elif score  >=  1.0:
        print(f"불가촉천민")
    elif score  >=  0.5:
        print(f"자벌레")
    elif score  >  0:
        print(f"플랑크톤")
    else:
        print("시대를 앞서가는 혁명의 씨앗")
    
if __name__ == "__main__":
    main()
