def main():
    students = [
        {"name": "윤인성", "korean": 87, "math": 98, "english": 88, "science": 95},
        {"name": "연하진", "korean": 92, "math": 78, "english": 76, "science": 83},
        {"name": "구지연", "korean": 76, "math": 68, "english": 94, "science": 84},
        {"name": "나선주", "korean": 98, "math": 96, "english": 75, "science": 79},
        {"name": "윤아린", "korean": 67, "math": 83, "english": 99, "science": 91},
        {"name": "윤명원", "korean": 74, "math": 91, "english": 69, "science": 95},
    ]
    print("이름\t총점\t평균")
    for student in students:
        score_sum = (student["korean"]+student["math"]+student["english"]+student["science"])
        score_average = score_sum/ 4
        print(f"{student['name']}\t{score_sum}\t{score_average:.2f}")

if __name__ == "__main__":
    main()
