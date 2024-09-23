def create_student(name, korean, math, english, science):
    return {
        "name": name,
        "korean": korean,
        "math": math,
        "english": english,
        "science": science}


def main():
    students = [
        create_student("윤인성", 87,98, 88, 95),
        create_student("연하진", 92, 78, 76, 83),
        create_student("구지연", 76,68, 94, 84),
        create_student("나선주", 98, 96, 75, 79),
        create_student("윤아린", 67, 83, 99, 91),
        create_student("윤명원", 74, 91, 69, 95),
    ]
    print("이름\t총점\t평균")
    for student in students:
        score_sum = (student["korean"]+student["math"]+student["english"]+student["science"])
        score_average = score_sum/ 4
        print(f"{student['name']}\t{score_sum}\t{score_average:.2f}")

if __name__ == "__main__":
    main()
