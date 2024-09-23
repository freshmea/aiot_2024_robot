class Student:
    def __init__(self, name, korean, math, english, science):
        self.name =  name # a_studnet.name = name ("A"),  b_studnet.name = name ("B")
        self.korean =  korean
        self.math =  math
        self.english = english
        self.science =  science
        # (*this) == self
        # C 에서 생량가능 this->_var --> _var


def main():
    students = [
        Student("윤인성", 87,98, 88, 95),
        Student("연하진", 92, 78, 76, 83),
        Student("구지연", 76,68, 94, 84),
        Student("나선주", 98, 96, 75, 79),
        Student("윤아린", 67, 83, 99, 91),
        Student("윤명원", 74, 91, 69, 95),
    ]
    print("이름\t총점\t평균")
    for student in students:
        score_sum = (student.korean + student.math + student.english + student.science)
        score_average = score_sum/ 4
        print(f"{student.name}\t{score_sum}\t{score_average:.2f}")
    # a_student = Student("A", 11, 22, 33, 44)
    # b_student = Student("B", 11, 22, 33, 44)
    # print(a_student.name)

if __name__ == "__main__":
    main()
