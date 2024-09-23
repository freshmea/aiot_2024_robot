class Student:
    count = 0
    students = []
    def __init__(self, name, korean, math, english, science):
        self.name =  name
        self.korean =  korean
        self.math =  math
        self.english = english
        self.science =  science
        self.sum = self.get_sum()
        self.average = self.get_average()
        Student.count += 1
        Student.students.append(self)

    def get_sum(self):
        return self.korean + self.math + self.english + self.science

    def get_average(self):
        return self.sum / 4

    def to_string(self):
        return f"{self.name}\t{self.get_sum()}\t{self.get_average():.2f}"

    def __str__(self):
        return f"{self.name}\t{self.get_sum()}\t{self.get_average():.2f}"

    def __eq__(self, value):
        return self.sum  == value.sum

    def __ne__(self, value):
        return self.sum  != value.sum

    def __gt__(self, value):
        return self.sum  > value.sum

    def __ge__(self, value):
        return self.sum  >= value.sum

    def __lt__(self, value):
        if isinstance(value, Student):
            return self.sum  < value.sum
        elif isinstance(value, int):
            return self.sum < value

    def __le__(self, value):
        return self.sum  <= value.sum

    @classmethod
    def print_students(cls):
        print("------- 학생목록 ------")
        print(f"총 학생수: {cls.count}")
        print("이름\t총점\t평균")
        for student in cls.students:
            print(student)
        print("------- ------ ------")


def main():
    Student("윤인성", 87, 98, 88, 95)
    Student("연하진", 92, 78, 76, 83)
    Student("구지연", 76, 68, 94, 84)
    Student("나선주", 98, 96, 75, 79)
    Student("윤아린", 67, 83, 99, 91)
    Student("윤명원", 74, 91, 69, 95)

    Student.print_students()



if __name__ == "__main__":
    main()
