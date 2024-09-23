class Student:
    def __init__(self, name, korean, math, english, science):
        self.name =  name
        self.korean =  korean
        self.math =  math
        self.english = english
        self.science =  science

    def get_sum(self):
        return self.korean + self.math + self.english + self.science

    def get_average(self):
        self.sum = self.get_sum()
        return self.sum / 4
    
    def to_string(self):
        return f"{self.name}\t{self.get_sum()}\t{self.get_average():.2f}"

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
        print(student.to_string())
    
if __name__ == "__main__":
    main()
