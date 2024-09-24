import numpy as np


def main():
    x = np.array([4,6,7,3,2])
    cond = np.array([1,2,0,2,2,2,1,0,1,0])
    cond2 = np.random.randint(0, 5, 30)
    print(x[cond])
    print(x[cond2])

if __name__ == "__main__":
    main()
