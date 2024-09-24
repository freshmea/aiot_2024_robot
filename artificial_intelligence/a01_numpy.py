import numpy as np


def main():
    li1 = [[1,2,3],[4,5,6],[7,8,9]]
    test_array = np.array(li1, dtype=np.int8)
    print(test_array.dtype)
    print(test_array.shape)
    print(test_array.ndim)
    print(test_array.size)
    for ele in test_array:
        print(ele)

if __name__ == "__main__":
    main()
