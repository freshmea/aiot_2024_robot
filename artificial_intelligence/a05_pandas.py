import numpy as np
import pandas as pd


def main():
    sr = pd.Series([1,2,3,4,5], name="number", dtype=np.int8,
                   index=list("abcde"))
    print(sr)
    print(sr['a'])
    x_1 = np.arange(1, 20)
    sr2 = pd.Series(x_1, name="x_1")
    print(sr2)
    

if __name__ == "__main__":
    main()
