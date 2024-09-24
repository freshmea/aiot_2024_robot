import numpy as np
import pandas as pd


def main():
    x_1 = np.arange(1, 25).reshape(4,6)
    df_1 = pd.DataFrame(x_1)
    print(df_1)
    print(df_1[0])
    print(df_1.iloc[0, 1])
    
    # read_csv
    df_data = pd.read_csv("/home/aa/aiot_2024_robot/artificial_intelligence/data/test.csv")
    print(df_data)

if __name__ == "__main__":
    main()
