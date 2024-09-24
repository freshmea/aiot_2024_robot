import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def main():
    df_1 = pd.read_csv("/home/aa/aiot_2024_robot/artificial_intelligence/data/iris.tab", delimiter="\t")
    print(df_1.head())
    print(df_1.info())
    model = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
    target = 'iris'
    feature = df_1.columns.drop('iris')
    model.fit(df_1[feature], df_1[target])

    # 결과 출력
    print(model.classes_)
    print()
    print(model.coef_)

    df_2 = pd.read_csv("/home/aa/aiot_2024_robot/artificial_intelligence/data/iris_predict.tab", delimiter="\t")
    predicted = model.predict(df_2[feature])
    print(predicted)

    cm = confusion_matrix(df_2['iris'], predicted)
    print(cm)


if __name__ == "__main__":
    main()
