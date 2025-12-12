import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def Ml(df):
    # type이 object인행 찾기
    object_cols = df.select_dtypes(include = 'object').columns
    df = pd.get_dummies(data = df, columns = object_cols)
    # print(df.head(5))
    target = ['Churn']
    x = df.drop(columns = target)
    y = df.loc[:,target]

    x_train, x_valid, y_train, t_valid = train_test_split(x, y,
                                                          stratify = y,
                                                          random_state = 42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

