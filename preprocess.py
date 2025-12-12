import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing():
    df = pd.read_csv('churn_data.csv')
    # print(df.head(5))

    # customerID 제거
    df = df.drop(columns = ['customerID'])
    print(df)

    # TotalCharges ' '값 -> 0으로 변환 그리고 float형으로 바꾸기
    df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    # print(df['TotalCharges'].dtype)

    # print(df['Churn'].value_counts())
    # Chunr값 'Yes' -> 1 'No' -> 0
    df['Churn'] = df['Churn'].map({'Yes' : 1, 'No' : 0})
    # print(df['Churn'].value_counts())

    # 결측치 비율 확인 -> 40% 이상인 컬럼 -> DeviceProtection제거
    # print(df.isnull().sum()/ len(df))
    df = df.drop(columns = ['DeviceProtection'])

    # 결측치 비율 40% 미만인 row 삭제
    df = df.dropna(axis = 0)
    # print(df.info())

    # SeniorCitizen 비율 확인
    # print(df['SeniorCitizen'].value_counts().plot(kind = 'bar'))

    # SeniorCitizen은 불균형이 심하므로 삭제
    df = df.drop(columns = ['SeniorCitizen'])

    return df
