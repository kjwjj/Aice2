import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def Ml(df):
    # type이 object인행 찾기
    object_cols = df.select_dtypes(include = 'object').columns
    df = pd.get_dummies(data = df, columns = object_cols)
    # print(df.head(5))
    target = 'Churn'
    x = df.drop(columns = target)
    y = df[target]

    x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                          stratify = y,
                                                          random_state = 42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    # LogisticRegression()
    lg = LogisticRegression()
    lg.fit(x_train, y_train)

    # KNeighborsClassifier()
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(x_train, y_train)

    # DecisionTreeClassifier()
    dt = DecisionTreeClassifier(max_depth = 10, random_state = 42)
    dt.fit(x_train, y_train)

    # RandomForestClassifier()
    rf = RandomForestClassifier(n_estimators = 3, random_state = 42)
    rf.fit(x_train, y_train)

    # XGBClassifier()
    xgb = XGBClassifier(n_estimators = 3, random_state = 42)
    xgb.fit(x_train, y_train)

    # LGBMClassifier()
    lgbm = LGBMClassifier(n_estimators = 3, random_state = 42)
    lgbm.fit(x_train, y_train)

    # lgbm성능평가
    y_pred = lgbm.predict(x_valid)
    cm = confusion_matrix(y_valid, y_pred)
    sns.heatmap(cm, annot = True)
    plt.show()
    print(classification_report(y_valid, y_pred, zero_division = 1))