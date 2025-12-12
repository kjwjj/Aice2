import matplotlib.pyplot as plt
import seaborn as sns

def visual(df):
    # histplot
    sns.histplot(data = df, x = 'tenure')
    plt.show()

    # kdeplot
    sns.kdeplot(data = df, x = 'tenure', hue = 'Churn')
    plt.show()

    # heatmap(컬럼간의 상관관계)
    sns.heatmap(df[['tenure','MonthlyCharges','TotalCharges']].corr(), annot = True)
    plt.show()