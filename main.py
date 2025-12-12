from preprocess import preprocessing
from visualization import visual
from Ml import Ml

if __name__ == '__main__':
    df = preprocessing()
    print('-----------전처리 완료-----------')
    visual(df)
    print('-----------시각화 완료-----------')
    Ml(df)
