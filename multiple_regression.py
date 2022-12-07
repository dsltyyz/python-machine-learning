import pandas
from sklearn import linear_model

def demo():
    df = pandas.read_csv("cars.csv")
    X = df[['Weight', 'Volume']]
    y = df['CO2']

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print(regr.coef_)

    # 预测重量为 2300kg、排量为 1300ccm 的汽车的二氧化碳排放量：
    predictedCO2_2300_1300 = regr.predict([[2300, 1300]])
    print(predictedCO2_2300_1300)
    predictedCO2_3300_1300 = regr.predict([[3300, 1300]])
    print(predictedCO2_3300_1300)