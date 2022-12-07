import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def demo():
    scale = StandardScaler()

    df = pandas.read_csv("cars2.csv")

    X = df[['Weight', 'Volume2']]
    y = df['CO2']

    scaledX = scale.fit_transform(X)

    regr = linear_model.LinearRegression()
    regr.fit(scaledX, y)

    scaled_2300_1300 = scale.transform([[2300, 1.3]])
    predictedCO2_2300_1300 = regr.predict([scaled_2300_1300[0]])
    print(predictedCO2_2300_1300)
    scaled_3300_1300 = scale.transform([[3300, 1.3]])
    predictedCO2_3300_1300 = regr.predict([scaled_3300_1300[0]])
    print(predictedCO2_3300_1300)