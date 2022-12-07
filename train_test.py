import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def demo():
    numpy.random.seed(2)

    x = numpy.random.normal(3, 1, 100)
    y = numpy.random.normal(150, 40, 100) / x

    train_x = x[:80]
    train_y = y[:80]

    test_x = x[80:]
    test_y = y[80:]

    # 多项式回归线
    mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
    myline = numpy.linspace(0, 6, 100)
    plt.scatter(train_x, train_y)
    plt.plot(myline, mymodel(myline))
    plt.show()

    # 求取5分钟的花费
    print(mymodel(5))
    # 训练集匹配度
    r1 = r2_score(train_y, mymodel(train_x))
    print(r1)
    # 测试集匹配度
    r2 = r2_score(test_y, mymodel(test_x))
    print(r2)

