import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def demo():
    x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
    y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

    # 多项式模型
    mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
    # 指定行的显示方式
    myline = numpy.linspace(1, 22, 100)
    # 绘制原始散点图
    plt.scatter(x, y)
    # 画出多项式回归线
    plt.plot(myline, mymodel(myline))
    plt.show()

    # 多项式回归中的拟合度
    print(r2_score(y, mymodel(x)))
    # 输出预测值
    print(mymodel(17))