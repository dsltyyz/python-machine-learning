import matplotlib.pyplot as plt
from scipy import stats


def demo():
    # 线性回归
    # 原始数据
    x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
    y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
    # 该方法返回线性回归的一些重要键值
    slope, intercept, r, p, std_err = stats.linregress(x, y)

    # 线性方程方法
    def myfunc(x):
        return slope * x + intercept

    # 获取描绘点的y轴列表数据
    mymodel = list(map(myfunc, x))

    # 原始数据散点图
    plt.scatter(x, y)
    # 新数据线性回归图
    plt.plot(x, mymodel)
    # 显示
    plt.show()
