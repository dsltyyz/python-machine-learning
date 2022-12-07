import numpy
import matplotlib.pyplot as plt

def demo():
    x = numpy.random.normal(5.0, 1.0, 100000)
    # 柱状图
    plt.hist(x, 100)
    plt.show()