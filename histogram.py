import numpy
import matplotlib.pyplot as plt

def demo():
    x = numpy.random.uniform(0.0, 5.0, 100000)
    # 柱状图
    plt.hist(x, 100)
    plt.show()