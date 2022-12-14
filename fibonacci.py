import matplotlib.pyplot as plt

def demo():
    def fib(x):
        return 1 if x == 1 or x == 2 else fib(x - 1) + fib(x - 2)

    x = range(1, 11)
    map_y = map(fib, x)
    y = list(map_y)
    # 散点图
    plt.scatter(x, y)
    plt.show()