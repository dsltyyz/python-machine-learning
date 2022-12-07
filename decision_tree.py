import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import os


def demo():
    # 定义环境变量
    os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'

    # 导入数据
    df = pandas.read_csv("shows.csv")

    # 数据处理
    d = {'UK': 0, 'USA': 1, 'N': 2}
    df['Nationality'] = df['Nationality'].map(d)
    d = {'YES': 1, 'NO': 0}
    df['Go'] = df['Go'].map(d)
    # print(df)

    # 测试数据分轴
    features = ['Age', 'Experience', 'Rank', 'Nationality']
    X = df[features]
    y = df['Go']
    # print(X)
    # print(y)

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    # data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    # graph = pydotplus.graph_from_dot_data(data)
    # graph.write_png('mydecisiontree.png')
    #
    # img = pltimg.imread('mydecisiontree.png')
    # imgplot = plt.imshow(img)
    # plt.show()

    print(dtree.predict([[34, 10, 20, 1]]))
    # print(dtree.predict([[40, 10, 6, 1]]))
