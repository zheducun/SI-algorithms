import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.axisartist as axisartist


def f1(x, y):
    return np.array((x**4-16*x**2+5*x)+(y**4-16*y**2+5*y)) + 160


def f2(x, y):
    result = -20*np.exp(-0.2*np.sqrt((x*x+y*y)/2)) - \
        np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2)+20+np.exp(1)
    return result


city1 = np.array([[0, 0], [1, 10], [2, 8], [9, 5], [1.3, 5], [7, 9],
                  [3, 3], [6, 4], [9, 8], [8.1, 6.8], [15, 16],
                  [12.5, 18.6], [14.8, 18.4], [2.3, 15.7], [9.1, 19]])
city2 = np.array([[2.35, 1.64], [-0.08, 2.04], [-2.06, -3.0], [-0.95, 1.81],
                  [-3.58, 3.15], [0.69, 2.33], [0.3, -0.99], [0.62, 2.02],
                  [0.42, -0.66], [-2.84, 3.23], [2.44, 1.29], [-1.08, -0.49],
                  [3.52, -2.17], [2.64, 2.1], [2.02, 3.68], [0.31, -1.91],
                  [3.61, 0.65], [1.04, 0.94], [-1.16, 2.44], [1.66, -3.53]])

if __name__ == "__main__":
    dataset = {}
    delta = 0.1
    X1 = np.arange(-4.0, 4.1, delta)
    Y1 = np.arange(-4.0, 4.1, delta)
    X, Y = np.meshgrid(X1, Y1)
    Z = f1(X, Y)
    # data = np.zeros([Z.size, 3])

    # for i, x in enumerate(X1):
    #     for j, y in enumerate(Y1):
    #         data[i*X1.shape[0]+j] = np.array([Z[i][j], x, y])
    # dataset["y0"] = data[:, 0]
    # dataset["x1"] = data[:, 1]
    # dataset["x2"] = data[:, 2]
    # df = pd.DataFrame(dataset, columns=dataset.keys())
    # df.to_csv("datas_00f.csv")

    # """
    # 二维等高线图
    # """
    # fig = plt.figure(figsize=(5, 4), dpi=200)
    # #填充颜色，f即filled
    # suf = plt.contourf(X, Y, Z, alpha=0.5)
    # fig.colorbar(suf)
    # # #20表示绘制的等高线数量=20
    # contour = plt.contour(X, Y, Z, 20, linewidths=0.5, linestyles="-.")
    # #等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    # plt.clabel(contour, fontsize=4, colors=('k', 'r'))
    # # plt.grid(1)
    # # plt.axis("off")
    # plt.show()
    """
    三维表面图形
    """
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap="rainbow", linewidth=0, antialiased=False)
    # ax.contour(X, Y, Z, offset=-2, cmap='rainbow')
    # # plt.axis("off")
    # plt.show()

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["x"].set_axisline_style("->", size=2.0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("->", size=2.0)
    ax.axis["y"].set_axis_direction('left')
    ax.annotate(s='向东/Km', xy=(1, 0), xytext=(15, 1))
    ax.annotate(s='向北/km', xy=(0, 1), xytext=(0.5, 18))
    x = city2[:, 0]
    y = city2[:, 1]
    plt.scatter(x, y, marker="D")
    for i, j in city2:
        plt.text(i, j-0.1, s=f"({i},{j})", va="top", ha="center")
    plt.show()
