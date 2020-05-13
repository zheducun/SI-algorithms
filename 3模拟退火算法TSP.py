import mpl_toolkits.axisartist as axisartist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from goalfunction import f1, f2, city1, city2
from copy import deepcopy
import numpy.random as random

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class SA:

    def __init__(self, pop=2, cities=city2):

        self._max = 4
        self._min = -4
        self.population = pop
        self.cities = cities

        self.Gfit = np.inf
        self.Gcoor = None
        self.cityDict = {rank: city for rank, city in enumerate(self.cities)}
        self.M = np.arange(self.cities.shape[0])
        self.inital()
        
    def inital(self):
        # 初始化
        self.swarms = np.zeros([self.population, self.cities.shape[0]], dtype=np.int)
        for rank in range(self.population):
            random.shuffle(self.M)
            self.swarms[rank] = self.M

    def Exchagne(self, swarms):
        # 扰动函数：将选择随机的两个位置进行交换
        n = self.swarms[0].shape[0]
        for swarm in swarms:
            i, j = 0, 0
            while i == j:
                i, j = np.random.randint(0, n, 2)
            swarm[i], swarm[j] = swarm[j], swarm[i]
        return swarms
    
    def RouteLength(self, swarms):
        # 计算路径长度
        lengths = np.zeros(self.population)
        for rank, swarm in enumerate(swarms):
            route = np.array([self.cityDict[i] for i in swarm])
            s = 0
            for i in range(route.shape[0]):
                mse = np.power(route[i-1, :]-route[i, :], 2)
                s += np.sqrt(np.sum(mse))
            lengths[rank] = np.round(s, 2)
        num = lengths.argmin()
        if lengths.min() < self.Gfit:
            self.Gfit = lengths.min()
            self.Gcoor = deepcopy(swarms[num])
        return lengths, num

    def update(self, T):
        """更新搜索位置"""
        length1, _ = self.RouteLength(self.swarms)
        newswarms = self.Exchagne(self.swarms)
        length2, num = self.RouteLength(newswarms)

        Delta = length2 - length1
        for index, d in enumerate(Delta):
            if d < 0:
                self.swarms[index] = newswarms[index]
            else:
                prob = min([max([np.exp(-d/T), 0.1]), 0.8])
                rand = np.random.random()
                if prob > rand:
                    self.swarms[index] = newswarms[index]
        return length2.min(), length2.mean(), num

    def iteration(self, Tmax=1000, alpha=0.99, iters=100):
        """迭代"""
        avgArray = np.zeros(iters)
        pbestArray = np.zeros(iters)
        GbestArray = np.zeros(iters)
        route = np.zeros([iters, self.swarms[0].shape[0]], dtype=np.int)
        self.Tmax = 1000
        self.Tmin = 0.001

        for _iter in tqdm(range(iters)):
            pbest, avg, num = self.update(self.Tmax)
            self.Tmax *= alpha
            route[_iter] = self.swarms[num]
            avgArray[_iter] = avg
            pbestArray[_iter] = pbest
            GbestArray[_iter] = self.Gfit
            
        return avgArray, pbestArray, GbestArray, route


if __name__ == "__main__":
    N = 500
    sa = SA(10)
    avgArray, pbestArray, GbestArray, route = sa.iteration(iters=N)
    print(sa.Gcoor, sa.Gfit)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 46)
    ax.axis["x"].set_axisline_style("->", size=2.0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("->", size=2.0)
    ax.axis["y"].set_axis_direction('left')
    ax.annotate(s='适应度值', xy=(1, 1), xytext=(1, 1))
    ax.annotate(s='迭代/次', xy=(N, 0), xytext=(N, 0))
    plt.plot(avgArray, "-.", lw=2)
    plt.plot(pbestArray, ":", lw=2)
    plt.plot(GbestArray, "--", lw=2)
    plt.legend(["avgArray", "pbestArray", "GbestArray"], loc="upper center")
    plt.show()

    """
    二维等高线图
    """
    delta = 0.1
    X1 = np.arange(-4.0, 4.1, delta)
    Y1 = np.arange(-4.0, 4.1, delta)
    X, Y = np.meshgrid(X1, Y1)
    Z = f2(X, Y)
    x, y = city2[:, 0], city2[:, 1]
    fig = plt.figure(figsize=(8, 8))
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["x"].set_axisline_style("->", size=2.0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("->", size=2.0)
    ax.axis["y"].set_axis_direction('left')
    ax.annotate(s='向东/Km', xy=(4, 0), xytext=(4, 0))
    ax.annotate(s='向北/km', xy=(0, 4), xytext=(0, 4))

    #数字 表示绘制的等高线数量
    contour = plt.contour(X, Y, Z, 20, linewidths=0.5, linestyles=":", colors=("k"))
    #等高线上标明z（即高度）的值，字体大小是4，颜色分别是黑色和红色
    plt.clabel(contour, fontsize=4, colors=('k', 'r'))
    plt.scatter(x, y, marker="D")
    for i, j in city2:
        plt.text(i, j-0.1, s=f"({i},{j})", va="top", ha="center")
    road = np.array([sa.cityDict[i] for i in sa.Gcoor])
    for index in range(road.shape[0]):
        m = road[[index-1, index], :]
        plt.plot(m[:, 0], m[:, 1], "--b")
    plt.show()
