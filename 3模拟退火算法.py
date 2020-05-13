import mpl_toolkits.axisartist as axisartist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from goalfunction import f1, f2

class SA:

    def __init__(self, pop=20):
        self._max = 4
        self._min = -4
        self.population = pop

        self.Gfit = np.inf
        self.Gcoor = None
        
        self.swarms = np.random.uniform(
            self._min, self._max, [self.population, 2])
    
    def Disturb(self): 
        #给旧点以随机扰动
        newSwarms = np.zeros([self.population, 2])
        for index, sw in enumerate(self.swarms):
            if np.random.randint(0, 2):
                newSwarms[index][0] = sw[0] + (self._max-sw[0])*np.random.random()*0.01
            else:
                newSwarms[index][0] = sw[0] + (self._min-sw[0])*np.random.random()*0.01
            if np.random.randint(0, 2):
                newSwarms[index][1] = sw[1] + (self._max-sw[1])*np.random.random()*0.01
            else:
                newSwarms[index][1] = sw[1] + (self._min-sw[1])*np.random.random()*0.01
        return newSwarms
    
    def Calculation(self, swarms):
        """计算适应度"""
        fit = f2(swarms[:, 0], swarms[:, 1])
        if fit.min() < self.Gfit:
            self.Gfit = fit.min()
            self.Gcoor = swarms[fit.argmin()]
        return fit

    def update(self, T):
        """更新搜索位置"""
        fit1 = self.Calculation(self.swarms)
        newswarms = self.Disturb()
        fit2 = self.Calculation(newswarms)

        Delta = fit2 - fit1
        for index, d in enumerate(Delta):
            if d < 0:
                self.swarms[index] =newswarms[index]
            else:
                prob = np.exp(-d/T)
                # print("---", d, prob)
                rand = np.random.random()
                if prob > rand:
                    self.swarms[index] = newswarms[index]
        return fit2.min(), fit2.mean()

    def iteration(self, Tmax=1e5, alpha=0.99, iters=100):
        """迭代"""
        avgArray = np.zeros(iters)
        pbestArray = np.zeros(iters)
        GbestArray = np.zeros(iters)
        route = np.zeros([iters, self.population, 2])

        self.T = Tmax
        self.Tmin = 1e-5
        for _iter in range(iters):
            pbest, avg = self.update(self.T)
            self.T *= alpha
            route[_iter] = self.swarms
            avgArray[_iter] = avg
            pbestArray[_iter] = pbest
            GbestArray[_iter] = self.Gfit
        return avgArray, pbestArray, GbestArray, route


if __name__ == "__main__":
    N = 10000
    sa = SA(20)
    avgArray, pbestArray, GbestArray, route = sa.iteration(iters=N)
    print(sa.Gcoor, sa.Gfit)

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
    ax.annotate(s='适应度值', xy=(1, 1), xytext=(1, 1))
    ax.annotate(s='迭代/次', xy=(N, 0), xytext=(N, 0))
    plt.plot(avgArray, "-.", lw=2)
    plt.plot(pbestArray, ":", lw=2)
    plt.plot(GbestArray, "--", lw=2)
    plt.legend(["avgArray", "pbestArray", "GbestArray"], loc="upper center")
    plt.show()

    plt.figure()
    delta = 0.1
    X1 = np.arange(-4, 4.1, delta)
    Y1 = np.arange(-4, 4.1, delta)
    X, Y = np.meshgrid(X1, Y1)
    Z = f2(X, Y)
    """
    二维等高线图figsize=(5, 4), dpi=200
    """
    # 20表示绘制的等高线数量=20
    contour = plt.contour(X, Y, Z, 20, linewidths=0.5, linestyles="-.")
    # 等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    plt.clabel(contour, fontsize=4, colors=('k', 'w'))
    for i in range(sa.population):
        plt.scatter(route[:, i, 0], route[:, i, 1], s=2)
        plt.text(route[0, i, 0], route[0, i, 1], "0")
        plt.text(route[-1, i, 0], route[-1, i, 1], "-1")
    plt.show()



