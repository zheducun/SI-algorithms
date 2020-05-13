import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from goalfunction import f1, f2


class PSO:

    def __init__(self):
        """粒子群算法"""
        self.population = 128
        self.dimension = 2
        self.X = np.random.uniform(-4, 4, [self.population, self.dimension])
        self.v = np.random.uniform(-0.4, 0.4, [self.population, self.dimension])

        self.Pfit = f2(self.X[:, 0], self.X[:, 1])
        self.Gfit = self.Pfit.min()
        self.Pbest = self.X
        self.Gbest = self.X[np.argmin(self.Pfit)]

    def move(self):
        self.weight = 0.5
        self.c1 = 0.2
        self.c2 = 0.01
        self.r1 = np.random.uniform(0, 1, [self.population, 1])
        self.r2 = np.random.uniform(0, 1, [self.population, 1])

        self.v = self.weight*self.v + self.r1*self.c1 * \
            (self.Pbest-self.X) + self.r2*self.c2*(self.Gbest-self.X)
        self.v[self.v > 0.4] = 0.4
        self.v[self.v < -0.4] = -0.4

        self.X = self.X + self.v
        self.X[self.X > 4] = 4
        self.X[self.X < -4] = -4

    def iteration(self):
        self.move()
        fit = f2(self.X[:, 0], self.X[:, 1])
        for rank, item in enumerate(fit-self.Pfit):
            if item < 0:
                self.Pfit[rank] = fit[rank]
                self.Pbest[rank] = self.X[rank]
        if self.Pfit.min() < self.Gfit:
            self.Gfit = self.Pfit.min()
            self.Gbest = self.X[np.argmin(self.Pfit)]
        
    def draw(self, rank):

        delta = 1
        X1 = np.arange(-41, 41, delta)
        Y1 = np.arange(-41, 41, delta)
        X, Y = np.meshgrid(X1, Y1)
        Z = f2(X, Y)
        """
        二维等高线图figsize=(5, 4), dpi=200
        """
        fig = plt.figure()
        #填充颜色，f即filled
        suf = plt.contourf(X, Y, Z, alpha=0.5)
        fig.colorbar(suf)
        #20表示绘制的等高线数量=20
        contour = plt.contour(X, Y, Z, 20, linewidths=0.5, linestyles="-.")
        #等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
        plt.clabel(contour, fontsize=4, colors=('k', 'w'))
        
        plt.scatter(self.X[:, 0], self.X[:, 1], s=2, marker="1", color="r")
        for xx in self.X:
            plt.text(
                xx[0], xx[1], f"{f2(xx[0], xx[1]):.3f}", fontsize=4, va="top", ha="center")
        plt.title(f"population {self.population} rank {rank} gbest {self.Gfit:.6f}", fontsize=6)
        plt.axis("off")
        plt.savefig(f"imgs/pso/a{rank:03d}.png", dpi=200)
        plt.close()
        # plt.show()

    def draw1(self, *args, xp0=0, yp0=0, xlm=40, ylm=40, mark=1):
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig = plt.figure()
        ax = axisartist.Subplot(fig, 1, 1, 1)
        fig.add_axes(ax)
        ax.annotate(s='x/km', xy=(xlm, yp0+1), xytext=(xlm, yp0+1))
        ax.annotate(s='y/km', xy=(xp0+1, ylm), xytext=(xp0+1, ylm))
        ax.axis[:].set_visible(False)
        ax.axis["x"] = ax.new_floating_axis(0, xp0)
        ax.axis["x"].set_axisline_style("->", size=2.0)
        ax.axis["y"] = ax.new_floating_axis(1, yp0)
        ax.axis["y"].set_axisline_style("->", size=2.0)
        ax.axis["y"].set_axis_direction('left')
        for rank, y in enumerate(args):
            plt.plot(y, "-", label=f"{rank}")
        plt.legend(loc="upper center")
        if mark == 1:
            plt.savefig(f"{pso.weight}_{pso.c1}_{pso.c2}xy.png", dpi=200)
        else:
            plt.savefig(f"{pso.weight}_{pso.c1}_{pso.c2}fit.png", dpi=200)
        plt.show()


if __name__ == "__main__":
    N = 1000
    pso = PSO()
    Gbest = []
    Gavg = []
    Gx = []
    Gy = []
    route = np.zeros([N, pso.population, 2])
    # plt.figure()
    # plt.ion()
    for i in range(N):
        # pso.draw(i)
        for j, item in enumerate(pso.X):
            route[i][j] = item
        Gx.append(pso.Gbest[0])
        Gy.append(pso.Gbest[1])
        Gavg.append(pso.Pfit.mean())
        Gbest.append(pso.Gfit)
        pso.iteration()


        # plt.plot(Gbest, ":")
        # plt.plot(i, pso.Gfit, ">")
        # plt.text(i, pso.Gfit, f"{pso.Gfit:.9f}", va="top")
        # # plt.savefig(f"imgs/pso/b{i:03d}.png", dpi=200)
        # plt.xlim(0, N)
        # plt.ylim(0, 1)
        # plt.pause(0.01)
        # plt.clf()


    # plt.ioff()
    # plt.close()
    pso.draw1(Gavg, Gbest, xlm=1000, ylm=max(Gavg), mark=0)
    pso.draw1(Gx, Gy, xp0=-29, yp0=0, xlm=100)
    print(f"{pso.Gfit:.7f},\n{np.round(pso.Gbest, 6)}")
    plt.figure()
    delta = 0.1
    X1 = np.arange(-4.1, 4.1, delta)
    Y1 = np.arange(-4.1, 4.1, delta)
    X, Y = np.meshgrid(X1, Y1)
    Z = f2(X, Y)
    """
    二维等高线图figsize=(5, 4), dpi=200
    """
    #20表示绘制的等高线数量=20
    contour = plt.contour(X, Y, Z, 20, linewidths=0.5, linestyles="-.")
    #等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    plt.clabel(contour, fontsize=4, colors=('k', 'w'))
    for i in range(pso.population):
        plt.scatter(route[:, i, 0], route[:, i, 1], s=2)
        plt.text(route[0, i, 0], route[0, i, 1], "0")
        plt.text(route[-1, i, 0], route[-1, i, 1], "-1")
    
    plt.legend([f"w {pso.weight} c1 {pso.c1} c2 {pso.c2}"], loc="upper center")
    plt.savefig(f"{pso.weight}_{pso.c1}_{pso.c2}.png", dpi=200)
    plt.show()
