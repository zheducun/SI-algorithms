import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np
from tqdm import tqdm
from goalfunction import f1 as f2


class GA:

    def __init__(self, pop=64):
        """
        Dimension 维度，所研究的问题设计的参数维度（向量维度）
        GeneLen 基因长度，编码长度（涉及精度）,52为极限
        Population 种群规模，（一次计算的规模）
        Iters 迭代总次数
        Speices 物种，一个世代的所有基因
        elite 精英，最优个体
        """
        self.Dimension = 2
        self.GeneLen = 52
        self.Population = pop
        self.Species = self.inital_species()
        self.elite = None
        self.bestfitness = 0
        self.pc = 0.8
        self.pm = 0.1

    def inital_species(self):
        """二进制编码"""
        return np.random.randint(0, 2, [self.Population, self.GeneLen])

    def decoding(self, A, _max=4, _min=-4):
        """解码操作，将二进制数转成十进制数"""
        B = np.zeros([A.shape[0], 2], dtype=np.float64)
        _half = self.GeneLen//2
        b1 = np.array([2**(_half-1-i) for i in range(_half)])
        b2 = np.array([2**(_half-1-i) for i in range(_half)])
        for index, a in enumerate(A):
            a1 = a[:_half]
            a2 = a[_half:]
            x = _min + (_max-_min) * a1.dot(b1.T) / (2**_half-1)
            y = _min + (_max-_min) * a2.dot(b1.T) / (2**_half-1)
            B[index][0] = x
            B[index][1] = y
        return B

    def selection(self):

        newSpecies = np.zeros([self.Population, self.GeneLen])
        B = self.decoding(self.Species)
        # ----------------------------------------------------
        # 两种适应度
        # ----------------------------------------------------
        self.fitness = np.exp(-f2(B[:, 0], B[:, 1]))
        # self.fitness = 1 / f2(B[:, 0], B[:, 1])
        if self.fitness.max() > self.bestfitness:
            self.bestfitness = self.fitness.max()
            self.elite = self.Species[self.fitness.argmax()]

        # ----------------------------------------------------
        # 传说中的联赛选择
        # ----------------------------------------------------
        # for rank in range(self.Population):
        #     n1, n2 = np.random.randint(0, self.Population, 2)
        #     choice = n1 if self.fitness[n1] > self.fitness[n2] else n2
        #     newSpecies[rank] = self.Species[choice]
        # self.Species = newSpecies
        # ----------------------------------------------------
        # 俄罗斯轮盘赌
        # ----------------------------------------------------
        prob = self.fitness/self.fitness.sum()
        probRank = prob.cumsum()
        for rank in range(self.Population):
            rand = np.random.random()
            for index, item in enumerate(probRank):
                if item > rand:
                    newSpecies[rank] = self.Species[index]
                    break
        self.Species = newSpecies
        return self.fitness.mean(), self.fitness.max()

    def crossover(self):

        newSpecies = np.zeros([self.Population, self.GeneLen])
        np.random.shuffle(self.Species)
        _half = self.Population // 2
        father = self.Species[:_half]
        monther = self.Species[_half:]
        for index in range(_half):
            son = father[index]
            daughter = monther[index]
            rand = np.random.random()
            if rand <= self.pc:
                rand1 = np.random.randint(0, self.GeneLen-1)
                rand2 = np.random.randint(rand1+1, self.GeneLen)
                # -------------------------------------------------------------
                # 对应位置交换基因序列
                # -------------------------------------------------------------
                son[rand1: rand2], daughter[rand1: rand2] =\
                    daughter[rand1: rand2], son[rand1: rand2]

            newSpecies[2*index] = son
            newSpecies[2*index+1] = daughter
        self.Species = newSpecies

    def mutation(self):

        for index in range(1, self.Population):
            for rank in range(self.GeneLen):
                rand = np.random.random()
                if rand <= self.pm:
                    self.Species[index][rank] = 1 - self.Species[index][rank]
        return 0

    def iterations(self, Iters=100):
        avgArray = np.zeros(Iters)
        pbestArray = np.zeros(Iters)
        GbestArray = np.zeros(Iters)
        route = np.zeros([Iters, self.Population, 2])
        for rank in range(Iters):
            for j, item in enumerate(self.decoding(self.Species)):
                route[rank][j] = item
            avg, pbest = self.selection()
            self.crossover()
            self.mutation()
            avgArray[rank] = avg
            pbestArray[rank] = pbest
            GbestArray[rank] = self.bestfitness
        return avgArray, pbestArray, GbestArray, route


if __name__ == "__main__":
    N = 500
    ga = GA(128)
    avgArray, pbestArray, GbestArray, route = ga.iterations(Iters=N)
    B = ga.decoding(ga.elite.reshape(1, ga.GeneLen))
    print(B)
    print(f2(B[:, 0], B[:, 1]))

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0.2)
    ax.axis["x"].set_axisline_style("->", size=2.0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("->", size=2.0)
    ax.axis["y"].set_axis_direction('left')
    ax.annotate(s='适应度值', xy=(1, 1), xytext=(1, 1))
    ax.annotate(s='迭代/次', xy=(N, 0.2), xytext=(N, 0.2))
    plt.plot(avgArray, "-.", lw=2)
    plt.plot(pbestArray, ":", lw=2)
    plt.plot(GbestArray, "--", lw=2)
    plt.legend(["avgArray", "pbestArray", "GbestArray"], loc="lower center")
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
    for i in range(ga.Population):
        plt.scatter(route[:, i, 0], route[:, i, 1], s=2)
        plt.text(route[0, i, 0], route[0, i, 1], "0")
        plt.text(route[-1, i, 0], route[-1, i, 1], "-1")
    plt.show()
