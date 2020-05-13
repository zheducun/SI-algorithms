import numpy as np
from goalfunction import f2, city2
from copy import deepcopy
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from tqdm import tqdm


class TSPGA:
    def __init__(self, pop=100):
        self.Gfit = np.inf
        self.Gcoor = None
        self.cities = city2
        self.num = self.cities.shape[0]
        self.population = pop
        self.cityDict = {rank: city for rank, city in enumerate(self.cities)}
        self.initial()
        self.pc = 0.9
        self.pm = 0.1

    def initial(self):
        self.M = np.arange(self.num)
        self.species = np.zeros([self.population, self.num], dtype=np.int)
        for rank in range(self.population):
            np.random.shuffle(self.M)
            self.species[rank] = deepcopy(self.M)

    def RouteLength(self, species):
        # 计算路径长度
        lengths = np.zeros(self.population)
        for rank, specie in enumerate(species):
            route = np.array([self.cityDict[i] for i in specie])
            s = 0
            for i in range(route.shape[0]):
                mse = np.power(route[i-1, :]-route[i, :], 2)
                s += np.sqrt(np.sum(mse))
            lengths[rank] = np.round(s, 2)
        num = lengths.argmin()
        if lengths.min() < self.Gfit:
            self.Gfit = lengths.min()
            self.Gcoor = deepcopy(species[num])
        return lengths, num

    def selection(self):
    
        newspecies = np.zeros([self.population, self.num], dtype=np.int)
        lengths, num = self.RouteLength(self.species)
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
        # fit = (1/lengths)**6
        fit = np.exp(-lengths)
        prob = fit/fit.sum()
        probRank = prob.cumsum()

        # plt.figure()
        # plt.plot(probRank)
        # plt.show()

        for rank in range(self.population):
            rand = np.random.random()
            for index, item in enumerate(probRank):
                if item > rand:
                    newspecies[rank] = self.species[index]
                    break
        self.species = deepcopy(newspecies)
        return lengths.mean(), lengths.min(), num

    def crossover(self):

        newSpecies = np.zeros([self.population, self.num], dtype=np.int)
        np.random.shuffle(self.species)
        _half = self.population // 2
        father = self.species[:_half]
        monther = self.species[_half:]

        for index in range(_half):
            son = father[index]
            daughter = monther[index]
            rand = np.random.random()
            if rand <= self.pc:
                newSon = np.zeros(self.num, dtype=np.int)
                newDaughter = np.zeros(self.num, dtype=np.int)
                # -------------------------------------------------------------
                # 保留半数基因序列rand2 = rand1+self.num//2
                # -------------------------------------------------------------
                rand1 = np.random.randint(0, self.num//2-1)
                rand2 = np.random.randint(self.num//2, self.num)

                newSon[rand1:rand2] = son[rand1:rand2]
                newDaughter[rand1:rand2] = daughter[rand1:rand2]
                indexs = np.array([list(son).index(i) for i in daughter[rand1:rand2]])
                son0 = np.delete(son, indexs, axis=0)
                indexs = np.array([list(daughter).index(i) for i in son[rand1:rand2]])
                daughter0 = np.delete(daughter, indexs, axis=0)

                newSon[:rand1] = daughter0[:rand1]
                newSon[rand2:] = daughter0[rand1:]
                newDaughter[:rand1] = son0[:rand1]
                newDaughter[rand2:] = son0[rand1:]

                newSpecies[2*index] = newSon
                newSpecies[2*index+1] = newDaughter
            else:
                newSpecies[2*index] = son
                newSpecies[2*index+1] = daughter
        self.species = deepcopy(newSpecies)

    def mutation(self):
        for index in range(self.population):
            rand = np.random.random()
            if rand <= self.pm:
                i, j , k = 0, 0, 0
                while (i-j)*(j-k)*(k-i) == 0:
                    i, j, k =  np.random.randint(0, self.num, 3)
                self.species[index, [i, j, k]] = self.species[index, [j, i, k]]

    def iterations(self, Iters=100):
        avgArray = np.zeros(Iters)
        pbestArray = np.zeros(Iters)
        GbestArray = np.zeros(Iters)
        route = np.zeros([Iters, self.num], dtype=np.int)
        for rank in tqdm(range(Iters)):
            avg, pbest, num = self.selection()
            route[rank] = self.Gcoor
            self.crossover()
            self.mutation()
            avgArray[rank] = avg
            pbestArray[rank] = pbest
            GbestArray[rank] = self.Gfit
        return avgArray, pbestArray, GbestArray, route


if __name__ == "__main__":
    N = 500
    ga = TSPGA(128)
    avgArray, pbestArray, GbestArray, route = ga.iterations(Iters=N)
    print(ga.Gcoor, ga.Gfit)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, GbestArray.min())
    ax.axis["x"].set_axisline_style("->", size=2.0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("->", size=2.0)
    ax.axis["y"].set_axis_direction('left')
    ax.annotate(s='适应度值', xy=(1, avgArray.max()), xytext=(1, avgArray.max()))
    ax.annotate(s='迭代/次', xy=(N, GbestArray.min()), xytext=(N, GbestArray.min()))
    plt.plot(avgArray, "-.", lw=2)
    plt.plot(pbestArray, ":", lw=2)
    plt.plot(GbestArray, "--", lw=2)
    plt.legend(["avgArray", "pbestArray", "GbestArray"])
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
    contour = plt.contour(X, Y, Z, 20, linewidths=0.5,
                        linestyles=":", colors=("k"))
    #等高线上标明z（即高度）的值，字体大小是4，颜色分别是黑色和红色
    plt.clabel(contour, fontsize=4, colors=('k', 'r'))
    plt.scatter(x, y, marker="D")
    for i, j in city2:
        plt.text(i, j-0.1, s=f"({i},{j})", va="top", ha="center")
    road = np.array([ga.cityDict[i] for i in ga.Gcoor])
    for index in range(road.shape[0]):
        m = road[[index-1, index], :]
        plt.plot(m[:, 0], m[:, 1], ":b")
    plt.legend([f"{ga.Gfit}"])
    plt.show()


    # plt.ion()
    # fig = plt.figure(figsize=(8, 8))
    # epoch = 0
    # for r in route:
    #     ax = axisartist.Subplot(fig, 1, 1, 1)
    #     fig.add_axes(ax)
    #     ax.axis[:].set_visible(False)
    #     ax.axis["x"] = ax.new_floating_axis(0, 0)
    #     ax.axis["x"].set_axisline_style("->", size=2.0)
    #     ax.axis["y"] = ax.new_floating_axis(1, 0)
    #     ax.axis["y"].set_axisline_style("->", size=2.0)
    #     ax.axis["y"].set_axis_direction('left')
    #     ax.annotate(s='向东/Km', xy=(4, 0), xytext=(4, 0))
    #     ax.annotate(s='向北/km', xy=(0, 4), xytext=(0, 4))

    #     #数字 表示绘制的等高线数量
    #     contour = plt.contour(X, Y, Z, 20, linewidths=0.5,
    #                           linestyles=":", colors=("k"))
    #     #等高线上标明z（即高度）的值，字体大小是4，颜色分别是黑色和红色
    #     plt.clabel(contour, fontsize=4, colors=('k', 'r'))
    #     plt.scatter(x, y, marker="D")
    #     for i, j in city2:
    #         plt.text(i, j-0.1, s=f"({i},{j})", va="top", ha="center")
    #     road = np.array([ga.cityDict[i] for i in ga.Gcoor])
    #     for index in range(road.shape[0]):
    #         m = road[[index-1, index], :]
    #         plt.plot(m[:, 0], m[:, 1], ":b")

    #     roadx = np.array([ga.cityDict[i] for i in r])
    #     for index in range(roadx.shape[0]):
    #         m = roadx[[index-1, index], :]
    #         plt.plot(m[:, 0], m[:, 1], "--k")
    #     plt.title(f"epoch {epoch}, {ga.Gfit}")
    #     epoch += 1
    #     plt.pause(0.1)
    #     plt.clf()

    # plt.ioff()
    # # plt.show()
