"""
IWO是2006年由A. R. Mehrabian等提出的一种从自然界杂草进化原理演化而来的随机搜索算法，
模仿杂草入侵的种子空间扩散、生长、繁殖和竞争性消亡的基本过程，具有很强的鲁棒性和自适应性。
IWO算法是一种高效的随机智能优化算法，以群体中优秀个体来指导种群的进化，
以正态分布动态改变标准差的方式将由优秀个体产生的子代个体叠加在父代个体周围，
再经过个体之间的竞争，得到最优个体。算法兼顾了群体的多样性和选择力度。
IWO相比其他的进化算法拥有更大的搜索空间和更好的性能。
与GA相比，IWO算法简单，易于实现，不需要遗传操作算子，能简单有效地收敛问题的最优解，
是一种强有力的智能优化工具
"""
import numpy as np
import matplotlib.pyplot as plt
from goalfunction import f1, f2
import mpl_toolkits.axisartist as axisartist


class IWO:

    def __init__(self, population=32):
        """
        一定数据的杂草初始分布在搜索空间中，位置随机，个数根据实际情况调整
        比如设置20株野草，即population=20
        其位置在整个搜索区域随机分布，则weeds为一个20x2的矩阵，其值为A1 + A2·A3
        weeds 最初20株草的位置坐标
        """
        self._max = 4
        self._min = -4

        self.population = population
        weeds1 = np.random.uniform(size=(self.population, 2))
        weeds2 = np.array([[self._min, self._min] for i in range(self.population)])
        weeds3 = np.array([[self._max-self._min, 0], [0, self._max-self._min]])
        self.weeds = weeds2 + np.dot(weeds1, weeds3)

    def reproducing(self):
        """
        繁殖——确定每一株草的子代数量
        分布在整个搜索空间的父代，根据父代的适应值产生下一代种子，种子的个数由适应度值决定，
        适应值高的产生的种子多，低的个体产生种子数少。
        这里函数值越低，适应度越高，fitness = 1/func
        利用公式
        Ns = (f-fmin)/(fmax-fmin)(smax-smin) + smin生成子代个数
        """
        value = f1(self.weeds[:, 0], self.weeds[:, 1])
        # fit = np.exp(-value)
        fit = 1 / value
        # plt.ion()
        
        # plt.plot(fit)
        # plt.pause(0.5)
        # plt.clf()

        fmax = fit.max()
        fmin = fit.min()
        smax = 24
        smin = 4
        self.sn_array = np.zeros(self.weeds.shape[0], dtype=int)
        for index, f in enumerate(fit):
            Ns = (f-fmin) / (fmax-fmin) * (smax-smin) + smin
            self.sn_array[index] = int(Ns)

        return self.sn_array

    def spreading(self, iters, iter_):
        """
        空间扩散——确定子代的位置
        子代个体按照一定规律分布在父代个体周围，分布位置规律满足正态分布（父代为轴线（均值），
        标准差随着代数不断变化）。
        确定均值、方差：
        利用公式 delta = (iterM-iter)**n / (iterM)**n * (delta0-deltaF) + deltaF
        利用Monte Carlo方法进行位置的采样：
        x1 = avg1 + delta * x11
        x2 = avg2 + delta * x22
        x11 = sqrt(-2*ln(rand1)) * cosPhi
        x22 = sqrt(-2*ln(rand1)) * sinPhi
        cosPhi = 2*(2*rand2-1)*rand3/((2*rand2-1)**2+rand3**2)
        sinPhi = ((2*rand2-1)**2-rand3**2)/((2*rand2-1)**2+rand3**2)
        """

        delta0 = 2
        deltaF = 0.01
        iterM = iters
        n = 2
        new_weeds = list(self.weeds)
        delta = (iterM-iter_)**n / (iterM)**n * (delta0-deltaF) + deltaF
        for index1, item1 in enumerate(self.weeds):
            new_weeds.append(item1)
            for index2 in range(self.sn_array[index1]):

                avg1 = item1[0]
                avg2 = item1[1]

                rand1 = np.random.random()
                rand2 = np.random.random()
                rand3 = np.random.random()

                cosPhi = 2*(2*rand2-1)*rand3/((2*rand2-1)**2+rand3**2)
                sinPhi = ((2*rand2-1)**2-rand3**2)/((2*rand2-1)**2+rand3**2)

                x11 = np.sqrt(-2*np.log(rand1)) * cosPhi
                x22 = np.sqrt(-2*np.log(rand1)) * sinPhi

                x1 = avg1 + delta * x11
                x2 = avg2 + delta * x22

                if x1 < self._min:
                    x1 = self._min
                if x1 > self._max:
                    x1 = self._max
                if x2 < self._min:
                    x2 = self._min
                if x2 > self._max:
                    x2 = self._max

                new_weeds.append([x1, x2])
        self.weeds = np.array(new_weeds)
        return self.weeds

    def weed_out(self):
        """
        竞争淘汰
        当一次繁殖的个体数超过种群数量的上限时，将子代和父代一起排序，适应值低的个体将被清除。
        简化版：超过1000个，重新开始。。。
        tabu:禁忌表
        """
        new_weeds = np.zeros([self.population, 2])
        tabu = []
        # --------------------------------------------------------------
        # 轮盘赌
        # --------------------------------------------------------------
        value = f1(self.weeds[:, 0], self.weeds[:, 1])
        # fit = np.exp(-value)
        fit = 1 / value
        prob = fit/fit.sum()
        probRank = prob.cumsum()
        for rank in range(self.population):
            rand = np.random.random()
            for index, item in enumerate(probRank):
                if item > rand and index not in tabu:
                    new_weeds[rank] = self.weeds[index]
                    tabu.append(index)              
                    break
        self.weeds = new_weeds
        

    def evolution(self, iters):

        """
        进化过程
        init()-->reproducing()-->spreading()-->weed_out()
        """
        avgArray = np.zeros(iters)
        pbestArray = np.zeros(iters)
        GbestArray = np.zeros(iters)
        route = {}
        route[0] = self.weeds

        value = f1(self.weeds[:, 0], self.weeds[:, 1])
        Bfit = value.min()
        Bweed = self.weeds[value.argmin()]
        # plt.figure()
        for rank in range(iters):
            while self.weeds.shape[0] < 100:
                self.reproducing()
                self.spreading(iters, rank)
            value = f1(self.weeds[:, 0], self.weeds[:, 1])
            route[rank] = self.weeds
            avgArray[rank] = value.mean()
            pbestArray[rank] = value.min()
            if value.min() < Bfit:
                Bfit = value.min()
                Bweed = self.weeds[value.argmin()]
            GbestArray[rank] = Bfit
            self.weed_out()

        return Bweed, avgArray, pbestArray, GbestArray, route


if __name__ == "__main__":
    N = 100
    iwo = IWO(64)
    Bweed, avgArray, pbestArray, GbestArray, route = iwo.evolution(N)
    print(Bweed)
    
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 2)
    ax.axis["x"].set_axisline_style("->", size=2.0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("->", size=2.0)
    ax.axis["y"].set_axis_direction('left')
    ax.annotate(s='适应度值', xy=(1, 8), xytext=(1, 8))
    ax.annotate(s='迭代/次', xy=(N, 0), xytext=(N, 0))
    plt.plot(avgArray, "-.", lw=2)
    plt.plot(pbestArray, ":", lw=2)
    plt.plot(GbestArray, "--", lw=2)
    plt.legend(["avgArray", "pbestArray", "GbestArray"], loc="lower center")
    plt.show()
    
    # plt.ion()
    plt.figure()  
    for i in range(N):
        delta = 0.1
        X1 = np.arange(-4, 4.1, delta)
        Y1 = np.arange(-4, 4.1, delta)
        X, Y = np.meshgrid(X1, Y1)
        Z = f1(X, Y)
        """
        二维等高线图figsize=(5, 4), dpi=200
        """
        # 20表示绘制的等高线数量=20
        contour = plt.contour(X, Y, Z, 20, linewidths=0.5, linestyles="-.")
        # 等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
        plt.clabel(contour, fontsize=4, colors=('k', 'w'))
        plt.scatter(route[i][:, 0], route[i][:, 1], s=2)
        plt.axis("off")
        plt.title(f"epoch {i}")
        # plt.pause(0.5)
        # plt.clf()
    plt.show()
