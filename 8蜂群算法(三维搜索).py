import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class BCO:

    def __init__(self):
        """
        nVar: 问题的维度
        nPop: 蜜蜂的数量
        nOnlooker: 跟随蜂数量
        Stay: 一个点固定的次数
        a: phi的系数
        staylim: 不更新次数极限，around四舍五入
        """
        self.nVar = 2
        self.Xlim = [-40, 40]
        self.Ylim = [-40, 40]
        self.nPop = 100
        self.nOnlooker = 100
        self.Stay = np.zeros(self.nPop)
        self.a = 1
        self.staylim = np.around(0.6*self.nVar*self.nPop)

    @staticmethod
    def tgfunc(x, y):
        x = x / 10
        y = y / 10
        z = (x**4-16*x**2+5*x)+(y**4-16*y**2+5*y) + 160
        # z = -20*np.exp(-0.2*np.sqrt((x*x+y*y)/2)) - \
        #     np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2)+20+np.exp(1)
        return z

    def initPos(self):
        """np.random.uniform:左闭右开，因此不够严谨"""
        self.k = self.Xlim[1] - self.Xlim[0]
        self.b = self.Xlim[1]
        self.Position = self.k * np.random.random([self.nPop, self.nVar]) - self.b
        self.fitness = self.tgfunc(self.Position[:, 0], self.Position[:, 1])

    def Hiredbee(self):
        for rank in range(self.nPop):
            index = rank
            while index != rank:
                index = np.random.randint(0, self.nPop)
            phi = self.a * (2*np.random.rand(self.nVar)-1)
            newPosition = self.Position[rank] + phi * \
                (self.Position[rank]-self.Position[index])
            fitness = self.tgfunc(newPosition[0], newPosition[1])
            if fitness < self.fitness[rank]:
                self.Position[rank] = newPosition
                self.fitness[rank] = fitness
            else:
                self.Stay[rank] += 1

    def Followbee(self):
        """
        跟随蜂
        """
        Avg = self.fitness.mean()
        prob = np.exp(-self.fitness/Avg)
        cumprob = (prob/prob.sum()).cumsum()
        for rank in range(self.nOnlooker):
            rand = np.random.random()
            for index, item in enumerate(cumprob):
                if rand <= item:break
            while True:
                index_ = np.random.randint(self.nPop)
                if index_ != index:break
            phi = self.a * (2*np.random.random(self.nVar)-1)
            newPosition = self.Position[rank] + phi * \
                (self.Position[rank]-self.Position[index])
            fitness = self.tgfunc(newPosition[0], newPosition[1])
            if fitness < self.fitness[rank]:
                self.Position[rank] = newPosition
                self.fitness[rank] = fitness
            else:
                self.Stay[rank] += 1

    def scouter(self):
        """
        当一个食物源许久不更新，则对其进行更新:重新随机位置。
        """
        for rank in range(self.nPop):
            if self.Stay[rank] >= self.staylim:
                self.Position[rank] = self.k * np.random.rand(1, self.nVar) - self.b
                self.fitness[rank] = self.tgfunc(self.Position[rank][0], self.Position[rank][1])
                self.Stay[rank] = 0
    
    def iteration(self, iters):
        """
        进行迭代时候到了。
        self.Gx, self.Gy:为最优解的x, y变量变化路径.
        """
        self.Gbest = np.zeros(iters)
        self.Pbest = np.zeros(iters)
        self.Pavg = np.zeros(iters)
        self.Px = np.zeros(iters)
        self.Py = np.zeros(iters)
        self.Gx, self.Gy = [], []
        self.route = np.zeros([iters, self.nPop, 2])
        self.initPos()
        self.Bestfit = self.fitness.min()
        for rank in range(iters):
            for index, item in enumerate(self.Position):
                self.route[rank][index] = item
            self.Pbest[rank] = self.fitness.min()
            self.Px[rank], self.Py[rank] = self.Position[self.fitness.argmin()][:]
            if self.Pbest[rank] <= self.Bestfit:
                self.Bestfit = self.Pbest[rank]
                self.Gx.append(self.Px[rank])
                self.Gy.append(self.Px[rank])
            self.Gbest[rank] = self.Bestfit
            self.Pavg[rank] = self.fitness.mean()
            self.Hiredbee()
            self.Followbee()
            self.scouter()
        return self.Bestfit, self.Gx[-1], self.Gy[-1]
import time
tb = time.time()
bco = BCO()
result = bco.iteration(100)
print(result)
print(time.time()-tb)
# plt.figure()
# delta = 1
# X1 = np.arange(-40, 40, delta)
# Y1 = np.arange(-40, 40, delta)
# X, Y = np.meshgrid(X1, Y1)
# Z = bco.tgfunc(X, Y)
# """
# 二维等高线图figsize=(5, 4), dpi=200
# """
# #20表示绘制的等高线数量=20
# contour = plt.contour(X, Y, Z, 20, linewidths=0.5, linestyles="-.")
# #等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
# plt.clabel(contour, fontsize=4, colors=('k', 'w'))
# for i in range(bco.nPop):
#     plt.scatter(bco.route[:, i, 0], bco.route[:, i, 1], s=2)
#     plt.text(bco.route[0, i, 0], bco.route[0, i, 1], "0")
#     plt.text(bco.route[-1, i, 0], bco.route[-1, i, 1], "-1")
# plt.show()
# fig = plt.figure()
# ax = axisartist.Subplot(fig, 1, 1, 1)
# fig.add_axes(ax)
# x_ = 100
# y_ = max(bco.Py.max(), bco.Px.max())
# xx_ = int(-28.9)
# yy_ = int(x_*0.02)
# ax.annotate(s='x', xy=(x_, xx_), xytext=(x_, xx_))
# ax.annotate(s='y', xy=(yy_, y_), xytext=(yy_, y_))
# ax.axis[:].set_visible(False)
# ax.axis["x"] = ax.new_floating_axis(0, -29)
# ax.axis["x"].set_axisline_style("->", size=2.0)
# ax.axis["y"] = ax.new_floating_axis(1, 0)
# ax.axis["y"].set_axisline_style("->", size=2.0)
# ax.axis["y"].set_axis_direction('left')
# plt.plot(bco.Px, ":")
# plt.plot(bco.Py, "--")
# plt.plot(bco.Gx, "--.")
# plt.plot(bco.Gy, "-.")
# plt.legend(["Px", "Py", "Gx", "Gy"], loc="upper center")
# plt.show()
fig = plt.figure()
ax = axisartist.Subplot(fig, 1, 1, 1)
fig.add_axes(ax)
x_ = 100
y_ = max(bco.Pbest.max(), bco.Pavg.max()*0.1)
xx_ = int(y_*0.1)
yy_ = int(x_*0.02)
ax.annotate(s='x', xy=(x_, xx_), xytext=(x_, xx_))
ax.annotate(s='y', xy=(yy_, y_), xytext=(yy_, y_))
ax.axis[:].set_visible(False)
ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["x"].set_axisline_style("->", size=2.0)
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["y"].set_axisline_style("->", size=2.0)
ax.axis["y"].set_axis_direction('left')
plt.plot(bco.Gbest, ":")
plt.plot(bco.Pbest, "--")
plt.plot(bco.Pavg*0.1, "--.")
plt.legend(["Gbest", "Pbest", "Pavg*0.1"], loc="upper center")
plt.show()
