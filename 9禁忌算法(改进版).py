import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def lengths(route):
    """这里i-1开始，可以计算一个闭环的路径"""
    s = 0
    for i in range(route.shape[0]):
        s += np.sqrt(np.sum(np.power(route[i-1, :]-route[i, :], 2)))
    return s

# print(lengths(cities))
def init_route(cities):
    """随机排序"""
    np.random.shuffle(cities)
    return cities

# print(init_route())
def candidate(route):
    """生成候选路径"""
    n = route.shape[0]
    i, j = 0, 0
    while i == j:
        i, j = np.random.randint(0, n, 2)
    route[[i, j], :] = route[[j, i], :]
    return route

def step(route, tabu):
    candidateSet = np.zeros([candLen+1, *cities.shape])
    candidateSet[0] = route
    count = 0
    while count < candLen:
        cs = candidate(route)
        if lengths(cs) not in tabu:
            candidateSet[count+1] = cs
            count += 1
    fit = lengths(candidateSet)
    return fit.min(), candidateSet[fit.argmin()]


def iteration(cities, iters):
    bflist = []
    tabu = []
    bestfit = np.inf
    bestroute = None
    route = init_route(cities)
    for rank in range(iters):
        pbest, pbPos = step(route, tabu)
        if pbest < bestfit:
            bestfit = pbest
            bestroute = pbPos
        bflist.append(bestfit)
        tabu.append(pbest)
        route = pbPos
        if len(tabu) > tabuLen:
            tabu.pop(0)
    plt.figure()
    plt.plot(bflist)
    plt.show()
    return bestfit, bestroute


cities = np.array([[0, 0], [1, 10], [2, 8], [9, 5], [1.3, 5], [7, 9],
                   [3, 3], [6, 4], [9, 8], [8.1, 6.8], [15, 16], [12.5, 18.6],
                   [14.8, 18.4], [2.3, 15.7], [9.1, 19]])
candLen = 100
tabuLen = 100
bestfit, bestroute = iteration(cities, 300)

plt.figure()
for index in range(cities.shape[0]):
    X = bestroute[[index-1, index], :]
    plt.plot(X[:, 0], X[:, 1], "k--")
plt.show()


