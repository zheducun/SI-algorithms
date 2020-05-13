import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from goalfunction import city2 as cities


def cities_dist(cities):
    """np.linalg.norm:求范数，默认二范数"""
    num = cities.shape[0]
    dist = np.zeros([num, num])
    for xi in range(num):
        for yi in range(xi, num):
            dist[xi][yi] = dist[yi][xi] = np.linalg.norm(cities[xi]-cities[yi])
    return dist


dist = cities_dist(cities)
antsNum = 35
citiesNum = cities.shape[0]
alpha = 1
beta = 5
rho = 0.05
Q = 1
"""
eta:启发函数
pheromone: 信息素
paths:每只蚂蚁对应的路径
"""
eta = 1 / dist
eta[np.isinf(eta)] = 0
pheromone = np.ones([citiesNum, citiesNum])
paths = np.zeros([antsNum, citiesNum]).astype(int)
iters = 300
length_avg = np.zeros(iters)
length_best = np.zeros(iters)
path_best = np.zeros([iters, citiesNum]).astype(int)

plt.ion()
plt.figure()
for rank in range(iters):
# for rank in tqdm(range(iters)):
    lengths = np.zeros(antsNum)

    plt.plot(cities[:, 0], cities[:, 1], "k.", marker="s")
    xb = [cities[i][0] for i in path_best[rank-1]]+[cities[path_best[rank-1][0]][0]]
    yb = [cities[i][1] for i in path_best[rank-1]]+[cities[path_best[rank-1][0]][1]]
    plt.plot(xb, yb, "r-")
    plt.pause(0.2)
    # for i in range(pheromone.shape[0]):
    #     for j in range(i+1, pheromone.shape[0]):
    #         x = 0.5*(cities[i][0]+cities[j][0])
    #         y = 0.5*(cities[i][1]+cities[j][1])
    #         plt.text(x, y, s=f"[{pheromone[i][j]:.2f}]")

    # 初始化蚂蚁的起点
    if antsNum <= citiesNum:
        paths[:, 0] = np.random.permutation(range(citiesNum))[:antsNum]
    else:
        # 蚂蚁数比城市数多，需要补足
        paths[:citiesNum, 0] = np.random.permutation(range(0, citiesNum))
        paths[citiesNum:, 0] = np.random.permutation(range(0, citiesNum))[:antsNum-citiesNum]

    for ant in range(antsNum):
        # 当前位置
        visiting = paths[ant, 0]

        plt.plot(cities[visiting][0], cities[visiting][1], "r", marker="p")
        plt.text(cities[visiting][0], cities[visiting][1] -
                 1, str(ant+1), va="top", ha="center")

        # 当前蚂蚁访问过的城市
        visited = set()
        visited.add(visiting)
        # 未曾到访的城市
        unvisited = set(range(citiesNum))
        unvisited.remove(visiting)
        for city in range(1, citiesNum):
            """根据前路的信息素浓度比例，按概率选择——轮盘赌"""
            uvNum = len(unvisited)
            uvl = list(unvisited)
            prob = np.zeros(uvNum)
            for k in range(uvNum):
                prob[k] = np.power(pheromone[visiting][uvl[k]],
                                   alpha)*np.power(eta[visiting][uvl[k]], beta)
            prob_dtb = (prob/prob.sum()).cumsum()
            rand = np.random.rand()
            for index, item in enumerate(prob_dtb):
                if item > rand:
                    nextCity = uvl[index]
                    break
            paths[ant, city] = nextCity
            unvisited.remove(nextCity)
            lengths[ant] += dist[visiting][nextCity]

            # s = "b:"
            # plt.plot([cities[visiting][0], cities[nextCity][0]],
            #          [cities[visiting][1], cities[nextCity][1]], s, alpha=0.5)
            # plt.legend([f"length_best: {length_best[rank-1]:.2f}",
            #             f"rank {rank} {lengths[ant]:.1f}"], loc="upper center")
            # plt.pause(0.1)

            visiting = nextCity
        lengths[ant] += dist[visiting][paths[ant, 0]]
        
        # plt.plot([cities[visiting][0], cities[paths[ant, 0]][0]],
        #          [cities[visiting][1], cities[paths[ant, 0]][1]], s, alpha=0.5)
        # plt.legend([f"length_best: {length_best[rank-1]:.2f}",
        #             f"rank {rank} {lengths[ant]:.1f}"], loc="upper center")
        # plt.pause(0.1)

        length_avg[rank] = lengths.mean()
        if rank == 0:
            length_best[rank] = lengths.min()
            path_best[rank] = paths[lengths.argmin()]
        else:
            if lengths.min() > length_best[rank-1]:
                length_best[rank] = length_best[rank-1]
                path_best[rank] = path_best[rank-1]
            else:
                length_best[rank] = lengths.min()
                path_best[rank] = paths[lengths.argmin()]
        
    delta_ph = np.zeros([citiesNum, citiesNum])
    for ant in range(antsNum):
        for city in range(citiesNum):
            delta_ph[paths[ant, city-1]][paths[ant, city]] += Q / \
                dist[paths[ant, city-1]][paths[ant, city]]
    pheromone = (1-rho) * pheromone + delta_ph

    pheromone[pheromone > 5] = 2
    pheromone[pheromone < 0.5] = 1
    plt.clf()
    print(rank, f"{length_best[rank]:.3f}", f"{length_avg[rank]:.1f}")
plt.ioff()
plt.figure()
plt.plot(length_avg)
plt.plot(length_best)
plt.show()
plt.figure()
plt.plot(cities[:, 0], cities[:, 1], "k.", marker="s")
xb = [cities[i][0] for i in path_best[-1]]+[cities[path_best[-1][0]][0]]
yb = [cities[i][1] for i in path_best[-1]]+[cities[path_best[-1][0]][1]]
for i in range(pheromone.shape[0]):
    for j in range(i+1, pheromone.shape[0]):
        x = 0.5*(cities[i][0]+cities[j][0])
        y = 0.5*(cities[i][1]+cities[j][1])
        plt.text(x, y, s=f"[{pheromone[i][j]:.2f}]")
plt.plot(xb, yb, "r-")
plt.show()
