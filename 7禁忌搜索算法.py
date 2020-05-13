import random
# import numpy as  np

# cities = np.array([[0, 0], [1, 10], [2, 8], [9, 5], [1.3, 5], [7, 9],
#                    [3, 3], [6, 4], [9, 8], [8.1, 6.8], [15, 16], [12.5, 18.6],
#                    [14.8, 18.4], [2.3, 15.7], [9.1, 19]])


class Tabu:
    def __init__(self, tabulen=100, preparelen=200):
        self.tabulen = tabulen
        self.preparelen = preparelen
        self.city, self.cityids, self.stid = self.loadcity2()  # 我直接把他的数据放到代码里了

        self.route = self.randomroute()
        self.tabu = []
        self.prepare = []
        self.curroute = self.route.copy()
        self.bestcost = self.costroad(self.route)
        self.bestroute = self.route

    def loadcity2(self, stid=1):
        city = {1: (1150.0, 1760.0), 2: (630.0, 1660.0), 3: (40.0, 2090.0), 4: (750.0, 1100.0),
                5: (750.0, 2030.0), 6: (1030.0, 2070.0), 7: (1650.0, 650.0), 8: (1490.0, 1630.0),
                9: (790.0, 2260.0), 10: (710.0, 1310.0), 11: (840.0, 550.0), 12: (1170.0, 2300.0),
                13: (970.0, 1340.0), 14: (510.0, 700.0), 15: (750.0, 900.0), 16: (1280.0, 1200.0),
                17: (230.0, 590.0), 18: (460.0, 860.0), 19: (1040.0, 950.0), 20: (590.0, 1390.0),
                21: (830.0, 1770.0), 22: (490.0, 500.0), 23: (1840.0, 1240.0), 24: (1260.0, 1500.0),
                25: (1280.0, 790.0), 26: (490.0, 2130.0), 27: (1460.0, 1420.0), 28: (1260.0, 1910.0),
                29: (360.0, 1980.0)}  # 原博客里的数据
        cityid = list(city.keys())
        return city, cityid, stid

    def costroad(self, road):
        #计算当前路径的长度 与原博客里的函数功能相同
        d = -1
        st = 0, 0
        cur = 0, 0
        city = self.city
        for v in road:
            if d == -1:
                st = city[v]
                cur = st
                d = 0
            else:
                # 计算所求解的距离，这里为了简单，视作二位平面上的点，使用了欧式距离
                d += ((cur[0]-city[v][0])**2+(cur[1]-city[v][1]) **2)**0.5  
                cur = city[v]
        d += ((cur[0]-st[0])**2+(cur[1]-st[1])**2)**0.5
        return d

    def randomroute(self):
        #产生一条随机的路
        stid = self.stid
        rt = self.cityids.copy()
        random.shuffle(rt)
        rt.pop(rt.index(stid))
        rt.insert(0, stid)
        return rt

    def randomswap2(self, route):
        #随机交换路径的两个节点
        route = route.copy()
        while True:
            a = random.choice(route)
            b = random.choice(route)
            if a == b or a == 1 or b == 1:
                continue
            ia = route.index(a)
            ib = route.index(b)
            route[ia] = b
            route[ib] = a
            return route

    def step(self):
        #搜索一步路找出当前下应该搜寻的下一条路
        rt = self.curroute
        i = 0
        while i < self.preparelen:  # 产生候选路径
            prt = self.randomswap2(rt)
            if int(self.costroad(prt)) not in self.tabu:  # 产生不在禁忌表中的路径
                self.prepare.append(prt.copy())
                i += 1
        c = [] 
        for r in self.prepare:
            c.append(self.costroad(r))
        mc = min(c)
        mrt = self.prepare[c.index(mc)]  # 选出候选路径里最好的一条
        if mc < self.bestcost:
            self.bestcost = mc
            self.bestroute = mrt.copy()  # 如果他比最好的还要好，那么记录下来
        self.tabu.append(mc)
        #也就是说 每个路径和他的长度是一一对应，这样比对起来速度快点，当然这样可能出问题，更好的有待研究
        self.curroute = mrt  # 用候选里最好的做下次搜索的起点
        self.prepare = []
        if len(self.tabu) > self.tabulen:
            self.tabu.pop(0)

t = Tabu()
for i in range(100):
    t.step()
print(t.city)
print(t.route)
print(t.bestcost)
print(t.curroute)
