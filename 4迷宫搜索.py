import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


with hp.File("data.h5", "r") as hf:
    points = hf["points"][:]
    length = hf["length"][:]


def in_box(x, y):
    for p, l in zip(points, length):
        if abs(x-p[0]) < l[0] or abs(y-p[1]) < l[1]:
            return 1
    return 0


# def nextPonit(x0, y0):
#     # 这是非常贪婪的走法，遇到障碍物容易走不下去
#     xd, yd = 100, 100
#     R0 = 1
#     while True:
#         rand1 = np.random.random()
#         rand2 = np.random.random()
#         alpha = rand1 * np.pi - np.pi / 2
#         if rand2 < np.cos(alpha):
#             break
#     alpha0 = np.arctan((yd-y0)/(xd-x0))
#     x1 = x0 + R0 * np.cos(alpha0-alpha)
#     y1 = y0 + R0 * np.sin(alpha0-alpha)
#     if x1 > 100:x1=100
#     if y1 > 100:y1=100
#     return x1, y1


def nextPonit(x0, y0):
    # 这是非常贪婪的走法，遇到障碍物容易走不下去
    xd, yd = 100, 100
    x1, y1 = x0, y0
    for rank in range(2):
        if np.random.random() < 0.5:
            x1 += np.random.choice([-1, 1, 1, 1, 1])
        else:
            y1 += np.random.choice([-1, 1, 1, 1, 1])
    if x1 > 100:
        x1 = 100
    if y1 > 100:
        y1 = 100
    if x1 < 0:x1=0
    if y1 < 0:y1=0
    return x1, y1

plt.ion()
for i in range(10):
    x, y = 0, 0
    fig = plt.figure()
    while x != 100 or y != 100:
        ax = axisartist.Subplot(fig, 1, 1, 1)
        fig.add_axes(ax)
        ax.axis[:].set_visible(False)
        ax.axis["x"] = ax.new_floating_axis(0, 0)
        ax.axis["x"].set_axisline_style("->", size=2.0)
        ax.axis["y"] = ax.new_floating_axis(1, 0)
        ax.axis["y"].set_axisline_style("->", size=2.0)
        ax.axis["y"].set_axis_direction('left')
        ax.annotate(s='向东/km', xy=(1, 0), xytext=(100, 1))
        ax.annotate(s='向北/km', xy=(0, 1), xytext=(0.5, 100))
        for rank in range(points.shape[0]):
            a = [points[rank][0]-length[rank][0], points[rank][0]+length[rank][0]]
            b1 = [points[rank][1]+length[rank][1], points[rank][1]+length[rank][1]]
            b2 = [points[rank][1]-length[rank][1], points[rank][1]-length[rank][1]]
            plt.plot(a, b1, "k")
            plt.plot(a, b2, "k")
            plt.plot([points[rank][0]-length[rank][0], points[rank][0]-length[rank][0]],
                    [points[rank][1]-length[rank][1], points[rank][1]+length[rank][1]], "k")
            plt.plot([points[rank][0]+length[rank][0], points[rank][0]+length[rank][0]],
                    [points[rank][1]-length[rank][1], points[rank][1]+length[rank][1]], "k")
            plt.fill_between(a, b1, b2, b1 > b2, color='k')
            plt.text(*points[rank], s=f"{rank}", c="white", ha="center", va="center")
        plt.scatter([0, 100], [0, 100], c="r", s=100, marker="*")
        # x1, y1 = nextPonit(x, y)
        # print(x, y)
        count = 0
        while True:
            count += 1
            x1, y1 = nextPonit(x, y)
            if not in_box(x1, y1) or count > 10:
                break
        plt.plot(x1, y1, "x")
        plt.pause(0.01)
        # plt.clf()
        x, y = x1, y1
    # plt.plot([x, x1], [y, y1], lw=1)
# plt.show()
