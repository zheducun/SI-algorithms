import krBP
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dataset = pd.read_csv("datas_00f.csv")
dataset = np.array(dataset)
np.random.shuffle(dataset)
num_train = int(0.8*dataset.shape[0])
train, test = dataset[:num_train], dataset[num_train:]
X_train, Y_train = train[:, 1:], train[:, :1]
X_test, Y_test = test[:, 1:], test[:, :1]

nn = krBP.BPNN(2)


def f(x, y):
    x = x / 10
    y = y / 10
    result = -20*np.exp(-0.2*np.sqrt((x*x+y*y)/2)) - \
        np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2)+20+np.exp(1)
    return result


# def f(x, y):
#         x = x / 10
#         y = y / 10
#         return np.array((x**4-16*x**2+5*x)+(y**4-16*y**2+5*y)) + 160


delta = 1
X1 = np.arange(-40, 40, delta)
Y1 = np.arange(-40, 40, delta)
X, Y = np.meshgrid(X1, Y1)
Z2 = f(X, Y)


for epoch in [1, 10, 100, 1000]:
    fig = plt.figure()
    nn.train(X_train, Y_train, X_test, Y_test, epoch)
    zx = np.zeros([X.size, 2])
    Z1 = np.zeros([*X.shape])
    for i, x in enumerate(X1):
        for j, y in enumerate(Y1):
            zx[i*Y1.shape[0]+j, :] = np.array([x, y])
    zz = nn.model.predict(zx)
    count = 0

    for i, x in enumerate(X1):
        for j, y in enumerate(Y1):
            Z1[i][j] = Z1[j][i] = np.array(zz[count])
            count += 1

    """
    三维表面图形
    """
    
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z1, cmap="rainbow", linewidth=0, antialiased=False)
    ax.contour(X, Y, Z1, offset=-2, cmap='rainbow')
    ax.set_title(f"{epoch}")

    fig = plt.figure(figsize=(5, 4), dpi=200)
    #填充颜色，f即filled
    contour1 = plt.contour(X, Y, Z1, 20, linewidths=0.5, linestyles="-")
    contour2 = plt.contour(X, Y, Z2, 20, linewidths=0.5, linestyles=":")
    #等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    plt.clabel(contour1, fontsize=4, colors=('g'))
    plt.clabel(contour2, fontsize=4, colors=('r'))

plt.show()
