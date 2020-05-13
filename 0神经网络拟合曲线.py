import numpy as np
# import npBP as BP
import krBP
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


f = lambda x: 0.1 * x**3 + 3 * x**2 + 10
x1 = np.linspace(-31, 11, 50)
x0 = np.random.randint(-30, 10, 12)
y0 = f(x0)
data = np.vstack((x0, y0))
data = data.T.reshape(12, 2)
x2 = np.random.randint(-30, 10, 3)
y2 = f(x2)

krnn = krBP.BPNN(1)

plt.ion()
fig = plt.figure()
ax = axisartist.Subplot(fig, 1, 1, 1)

for epoch in [1, 10, 100, 500, 1000]:
    np.random.shuffle(data)
    inputs = data[:, 0]
    labels = data[:, 1]
    krnn.train(inputs, labels, inputs, labels, epoch)

    yy = krnn.model.predict(x1)
    yy = yy.reshape(*x1.shape)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["x"].set_axisline_style("->", size=2.0)
    ax.axis["y"] = ax.new_floating_axis(1, -30)
    ax.axis["y"].set_axisline_style("->", size=2.0)
    ax.axis["y"].set_axis_direction('left')
    ax.annotate(s='x', xy=(12, 20), xytext=(12, 20))
    ax.annotate(s='y', xy=(-28, 400), xytext=(-28, 400))
    np.random.shuffle(data)
    inputs = data[:, 0]
    labels = data[:, 1]
    h = krnn.train(inputs, labels, x2, y2, epoch)
    yy = krnn.model.predict(x1)
    yy = yy.reshape(*x1.shape)
    ax.plot(x0, y0, "kD")
    ax.plot(x2, y2, "d")
    ax.plot(x1, f(x1), "r:")
    ax.plot(x1, yy, "--", label=f"epoch {epoch}")
    ax.legend(loc="upper center")
    plt.pause(0.1)

plt.pause(1000)
plt.ioff()
