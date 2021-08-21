import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from modules import calc


def quiver(data, speed=0, mode=None, out_dir="/"):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.xlim(0, 156)
    plt.ylim(122, 0)

    wid, hei = 1280, 1024
    grid = 24
    I = wid // grid - 1 if wid % grid == 0 else wid // grid
    J = hei // grid - 1 if hei % grid == 0 else hei // grid
    x = data[:, 0].reshape((J, I))
    y = data[:, 1].reshape((J, I))
    u = data[:, 2].reshape((J, I))
    v = - data[:, 3].reshape((J, I))
    o = np.sqrt(u ** 2 + v ** 2)
    omega = calc.vorticity(data=data, grid=(I, J), option="abs").reshape((J, I))

    plt.pcolormesh(x, y, o/175, cmap='jet', alpha=1)
    pp = plt.colorbar()
    for t in pp.ax.get_yticklabels():
        t.set_fontsize(24)
    pp.set_label('\n|'r'$\omega$| $\it{d}$ / $\it{U}$ [-]', fontsize=28)
    plt.clim(0, 1.25)

    # velocity normalize
    u /= o
    v /= o

    x = x[1::3, 1::3]
    y = y[1::3, 1::3]
    u = u[1::3, 1::3]
    v = v[1::3, 1::3]

    plt.quiver(x, y, u, v, scale_units='xy', scale=0.12, width=0.007)

    if mode == "show":
        plt.show()

    elif mode == "save":
        fig.savefig(out_dir + f"mask_graph0.png", dpi=300)


def graph():
    _dir = "../../data/"
    ave_data = np.loadtxt(_dir + "ave.csv", delimiter=",")
    mask_data = np.loadtxt(_dir + "mask.csv", delimiter=",")
    quiver(mask_data, mode="save", out_dir=_dir)


def main():
    graph()


if __name__ == '__main__':
    main()