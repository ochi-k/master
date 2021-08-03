import os

import numpy as np
from matplotlib import pyplot as plt

from modules import average, calibration, calc


# global params
super_dir = "../data/"

U = [175, 200, 225]

start = 0
end = 1000


def time_ave():
    for u in U:
        piv_dir = super_dir + f"{u}_302/piv/"
        ave_dir = super_dir + f"{u}_302/ave/"
        average.time_average(in_dir=piv_dir, out_dir=ave_dir, start=start, end=end)


def cali():
    for u in U:
        ave_dir = super_dir + f"{u}_302/ave/"
        data = np.loadtxt(ave_dir + "ave.csv", delimiter=",")
        data = calibration.space_calibration(data, length=57, p0=(823, 293), p1=(816, 575))
        data = calibration.time_calibration(data, fps=u)
        np.savetxt(ave_dir + 'ave.csv', data, delimiter=',')


def divergence():
    for u in U:
        ave_dir = super_dir + f"{u}_302/ave/"
        data = np.loadtxt(ave_dir + "ave.csv", delimiter=",")
        data[:, 4] = calc.divergence(data=data, grid=(36, 36))
        np.savetxt(ave_dir + 'div.csv', data, delimiter=',')


def quiver(data, speed=0, mode=None, out_dir="/"):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0, 180)
    plt.ylim(180, 0)

    J, I = 36, 36
    x = data[:, 0].reshape((J, I))
    y = data[:, 1].reshape((J, I))
    u = data[:, 2].reshape((J, I))
    v = - data[:, 3].reshape((J, I))
    o = np.sqrt(u ** 2 + v ** 2)
    d = data[:, 4].reshape((J, I))

    # velocity normalize
    u /= o
    v /= o
    o /= speed

    plt.pcolormesh(x, y, d, cmap='jet', alpha=1)
    pp = plt.colorbar()
    for t in pp.ax.get_yticklabels():
        t.set_fontsize(24)
    # pp.set_label('\n|'r'$\mathbf{u}$| / $\it{U}$ [-]', fontsize=28)
    pp.set_label(''r'$\nabla$ $\cdot$ $\mathbf{u}$', fontsize=28)
    plt.clim(-10, 10)
    # plt.clim(0, 1.5)

    x = x[0::2, 0::2]
    y = y[0::2, 0::2]
    u = u[0::2, 0::2]
    v = v[0::2, 0::2]

    plt.quiver(x, y, u, v, scale_units='xy', scale=0.15)

    if mode == "show":
        plt.show()

    elif mode == "save":
        # fig.savefig(out_dir + f"ave_graph_{speed}.png", dpi=300)
        fig.savefig(out_dir + f"div_graph_{speed}.png", dpi=300)


def graph():
    for u in U:
        ave_dir = super_dir + f"{u}_302/ave/"
        data = np.loadtxt(ave_dir + "div.csv", delimiter=",")
        # quiver(data, speed=u, mode="save", out_dir=ave_dir)
        quiver(data, speed=u, mode="save", out_dir=super_dir + f"{u}_302/ave/")


def main():
    # time_ave()
    # cali()
    # divergence()
    graph()


if __name__ == '__main__':
    main()
