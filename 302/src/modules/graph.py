from matplotlib import pyplot as plt
import numpy as np


def quiver(data, speed=0, mode=None, out_dir="/"):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0, 150)
    plt.ylim(0, 120)

    J, I = 42, 53
    x = data[:, 0].reshape((J, I))
    y = data[:, 1].reshape((J, I))
    u = data[:, 2].reshape((J, I))
    v = data[:, 3].reshape((J, I))
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
    pp.set_label('\n|'r'$\mathbf{u}$| / $\it{U}$ [-]', fontsize=28)
    plt.clim(-10, 10)

    x = x[0::2, 0::2]
    y = y[0::2, 0::2]
    u = u[0::2, 0::2]
    v = v[0::2, 0::2]

    plt.quiver(x, y, u, v, scale_units='xy', scale=0.15)

    if mode == "show":
        plt.show()

    elif mode == "save":
        fig.savefig(out_dir + f"ave_graph_{speed}.png", dpi=300)
