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

    plt.pcolormesh(x, y, o, cmap='jet', alpha=1)
    pp = plt.colorbar()
    for t in pp.ax.get_yticklabels():
        t.set_fontsize(24)
    pp.set_label('\n|'r'$\mathbf{u}$| / $\it{U}$ [-]', fontsize=28)
    plt.clim(0, 1.5)

    x = x[0::2, 0::2]
    y = y[0::2, 0::2]
    u = u[0::2, 0::2]
    v = v[0::2, 0::2]

    plt.quiver(x, y, u, v, scale_units='xy', scale=0.15)

    if mode == "show":
        plt.show()

    elif mode == "save":
        fig.savefig(out_dir + f"ave_graph_{speed}.png", dpi=300)


def m_pso(data):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.xlabel('\ngeneration', fontsize=28)
    plt.ylabel('\n$\it{m}$', fontsize=28)
    plt.xticks(data[:, 0] + 1, (data[:, 0] + 1).astype(int), fontsize=24)
    plt.yticks(fontsize=24)
    plt.scatter(data[:, 0] + 1, data[:, 1], s=100)
    plt.grid(which='both', color='black', linestyle='-')
    fig.show()
    fig.savefig("../../data/pso_m.png", dpi=300)


def z0_pso(data):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.xlabel('\n$\it{x}$', fontsize=28)
    plt.ylabel('\n$\it{y}$', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(9.5, 10.5)
    plt.ylim(-5.5, -4.5)
    plt.scatter(data[:, 2], data[:, 3], s=100, c='r')
    plt.grid(which='both', color='black', linestyle='-')
    fig.show()
    fig.savefig("../../data/pso_z0.png", dpi=300)


def error_pso(data):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.xlabel('\ngeneration', fontsize=28)
    plt.ylabel('\nError', fontsize=28)
    plt.xticks(data[:, 0]+1, (data[:, 0]+1).astype(int), fontsize=24)
    plt.yticks(fontsize=24)
    plt.scatter(data[:, 0]+1, data[:, 4], s=100, c='g')
    plt.yscale('log')
    plt.grid(which='both', color='black', linestyle='-')
    fig.show()
    fig.savefig("../../data/pso_error.png", dpi=300)


def m_adam(data):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.xlabel('\niteration', fontsize=28)
    plt.ylabel('\n$\it{m}$', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.scatter(data[:, 0] + 1, data[:, 1])
    plt.grid(which='both', color='black', linestyle='-')
    fig.show()
    fig.savefig("../../data/adam_m.png", dpi=300)


def z0_adam(data):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.xlabel('\n$\it{x}$', fontsize=28)
    plt.ylabel('\n$\it{y}$', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(9.9, 10.1)
    plt.ylim(-5.1, -4.9)
    plt.scatter(data[:, 2], data[:, 3], c='r')
    plt.grid(which='both', color='black', linestyle='-')
    fig.show()
    fig.savefig("../../data/adam_z0.png", dpi=300)


def error_adam(data):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.xlabel('\niteration', fontsize=28)
    plt.ylabel('\nError', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.scatter(data[:, 0]+1, data[:, 4], c='g')
    plt.yscale('log')
    plt.grid(which='both', color='black', linestyle='-')
    fig.show()
    fig.savefig("../../data/adam_error.png", dpi=300)


if __name__ == '__main__':
    pso_csv = np.loadtxt("../../data/pso_result.csv", delimiter=',', skiprows=1)
    adam_csv = np.loadtxt("../../data/adam_result.csv", delimiter=',', skiprows=1)

    m_pso(pso_csv)
    z0_pso(pso_csv)
    error_pso(pso_csv)

    m_adam(adam_csv)
    z0_adam(adam_csv)
    error_adam(adam_csv)
