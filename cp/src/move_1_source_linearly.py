import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

from modules import c_potential as cp


# set global variables
x_min, x_max = -50, 50
y_min, y_max = -50, 50

x = np.array([x_min, x_max])
y = np.array([y_min, y_max])
n = np.array([100, 100])  # grid_num(#x, #y)

dt = 0.01

field = cp.Field(x, y, n)
X, Y = field.coordinates

X_1d = X.reshape(-1)
Y_1d = Y.reshape(-1)


def update_position(vel, t, pars):
    global x, y

    # instance
    source = cp.Source(m=100, z=(X, Y), z0=(-50 + vel * t, 0))

    # velocity
    u = source.vx
    v = source.vy

    # save coordinates and velocity
    u_1d = u.reshape(-1)
    v_1d = v.reshape(-1)
    df = pd.DataFrame(np.array([X_1d, Y_1d, u_1d, v_1d]).T).dropna()
    df.to_csv(f'../data/move_1_source_linearly/vel_field/t_{t}.csv', header=False, index=False)

    # update particle position
    x_list = np.linspace(x[0], x[1], n[0] + 1)
    y_list = np.linspace(y[0], y[1], n[0] + 1)

    px_list = []  # px storage list
    py_list = []  # py storage list

    for par in pars:
        # get nearest value
        px_idx = np.abs(x_list - par.x).argmin()
        py_idx = np.abs(y_list - par.y).argmin()

        # update
        par.x += u[py_idx, px_idx] * dt
        par.y += v[py_idx, px_idx] * dt

        px_list.append(par.x)
        py_list.append(par.y)

    # save particle positions
    df2 = pd.DataFrame(np.array([px_list, py_list]).T).dropna()
    df2.to_csv(f'../data/move_1_source_linearly/particle_position/t_{t}.csv', header=False, index=False)

    return u, v, pars


def graph(u, v, pars, t):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.streamplot(X, Y, u, v, density=2.5, color='k', arrowstyle='-', linewidth=0.75, zorder=1)  # streamline

    for par in pars:
        plt.scatter(par.x, par.y, s=120, zorder=2)

    # plt.show()
    fig.savefig(f'../data/move_1_source_linearly/graph/t_{t}.png', dpi=300)

    plt.clf()
    plt.close()


if __name__ == '__main__':
    # instance tracer particles
    particles = []

    for py in range(-45, 55, 5):
        for px in range(-45, 55, 5):
            p = cp.Item()
            p.x, p.y = px + 0.5, py + 0.5
            particles.append(p)

    for i in tqdm(range(1000)):
        V_x, V_y, particles = update_position(vel=0.1, t=i, pars=particles)
        graph(V_x, V_y, particles, i)
