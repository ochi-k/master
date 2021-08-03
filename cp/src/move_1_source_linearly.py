import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from modules import c_potential as cp


def draw(vel, t, pars):
    x_min, x_max = -50, 50
    y_min, y_max = -50, 50

    x = np.array([x_min, x_max])
    y = np.array([y_min, y_max])
    n = np.array([100, 100])  # grid_num(#x, #y)

    dt = 1

    # instance
    field = cp.Field(x, y, n)
    X, Y = field.coordinates

    source = cp.Source(m=10, z=(X, Y), z0=(-50 + vel * t, 0))

    # velocity
    u = source.vx
    v = source.vy

    # save coordinates and velocity
    X_1d = X.reshape(-1)
    Y_1d = Y.reshape(-1)
    u_1d = u.reshape(-1)
    v_1d = v.reshape(-1)
    df = pd.DataFrame(np.array([X_1d, Y_1d, u_1d, v_1d]).T).dropna()
    df.to_csv(f'../data/move_1_source_linearly/vel_field/t_{t}.csv', header=False, index=False)

    # update particle position
    x_list = np.linspace(x[0], x[1], n[0]+1)
    y_list = np.linspace(y[0], y[1], n[0]+1)

    px_list = []
    py_list = []

    for par in pars:
        # get nearest value
        px_idx = np.abs(x_list - par.x).argmin()
        py_idx = np.abs(y_list - par.y).argmin()

        # update
        par.x += u[py_idx, px_idx] * dt
        par.y += v[py_idx, px_idx] * dt

        px_list.append(par.x)
        py_list.append(par.y)

    # graph
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.streamplot(X, Y, u, v, density=2, color='k', arrowstyle='-', linewidth=1)  # streamline

    plt.show()

    plt.clf()
    plt.close()

    return


if __name__ == '__main__':
    # instance tracer particles
    particles = []

    for py in range(-45, 50, 5):
        for px in range(-45, 50, 5):
            p = cp.Item
            p.x, p.y = px, py
            particles.append(p)

    for i in range(1000):
        draw(vel=0.1, t=i, pars=particles)
