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

dt = 1

field = cp.Field(x, y, n)
X, Y = field.coordinates

X_1d = X.reshape(-1)
Y_1d = Y.reshape(-1)


def update_position(vel, t, pars):
    global x, y

    # instance
    # one source
    # source = cp.Source(m=100, z=(X, Y), z0=(-50 + vel * t, 0))
    # u, v = source.vx, source.vy

    # two source
    # source1 = cp.Source(m=50, z=(X, Y), z0=(-50 + vel * t, 20))
    # source2 = cp.Source(m=50, z=(X, Y), z0=(-50 + vel * t, -20))
    # u, v = source1.vx + source2.vx, source1.vy + source2.vy

    # triangle
    source1 = cp.Source(m=33, z=(X, Y), z0=(-50 + vel * t, 20))
    source2 = cp.Source(m=33, z=(X, Y), z0=(-50 + vel * t, -20))
    source3 = cp.Source(m=33, z=(X, Y), z0=(-85 + vel * t, 0))
    u, v = source1.vx + source2.vx + source3.vx, source1.vy + source2.vy + source3.vy

    # slope_45
    # source1 = cp.Source(m=33, z=(X, Y), z0=(-50 + vel * t, 20))
    # source2 = cp.Source(m=33, z=(X, Y), z0=(-90 + vel * t, -20))
    # source3 = cp.Source(m=33, z=(X, Y), z0=(-70 + vel * t, 0))
    # u, v = source1.vx + source2.vx + source3.vx, source1.vy + source2.vy + source3.vy

    # save coordinates and velocity
    u_1d = u.reshape(-1)
    v_1d = v.reshape(-1)
    df = pd.DataFrame(np.array([X_1d, Y_1d, u_1d, v_1d]).T).dropna()
    # df.to_csv(f'../data/move_source_linearly/vel_field/t_{t}.csv', header=False, index=False)

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
    # df2.to_csv(f'../data/move_source_linearly/particle_position/t_{t}.csv', header=False, index=False)

    return u, v, pars


def graph(vel, t, pars):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    # plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    # plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # plt.streamplot(X, Y, u, v, density=2.5, color='k', arrowstyle='-', linewidth=0.75, zorder=1)  # streamline

    # one source
    # plt.scatter(-50 + vel * t, 0, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)

    # two source
    # plt.scatter(-50 + vel * t, 20, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)
    # plt.scatter(-50 + vel * t, -20, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)

    # triangle
    plt.scatter(-50 + vel * t, 20, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)
    plt.scatter(-50 + vel * t, -20, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)
    plt.scatter(-85 + vel * t, 0, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)

    # slope_45
    # plt.scatter(-50 + vel * t, 20, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)
    # plt.scatter(-90 + vel * t, -20, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)
    # plt.scatter(-70 + vel * t, 0, s=600, c="#ffffff", linewidths=6, edgecolors='#2a68d4', zorder=10)

    for par in pars:
        plt.scatter(par.x, par.y, s=120, c="#4f4f4f", marker='D', zorder=2)

    # plt.show()
    # fig.savefig(f'../data/move_source_linearly/graph_interval_0/t_{t}.png', dpi=300)
    # fig.savefig(f'../data/move_source_linearly/graph_interval_10/t_{t}.png', dpi=300)
    # fig.savefig(f'../data/move_source_linearly/graph_interval_20/t_{t}.png', dpi=300)
    fig.savefig(f'../data/move_source_linearly/graph_triangle/t_{t}.png', dpi=300)
    # fig.savefig(f'../data/move_source_linearly/graph_slope_45/t_{t}.png', dpi=300)

    plt.clf()
    plt.close()


if __name__ == '__main__':
    # instance tracer particles
    particles = []

    for py in range(-45, 55, 5):
        for px in range(-45, 55, 5):
            p = cp.Item()
            p.x, p.y = px, py
            particles.append(p)

    for i in tqdm(range(11)):
        V_x, V_y, particles = update_position(vel=10, t=i, pars=particles)
        graph(vel=10, t=i, pars=particles)
