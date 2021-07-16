import numpy as np


def d_dx2(a, dx):
    return (a[2][2] - a[2][0] + 2 * (a[1][2] - a[1][0]) + a[0][2] - a[0][0]) / 8 / dx


def d_dy2(a, dy):
    return (a[2][2] - a[0][2] + 2 * (a[2][1] - a[0][1]) + a[2][0] - a[0][0]) / 8 / dy


def divergence(data, grid):
    dx = data[0][0]
    dy = data[0][1]

    u = data[:, 2]
    v = data[:, 3]

    u = u.reshape(*grid[::-1])
    v = v.reshape(*grid[::-1])

    du_dx = np.full_like(u, np.nan)
    dv_dy = np.full_like(v, np.nan)

    for j in range(1, grid[1]-1):
        for i in range(1, grid[0]-1):
            du_dx[j][i] = d_dx2(u[j - 1:j + 2, i - 1:i + 2], dx)
            dv_dy[j][i] = d_dy2(v[j - 1:j + 2, i - 1:i + 2], dy)

    return (du_dx + dv_dy).flatten()


def vorticity(data, grid, option=None):
    dx = data[0][0]
    dy = data[0][1]

    u = data[:, 2]
    v = data[:, 3]

    u = u.reshape(*grid[::-1])
    v = v.reshape(*grid[::-1])

    du_dy = np.full_like(u, np.nan)
    dv_dx = np.full_like(v, np.nan)

    for j in range(1, grid[1]-1):
        for i in range(1, grid[0]-1):
            du_dy[j][i] = d_dy2(u[j - 1:j + 2, i - 1:i + 2], dy)
            dv_dx[j][i] = d_dx2(v[j - 1:j + 2, i - 1:i + 2], dx)

    if option == 'abs':
        return np.abs(dv_dx - du_dy).flatten()

    return (dv_dx - du_dy).flatten()


def shear(data, grid, option=None):
    dx = data[0][0]
    dy = data[0][1]

    u = data[:, 2]
    v = data[:, 3]

    u = u.reshape(*grid[::-1])
    v = v.reshape(*grid[::-1])

    du_dy = np.full_like(u, np.nan)
    dv_dx = np.full_like(v, np.nan)

    for j in range(1, grid[1] - 1):
        for i in range(1, grid[0] - 1):
            du_dy[j][i] = d_dy2(u[j - 1:j + 2, i - 1:i + 2], dy)
            dv_dx[j][i] = d_dx2(v[j - 1:j + 2, i - 1:i + 2], dx)

    if option == 'abs':
        return np.abs(dv_dx + du_dy).flatten()

    return (dv_dx + du_dy).flatten()