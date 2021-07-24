import functools

import numpy as np
import pandas as pd
import torch


"""
calc velocity property
"""


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

    for j in range(1, grid[1] - 1):
        for i in range(1, grid[0] - 1):
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

    for j in range(1, grid[1] - 1):
        for i in range(1, grid[0] - 1):
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


"""
Calc fitting
"""


def loss_func(vel, x, y, U, m, x0, y0):
    # calc velocity
    _u = m * (x - x0) / ((x - x0) ** 2 + (y - y0) ** 2) \
         + m * (x + x0) / ((x + x0) ** 2 + (y - y0) ** 2)
    _v = - U + m * (y - y0) / ((x - x0) ** 2 + (y - y0) ** 2) \
         + m * (y - y0) / ((x + x0) ** 2 + (y - y0) ** 2)

    # concatenate velocity
    _vel_pred = torch.cat([_u, _v], axis=1)

    # calc error
    error = (vel - _vel_pred).norm() / vel.norm()

    return error


def test_func(vel, x, y, U, m, x0, y0):
    # calc velocity
    _u = m * (x - x0) / ((x - x0) ** 2 + (y - y0) ** 2)
    _v = - U + m * (y - y0) / ((x - x0) ** 2 + (y - y0) ** 2)
    _u_inv = ((x - x0) ** 2 + (y - y0) ** 2) / m / (x - x0)
    _v_inv = ((x - x0) ** 2 + (y - y0) ** 2) / (m * (y - y0) - U * ((x - x0) ** 2 + (y - y0) ** 2))

    # concatenate velocity
    _vel_pred = torch.cat([_u, _v], axis=1)
    _vel_inv_pred = torch.cat([_u_inv, _v_inv], axis=1)
    a = np.count_nonzero(np.isnan(_vel_pred.detach().numpy().copy()))

    # calc error
    error = (vel - _vel_pred).norm() / vel.norm()
    # error = (vel - _vel_inv_pred).norm() / vel.norm()

    d = vel.norm()
    dd = (vel - _vel_inv_pred).detach().numpy().copy()
    ddd = (vel - _vel_inv_pred).norm()
    dddd = np.count_nonzero(np.isnan(dd))

    return error


def my_fitting(data, U):
    """
    Optimize (= Minimize) loss function by params: [m, x0, y0]
    Loss function: Vector norm2 error (rel)

    :param: data: nd-array[x, y, u, v], shape -> (~, 4)

    :return: optimized params: [m, x0, y0]
    """

    # remove NaN
    data_drop_na = pd.DataFrame(data).dropna()
    x = torch.tensor(data_drop_na[[0]].values)
    y = torch.tensor(data_drop_na[[1]].values)
    vel = torch.tensor(data_drop_na[[2, 3]].values)
    vel_inv = torch.tensor(data_drop_na[[4, 5]].values)

    # set params and optimizer
    m = torch.tensor(2678.81709086, requires_grad=True)
    x0 = torch.tensor(11.03601662, requires_grad=True)
    y0 = torch.tensor(-5.18455088, requires_grad=True)
    params = [m, x0, y0]

    lr = 1  # learning rate
    optimizer = torch.optim.Adam(params, lr)

    # optimization
    outputs = None
    m_list = []
    x0_list = []
    y0_list = []
    f_list = []
    i_list = []

    for i in range(1000):
        print(f"Step: {i}")
        optimizer.zero_grad()
        outputs = test_func(vel, x, y, U, *params)
        outputs.backward()
        optimizer.step()
        m_list.append(m.item())
        x0_list.append(x0.item())
        y0_list.append(y0.item())
        f_list.append(outputs.item())
        i_list.append(i)

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(15, 10))
    plt.scatter(i_list, f_list)
    plt.ylim(0, 1)
    fig.show()

    return outputs
