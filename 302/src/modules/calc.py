import numpy as np
import pandas as pd
import torch


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


def norm_error(v_true, v_pred, rel=True):
    # remove NaN
    vel = pd.DataFrame(np.hstack([v_true, v_pred])).dropna
    v_t = vel[[0, 1]].value
    v_p = vel[[2, 3]].value

    error = np.linalg.norm(v_t - v_p, ord=2)

    if rel:
        error /= np.linalg.norm(v_t, ord=2)

    return error


def my_fitting(data, U):
    """
    Optimize (= Minimize) loss function by params: [m, x0, y0]
    Loss function: Vector norm2 error (rel)

    :param: data: nd-array[x, y, u, v], shape -> (~, 4)

    :return: optimized params: [m, x0, y0]
    """

    def _loss_func(_vel, _m, _x0, _y0):
        # calc velocity
        _u = _m * (x - _x0) / ((x - _x0) ** 2 - (y - _y0) ** 2) \
             + _m * (x + _x0) / ((x + _x0) ** 2 - (y - _y0) ** 2)
        _v = - U + _m * (y - _y0) / ((x - _x0) ** 2 - (y - _y0) ** 2) \
             + _m * (y - _y0) / ((x + _x0) ** 2 - (y - _y0) ** 2)

        # transform to numpy array
        _vel_numpy = _vel.detach().numpy().copy()
        _u_numpy = _u.detach().numpy().copy()
        _v_numpy = _v.detach().numpy().copy()

        error = np.linalg.norm(_vel_numpy - np.hstack([_u_numpy, _v_numpy]), ord=2) / np.linalg.norm(_vel_numpy, ord=2)

        return torch.tensor(error, requires_grad=True)

    # remove NaN
    data_drop_na = pd.DataFrame(data).dropna()
    x = torch.tensor(data_drop_na[[0]].values)
    y = torch.tensor(data_drop_na[[1]].values)
    vel = torch.tensor(data_drop_na[[2, 3]].values)

    # set params and optimizer
    m = torch.tensor(2000.0, requires_grad=True)
    x0 = torch.tensor(10.0, requires_grad=True)
    y0 = torch.tensor(-5.0, requires_grad=True)
    params = [m, x0, y0]

    lr = 0.01  # learning rate
    optimizer = torch.optim.Adam(params, lr)

    # optimization
    m_list = []
    x0_list = []
    y0_list = []
    f_list = []

    for i in range(10000):
        print(f"Step: {i}")
        optimizer.zero_grad()
        outputs = _loss_func(vel, *params)
        outputs.backward()
        optimizer.step()
        m_list.append(m.item())
        x0_list.append(x0.item())
        y0_list.append(y0.item())
        f_list.append(outputs.item())

    return outputs
