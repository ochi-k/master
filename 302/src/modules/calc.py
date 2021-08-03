import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.signal import fftconvolve
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


def stream_function(data, grid, option=None):
    """Solve stream function and velocity potential from given u and v.

    u and v are given in an even grid (n x m).

    stream function (psi) and velocity potential (chi) are defined on a dual
    grid ((n+1) x (m+1)), where psi and chi are defined on the 4 corners
    of u and v.

    Define:

        u = u_chi + u_psi
        v = v_chi + v_psi

        u_psi = -dpsi/dy
        v_psi = dpsi/dx
        u_chi = dchi/dx
        v_chi = dchi/dy


    Define 2 2x2 kernels:

        k_x = |-0.5 0.5|
              |-0.5 0.5| / dx

        k_y = |-0.5 -0.5|
              |0.5   0.5| / dy

    Then u_chi = chi \bigotimes k_x
    where \bigotimes is cross-correlation,

    Similarly:

        v_chi = chi \bigotimes k_y
        u_psi = psi \bigotimes -k_y
        v_psi = psi \bigotimes k_x

    Define cost function J = (uhat - u)**2 + (vhat - v)**2

    Gradients of chi and psi:

        dJ/dchi = (uhat - u) du_chi/dchi + (vhat - v) dv_chi/dchi
        dJ/dpsi = (uhat - u) du_psi/dpsi + (vhat - v) dv_psi/dpsi

        du_chi/dchi = (uhat - u) \bigotimes Rot180(k_x) = (uhat - u) \bigotimes -k_x
        dv_chi/dchi = (vhat - v) \bigotimes Rot180(k_y) = (vhat - v) \bigotimes -k_y
        du_psi/dpsi = (uhat - u) \bigotimes k_x
        dv_psi/dpsi = (vhat - v) \bigotimes Rot180(k_x) = (vhat - v) \bigotimes -k_x

    Add optional regularization term:

        J = (uhat - u)**2 + (vhat - v)**2 + lambda*(chi**2 + psi**2)

    Reference:
        https://stackoverflow.com/questions/49557329/compute-stream-function-from-x-and-y-velocities-by-integration-in-python#
    """

    def uRecon(_s_f, _v_p, _kernel_x, _kernel_y):
        u_chi = fftconvolve(_v_p, -_kernel_x, mode='valid')
        u_psi = fftconvolve(_s_f, _kernel_y, mode='valid')

        return u_psi + u_chi

    def vRecon(_s_f, _v_p, _kernel_x, _kernel_y):
        v_chi = fftconvolve(_v_p, -_kernel_y, mode='valid')
        v_psi = fftconvolve(_s_f, -_kernel_x, mode='valid')

        return v_psi + v_chi

    def costFunc(_params, _u, _v, _kernel_x, _kernel_y, _pad_shape, _lam):
        pp = _params.reshape(_pad_shape)
        _s_f = pp[0]
        _v_p = pp[1]
        _u_hat = uRecon(_s_f, _v_p, _kernel_x, _kernel_y)
        _v_hat = vRecon(_s_f, _v_p, _kernel_x, _kernel_y)
        j = (_u_hat - _u) ** 2 + (_v_hat - _v) ** 2
        j = j.mean()
        j += _lam * np.mean(_params ** 2)

        return j

    def jac(_params, _u, _v, _kernel_x, _kernel_y, _pad_shape, _lam):
        pp = _params.reshape(_pad_shape)
        _s_f = pp[0]
        _v_p = pp[1]
        _u_hat = uRecon(_s_f, _v_p, _kernel_x, _kernel_y)
        _v_hat = vRecon(_s_f, _v_p, _kernel_x, _kernel_y)

        du = _u_hat - _u
        dv = _v_hat - _v

        dv_p_u = fftconvolve(du, _kernel_x, mode='full')
        dv_p_v = fftconvolve(dv, _kernel_y, mode='full')

        ds_f_u = fftconvolve(du, -_kernel_y, mode='full')
        ds_f_v = fftconvolve(dv, _kernel_x, mode='full')

        ds_f = ds_f_u + ds_f_v
        dv_p = dv_p_u + dv_p_v

        re = np.vstack([ds_f[None, :, :], dv_p[None, :, :]])
        re = re.reshape(_params.shape)
        re = re + _lam * _params / _u.size

        return re

    # import coordinates as grid
    X = data[:, 0].reshape(*grid[::-1])
    Y = data[:, 1].reshape(*grid[::-1])
    dx = data[0, 0]
    dy = data[0, 1]

    # import velocity as grid
    u = data[:, 2].reshape(*grid[::-1])
    v = data[:, 3].reshape(*grid[::-1])

    # create convolution kernel
    kernel_x = np.array([[-0.5, 0.5], [-0.5, 0.5]]) / dx
    kernel_y = np.array([[-0.5, -0.5], [0.5, 0.5]]) / dy

    # integrate to make an initial guess
    int_x = integrate.cumtrapz(v, X, axis=1, initial=0)[0]
    int_y = integrate.cumtrapz(u, Y, axis=0, initial=0)
    psi1 = int_x - int_y

    int_x = integrate.cumtrapz(v, X, axis=1, initial=0)
    int_y = integrate.cumtrapz(u, Y, axis=0, initial=0)[:, 0][:, None]
    psi2 = int_x - int_y

    psi = 0.5 * (psi1 + psi2)

    int_x = integrate.cumtrapz(u, X, axis=1, initial=0)[0]
    int_y = integrate.cumtrapz(v, Y, axis=0, initial=0)
    chi1 = int_x + int_y

    int_x = integrate.cumtrapz(u, X, axis=1, initial=0)
    int_y = integrate.cumtrapz(v, Y, axis=0, initial=0)[:, 0][:, None]
    chi2 = int_x + int_y

    chi = 0.5 * (chi1 + chi2)

    # pad to add 1 row/column
    s_f = np.pad(psi, (1, 0), 'edge')
    v_p = np.pad(chi, (1, 0), 'edge')
    params = np.vstack([s_f[None, :, :], v_p[None, :, :]])

    # optimize
    pad_shape = params.shape
    lam = 0.001  # regularization parameter

    opt = optimize.minimize(costFunc, params,
                            args=(u, v, kernel_x, kernel_y, pad_shape, lam),
                            method='Newton-CG',
                            jac=jac)

    params = opt.x.reshape(pad_shape)

    # for debug
    s_f = params[0]
    v_p = params[1]
    u_hat = uRecon(s_f, v_p, kernel_x, kernel_y)
    v_hat = vRecon(s_f, v_p, kernel_x, kernel_y)

    if option == "stream_function":
        return s_f[:-1, :-1].flatten()

    elif option == "velocity_potential":
        return v_p[:-1, :-1].flatten()

    elif option == "both":
        return s_f[:-1, :-1].flatten(), v_p[:-1, :-1].flatten()

    return params


"""
calc fitting
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
    m = torch.tensor(2020.7799, requires_grad=True)
    x0 = torch.tensor(10.0186, requires_grad=True)
    y0 = torch.tensor(-5.0152, requires_grad=True)
    params = [m, x0, y0]

    lr = 0.01  # learning rate
    optimizer = torch.optim.Adam(params, lr)

    # optimization
    outputs = None
    m_list = []
    x0_list = []
    y0_list = []
    f_list = []
    i_list = []

    for i in range(100000):
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

        if outputs < 1e-4:
            break

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(15, 10))
    plt.scatter(i_list, f_list)
    plt.yscale('log')
    plt.ylim(0, 1)
    fig.show()

    return outputs
