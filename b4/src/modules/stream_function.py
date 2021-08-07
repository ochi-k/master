import numpy as np
from scipy import integrate, optimize
from scipy.signal import fftconvolve


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
