import numpy as np
import torch

from gpu_modules import model, stream_function


def rel_vector_norm(vel, x, y, U, params):  # rel vector norm
    # calc predicted velocity
    uf1s = model.UniformFlowAndOneSource(x=x, y=y, U=U, params=params)
    vel_pred = uf1s.vel
    
    # calc error
    if type(vel_pred).__module__ != "numpy":
        error_norm = torch.norm(vel - vel_pred)
        vel_norm = torch.norm(vel)
        error = error_norm / vel_norm
    else:
        error_norm = np.linalg.norm(vel - vel_pred, ord=2)
        vel_norm = np.linalg.norm(vel, ord=2)
        error = error_norm / vel_norm

    del uf1s

    return error


def global_vector_norm(vel, x, y, U, params):  # divided by representative velocity
    # calc predicted velocity
    uf1s = model.UniformFlowAndOneSource(x=x, y=y, U=U, params=params)
    vel_pred = uf1s.vel

    # calc error
    if type(vel_pred).__module__ != "numpy":
        error_norm = torch.norm(vel - vel_pred)
        error = error_norm / U
    else:
        error_norm = np.linalg.norm(vel - vel_pred, ord=2)
        error = error_norm / U

    del uf1s

    return error


def speed(vel, x, y, U, params):  # speed comparison
    # calc predicted velocity
    uf1s = model.UniformFlowAndOneSource(x=x, y=y, U=U, params=params)
    speed_pred = uf1s.speed

    # calc error
    if type(speed_pred).__module__ != "numpy":
        error = torch.abs(vel[:, 0] ** 2 + vel[:, 1] ** 2 - speed_pred ** 2)
    else:
        error = np.abs(vel[:, 0] ** 2 + vel[:, 1] ** 2 - speed_pred ** 2)

    del uf1s

    return error


def vector_cc(vel, x, y, U, params):  # cross correlation of vectors
    # calc predicted velocity
    uf1s = model.UniformFlowAndOneSource(x=x, y=y, U=U, params=params)
    vel_pred = uf1s.vel

    # calc error
    if type(vel_pred).__module__ != "numpy":
        cc = torch.sum(vel[:, 0] * vel_pred[:, 0] + vel[:, 1] * vel_pred[:, 1])\
             / torch.sqrt(torch.sum(vel[:, 0] ** 2 + vel[:, 1] ** 2))\
             / torch.sqrt(torch.sum(vel_pred[:, 0] ** 2 + vel_pred[:, 1] ** 2))
        error = torch.tensor(1) - torch.abs(cc)
    else:
        cc = (vel[:, 0] * vel_pred[:, 0] + vel[:, 1] * vel_pred[:, 1]).sum()\
             / np.sqrt((vel[:, 0] ** 2 + vel[:, 1] ** 2).sum())\
             / np.sqrt((vel_pred[:, 0] ** 2 + vel_pred[:, 1] ** 2).sum())
        error = 1 - np.abs(cc)

    del uf1s

    return error


# def stream_function_cc(vel, x, y, U, params, grid):  # cross correlation of stream function
#     # calc stream function
#     s_f = stream_function.stream_function(data=vel, grid=grid, option="stream_function")
#
#     # calc predicted stream function
#     m = model.UniformFlowAndOneSource(x=vel[:, 0], y=vel[:, 1], U=U, params=params)
#     s_f_pred = m.stream_function
#
#     # calc error
#     cc = (s_f - np.mean(s_f)) * (s_f_pred - np.mean(s_f_pred))\
#          / np.sqrt((s_f - np.mean(s_f))**2)\
#          / np.sqrt((s_f_pred - np.mean(s_f_pred))**2)
#
#     error = 1 - np.abs(cc)
#
#     return error
