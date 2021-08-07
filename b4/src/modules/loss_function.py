import numpy as np

import model
import stream_function


def rel_vector_norm(vel, U, params):  # rel vector norm
    # calc predicted velocity
    m = model.UniformFlowAndOneSource(x=vel[:, 0], y=vel[:, 1], U=U, params=params)
    vel_pred = m.vel
    
    # calc error
    error_norm = np.linalg.norm(vel[:, 2:4] - vel_pred, ord=2)
    vel_norm = np.linalg.norm(vel[:, 2:4], ord=2)
    error = error_norm / vel_norm

    return error


def global_vector_norm(vel, U, params):  # divided by representative velocity
    # calc predicted velocity
    m = model.UniformFlowAndOneSource(x=vel[:, 0], y=vel[:, 1], U=U, params=params)
    vel_pred = m.vel

    # calc error
    error_norm = np.linalg.norm(vel[:, 2:4] - vel_pred, ord=2)
    error = error_norm / U

    return error


def speed(vel, U, params):  # speed comparison
    # calc predicted velocity
    m = model.UniformFlowAndOneSource(x=vel[:, 0], y=vel[:, 1], U=U, params=params)
    speed_pred = m.speed

    # calc error
    error = np.abs(vel[:, 2] ** 2 + vel[:, 3] ** 2 - speed_pred ** 2)

    return error


def vector_cc(vel, U, params):  # cross correlation of vectors
    # calc predicted velocity
    m = model.UniformFlowAndOneSource(x=vel[:, 0], y=vel[:, 1], U=U, params=params)
    vel_pred = m.vel

    # calc error
    cc = (vel[:, 2] * vel_pred[:, 0] + vel[:, 3] * vel_pred[:, 1])\
         / np.sqrt(vel[:, 2] ** 2 + vel[:, 3] ** 2)\
         / np.sqrt(vel_pred[:, 0] ** 2 + vel_pred[:, 1] ** 2)

    error = 1 - np.abs(cc)

    return error


def stream_function_cc(vel, U, params, grid):  # cross correlation of stream function
    # calc stream function
    s_f = stream_function.stream_function(data=vel, grid=grid, option="stream_function")

    # calc predicted stream function
    m = model.UniformFlowAndOneSource(x=vel[:, 0], y=vel[:, 1], U=U, params=params)
    s_f_pred = m.stream_function

    # calc error
    cc = (s_f - np.mean(s_f)) * (s_f_pred - np.mean(s_f_pred))\
         / np.sqrt((s_f - np.mean(s_f))**2)\
         / np.sqrt((s_f_pred - np.mean(s_f_pred))**2)

    error = 1 - np.abs(cc)

    return error
