import numpy as np
import torch

import stream_function


class Flow(object):

    def __init__(self, x, y, grid=(0, 0)):
        self.x = x
        self.y = y

        self.grid = grid

        self.u = None
        self.v = None

    @property
    def vel(self):
        if type(self.u).__module__ != "numpy":
            return torch.cat([self.u, self.v], dim=1)
        return np.vstack([self.u, self.v]).T

    @property
    def speed(self):
        if type(self.u).__module__ != "numpy":
            return torch.sqrt(self.u ** 2 + self.v ** 2)
        return np.sqrt(self.u ** 2 + self.v ** 2)

    @property
    def stream_function(self):
        data_xy = np.vstack([self.x, self.y])
        data_uv = np.vstack([self.u, self.v])
        data = np.vstack([data_xy, data_uv]).T

        return stream_function.stream_function(data=data, grid=self.grid, option="stream_function")


class UniformFlowAndOneSource(Flow):

    def __init__(self, x, y, U, alpha, params):
        super().__init__(x=x, y=y)
        self.U = U
        self.alpha = alpha
        self.params = params
        self.calc_vel()

    def calc_vel(self):
        m, x0, y0 = self.params

        self.u = self.U * np.cos(self.alpha) + m * (self.x - x0) / ((self.x - x0) ** 2 + (self.y - y0) ** 2)
        self.v = self.U * np.sin(self.alpha) + m * (self.y - y0) / ((self.x - x0) ** 2 + (self.y - y0) ** 2)
