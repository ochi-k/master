import sys

import numpy as np


EPS = sys.float_info.epsilon  # machine epsilon


class Field(object):

    def __init__(self, x_range, y_range, div_num):
        self._x = np.linspace(x_range[0], x_range[1], div_num[0]+1)
        self._y = np.linspace(y_range[0], y_range[1], div_num[1]+1)
        self._coordinates = np.zeros((2, div_num[1]+1, div_num[0]+1))
        self.prevent_zero_div()
        self.mesh()

    def prevent_zero_div(self):
        self._x += EPS
        self._y += EPS

    def mesh(self):
        self._coordinates[0], self._coordinates[1] = np.meshgrid(self._x, self._y)
        return

    @property
    def coordinates(self):
        return self._coordinates


class Item(object):
    """item object
    :ivar _Vx: x component of velocity
    :ivar _Vy: y component of velocity
    """

    def __init__(self):
        self._Vx = 0.0
        self._Vy = 0.0

    @property
    def v(self):
        """getter/setter of velocity
        :return array[V_x, V_y]: Optional(Tuple[float, float])
        """

        return self._Vx, self._Vy

    @property
    def vx(self):  # getter/setter of x component
        return self._Vx

    @property
    def vy(self):  # getter/setter of y component
        return self._Vy


class UniformFlow(Item):
    """uniform flow object
    :ivar _Vx: x component of velocity
    :ivar _Vy: y component of velocity
    :ivar _U: flow speed
    :ivar _alpha: angle
    """

    def __init__(self, U=0, alpha=0):
        """
        :param U: flow speed [m/s]
        :param alpha: angle [rad]
        """

        super().__init__()
        self._U = U
        self._alpha = alpha
        self.calc()

    def calc(self):  # velocity calculation
        self._Vx = self._U * np.cos(self._alpha)
        self._Vy = self._U * np.sin(self._alpha)


class Source(Item):
    """source object
    :ivar _Vx: x component of velocity
    :ivar _Vy: y component of velocity
    :ivar _m: source strength
    :ivar _x: x coordinate
    :ivar _y: y coordinate
    :ivar _x0: x coordinate of source
    :ivar _y0: y coordinate of source
    """

    def __init__(self, m=0, z=(0, 0), z0=(0, 0)):
        """
        :param m: source strength [m^2/s]
        :param z: coordinates [m]
        :param z0: source coordinates [m]
        """

        super().__init__()
        self._m = m
        self._x, self._y = z
        self._x0, self._y0 = z0
        self.calc()

    def calc(self):  # velocity calculation
        self._Vx = self._m * (self._x - self._x0) / ((self._x - self._x0)**2 + (self._y - self._y0)**2)
        self._Vy = self._m * (self._y - self._y0) / ((self._x - self._x0)**2 + (self._y - self._y0)**2)


class VortexLine(Item):
    """vortex line object
    :ivar _Vx: x component of velocity
    :ivar _Vy: y component of velocity
    :ivar _k: vortex strength
    :ivar _x: x coordinate
    :ivar _y: y coordinate
    :ivar _x0: x coordinate of vortex
    :ivar _y0: y coordinate of vortex
    """

    def __init__(self, k=0, z=(0, 0), z0=(0, 0)):
        """
        :param k: vortex strength [m^2/s]
        :param z: coordinates [m]
        :param z0: vortex coordinates [m]
        """

        super().__init__()
        self._k = k
        self._x, self._y = z
        self._x0, self._y0 = z0
        self.calc()

    def calc(self):  # velocity calculation
        self._Vx = self._k * (self._y - self._y0) / ((self._x - self._x0)**2 + (self._y - self._y0)**2)
        self._Vy = - self._k * (self._x - self._x0) / ((self._x - self._x0) ** 2 + (self._y - self._y0) ** 2)


if __name__ == '__main__':
    x_min, x_max = -50, 50
    y_min, y_max = -50, 50
    x = np.array([x_min, x_max])
    y = np.array([y_min, y_max])
    n = np.array([100, 100])  # grid_num(#x, #y)

    # instance
    field = Field(x, y, n)
    X, Y = field.coordinates

    uni_flow = UniformFlow(U=200, alpha=np.deg2rad(-90))
    source = Source(m=2000, z=(X, Y), z0=(10, -5))

    # velocity: v_x = u(y, x), v_y = v(y, x)
    u = uni_flow.vx + source.vx
    v = uni_flow.vy + source.vy

    # save coordinates and velocity
    import pandas as pd
    X_1d = X.reshape(-1)
    Y_1d = Y.reshape(-1)
    u_1d = u.reshape(-1)
    v_1d = v.reshape(-1)
    df = pd.DataFrame(np.array([X_1d, Y_1d, u_1d, v_1d, 1/u_1d, 1/v_1d]).T).dropna()
    df.to_csv('../../data/sample_cp.csv', header=False, index=False)

    # graph
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.streamplot(X, Y, u, v, density=2, color='k', arrowstyle='-', linewidth=1)  # streamline

    # plt.show()
    fig.savefig("../../data/cp_img.png", dpi=300)
