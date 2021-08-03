import numpy as np
from matplotlib import pyplot as plt

from modules import calc


# global params
super_dir = "../data/b4/"

U = [175, 200, 225]
Q = [215, 304, 429]
D = [18, 24, 30]


def stream_function():
    for u in U:
        for q in Q:
            for d in D:
                ave_path = super_dir + f"ave/ave_u_{u}_q_{q}_d_{d}_ppm_0.csv"
                data = np.loadtxt(ave_path, delimiter=",")
                s_f, v_p = calc.stream_function(data=data, grid=(42, 53), option="both")
                data = np.hstack([data[:, :4], s_f[:, np.newaxis]])
                data = np.hstack([data, v_p[:, np.newaxis]])
                np.savetxt(super_dir + f"s_f/u_{u}_q_{q}_d_{d}_ppm_0.csv", data, delimiter=',')


if __name__ == '__main__':
    stream_function()
