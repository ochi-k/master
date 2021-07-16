import math


def space_calibration(data, length, p0, p1):
    return data * length / math.hypot(p0[0] - p1[0], p0[1] - p1[1])

def time_calibration(data, fps):
    data[:, 2:4] *= fps
    return data
