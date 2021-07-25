import numpy as np
import pandas as pd

from modules import pso_gpu as pso
from modules import graph


if __name__ == '__main__':
    print("start program!")

    df = pd.DataFrame([["m", "x0", "y0", "error"]])

    U = [175, 200, 225, 250]
    Q = [215, 304, 429]
    D = [18, 24, 30]

    for u in U:
        for q in Q:
            for d in D:
                data = np.loadtxt(f"../data/ave/ave_u_{u}_q_{q}_d_{d}_ppm_0.csv", delimiter=",")
                graph.quiver(data, u, "show")
                result = pso.pso(data, U=u)
                df = pd.concat([df, result])

    # save result
    df.to_csv('../data/new_result.csv')

    print("\nPSO for gpu fin.")
