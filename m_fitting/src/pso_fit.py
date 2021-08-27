import sys
sys.path.append("./modules")

import numpy as np
import pandas as pd

import pso_gpu as pso


if __name__ == '__main__':
    print("start program!")

    piv_data = np.loadtxt("../data/mask.csv", delimiter=",")
    result = pso.pso(piv_data, U=175)

    df = pd.DataFrame([["m", "x0", "y0", "error"]])
    df = pd.concat([df, result])

    df.to_csv('../data/result_pso_175.csv', index=False)

    print("\nprogram fin.")
