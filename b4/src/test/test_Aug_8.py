import sys
sys.path.append("../gpu_modules")

import numpy as np
import pandas as pd

import pso_gpu as pso


sys.path.append("../gpu_modules")


if __name__ == '__main__':
    print("start program!")

    df = pd.DataFrame([["m", "x0", "y0", "error"]])

    sample_data = np.loadtxt("../../data/sample_cp.csv", delimiter=",")
    result = pso.pso(sample_data, U=200)
    df = pd.concat([df, result])

    df.to_csv('../../data/result_speed.csv', index=False)

    print("\ntest Aug 8 fin.")
