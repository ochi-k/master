import sys
sys.path.append("./modules")

import numpy as np

import adam


if __name__ == '__main__':
    print("start program!")

    piv_data = np.loadtxt("../data/mask.csv", delimiter=",")
    result = adam.adam(piv_data, m=2580, x0=112, y0=57, U=175)

    result.to_csv('../data/result_adam_175.csv', index=False)

    print("\nprogram fin.")
