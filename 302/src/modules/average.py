import math

import numpy as np


def time_average(in_dir, out_dir, start, end):
    f_n = "result_"
    t_ave = None
    check = None
    
    for t in range(start, end):
        f_n_tmp = f_n + f"{t}_{t + 1}.csv"
        df = np.loadtxt(in_dir + f_n_tmp, delimiter=",")
        
        if t == start:
            t_ave = np.zeros_like(df)
            check = np.zeros_like(df) + end - start + 1e-16

        for j in range(df.shape[0]):
            for i in range(df.shape[1]):
                if math.isnan(df[j][i]):
                    check[j][i] -= 1

                else:
                    t_ave[j][i] += df[j][i]

    t_ave /= check

    np.savetxt(out_dir + "ave.csv", t_ave, delimiter=',')