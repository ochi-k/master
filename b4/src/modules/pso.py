import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import loss_function as lf


def update_x(x, v):
    return x + v


def update_v(x, v, p_i, p_g, w=0.5, r_max=1.0, c1=1, c2=1):
    new_v = np.zeros_like(v)

    # set two randoms (default: [0, 1.0])
    r1 = random.uniform(0, r_max)
    r2 = random.uniform(0, r_max)

    for i in range(len(x)):
        new_v[i] = w * float(v[i]) + c1 * r1 * (float(p_i[i]) - float(x[i])) + c2 * r2 * (float(p_g[i]) - float(x[i]))

    return new_v


def pso(vel_data, U):
    # set seeds
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

    # remove NaN
    data_drop_na = pd.DataFrame(vel_data).dropna()
    x = data_drop_na[[0]].values
    y = data_drop_na[[1]].values
    vel = data_drop_na[[2, 3]].values

    # set params
    n = 100  # particles
    dim = 3  # dimensions
    generation = 10  # max generations
    m_range = [0, 3000]  # m range
    x0_range = [-40, 40]  # x0 range
    y0_range = [-40, 40]  # y0 range
    loss_func = lf.speed  # loss function

    # initialize particle position
    xs = np.zeros((n, dim))
    vs = np.zeros_like(xs)

    for i in range(n):
        xs[i][0] = random.uniform(*m_range)   # m init
        xs[i][1] = random.uniform(*x0_range)  # x0 init
        xs[i][2] = random.uniform(*y0_range)  # y0 init

    # set vars for evaluation
    p_i = xs                                                      # best position of the i-th particle
    best_scores = [loss_func(vel, x, y, U, i) for i in tqdm(xs)]  # eval personal best
    best_particle = np.argmin(best_scores)                        # particle index in minimum evals
    p_g = p_i[best_particle]                                      # global best
    judge = 100
    tmp_p_g = None
    flag = 0

    # generations loop
    print("\n[generate]")
    for t in tqdm(range(generation)):
        for i in tqdm(range(n), desc=f"\n[Gen. {t+1} / {generation}]"):
            # update velocity
            vs[i] = update_v(xs[i], vs[i], p_i[i], p_g, r_max=0.5)

            # update position
            xs[i] = update_x(xs[i], vs[i])

            # calc personal best
            score = loss_func(vel, x, y, U, xs[i])
            if score < best_scores[i]:
                best_scores[i] = score
                p_i[i] = xs[i]

        # update global best
        if np.min(best_scores) < judge:
            flag = 0
            judge = np.min(best_scores)
            best_particle = np.argmin(best_scores)
            tmp_p_g = p_i[best_particle]

        else:
            flag += 1

        if flag > 2:
            break

        best_particle = np.argmin(best_scores)
        p_g = p_i[best_particle]

    print(f"\n{tmp_p_g}")
    print(f"{judge}")

    result = pd.DataFrame([np.append(p_g, min(best_scores))])

    return result
