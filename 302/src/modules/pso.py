import random

import numpy as np
from tqdm import tqdm


def eval_func(vel, U, params):  # evaluation function
    x = vel[:, 0]
    y = vel[:, 1]
    m, x0, y0 = params

    # calc velocity
    u = m * (x - x0) / ((x - x0) ** 2 + (y - y0) ** 2)
    v = - U + m * (y - y0) / ((x - x0) ** 2 + (y - y0) ** 2)
    vel_pred = np.vstack([u, v]).T
    debug = np.count_nonzero(np.isnan(vel_pred))

    # calc error
    error_norm = np.linalg.norm(vel[:, 2:4] - vel_pred, ord=2)
    vel_norm = np.linalg.norm(vel[:, 2:4], ord=2)
    error = error_norm / vel_norm

    return error


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


def main(vel_data, U):
    # set seeds
    random.seed(0)
    np.random.seed(0)

    # set params
    n = 1000000           # particles
    dim = 3               # dimensions
    generation = 1000     # max generations
    m_range = [0, 3000]   # m range
    x0_range = [-40, 40]  # x0 range
    y0_range = [-40, 40]  # y0 range

    # initialize particle position
    xs = np.zeros((n, dim))
    vs = np.zeros_like(xs)

    for i in range(n):
        xs[i][0] = random.uniform(*m_range)   # m init
        xs[i][1] = random.uniform(*x0_range)  # x0 init
        xs[i][2] = random.uniform(*y0_range)  # y0 init

    # set vals for evaluation
    p_i = xs                                               # best position of the i-th particle
    best_scores = [eval_func(vel_data, U, x) for x in xs]  # eval personal best
    best_particle = np.argmin(best_scores)                 # particle index in minimum evals
    p_g = p_i[best_particle]                               # global best

    # generations loop
    for t in tqdm(range(generation)):
        file = open("../../data/pso/pso" + str(t + 1) + ".txt", "w")

        # particles loop
        pbar = tqdm(range(n))
        for i in pbar:
            # show progress bar
            pbar.set_description(f"[Gen. {t}]")

            # write file
            file.write(str(xs[i][0]) + " " + str(xs[i][1]) + " " + str(xs[i][2]) + "\n")

            # update velocity
            vs[i] = update_v(xs[i], vs[i], p_i[i], p_g)

            # update position
            xs[i] = update_x(xs[i], vs[i])

            # calc personal best
            score = eval_func(vel_data, U, xs[i])
            if score < best_scores[i]:
                best_scores[i] = score
                p_i[i] = xs[i]

        # update global best
        best_particle = np.argmin(best_scores)
        p_g = p_i[best_particle]
        file.close()

        print(f"{t}/{generation} generation fin.")

    print(p_g)
    print(min(best_scores))


if __name__ == '__main__':
    data = np.loadtxt("../../data/sample_cp.csv", delimiter=",")
    main(data, U=200)
