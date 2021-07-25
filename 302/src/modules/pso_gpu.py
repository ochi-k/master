import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def eval_func(vel, x, y, U, params):  # evaluation function
    m, x0, y0 = params

    # calc velocity
    u = m * (x - x0) / ((x - x0) ** 2 + (y - y0) ** 2)
    v = - U + m * (y - y0) / ((x - x0) ** 2 + (y - y0) ** 2)
    vel_pred = torch.cat([u, v], axis=1)
    # debug = np.count_nonzero(np.isnan(vel_pred.cpu().detach().numpy().copy()))

    # calc error
    error_norm = torch.norm(vel - vel_pred)
    vel_norm = torch.norm(vel)
    error = error_norm / vel_norm

    return error


def update_x(x, v):
    return x + v


def update_v(device, x, v, p_i, p_g, w=0.5, r_max=1.0, c1=1, c2=1):
    new_v = torch.zeros_like(v, device=device)

    # set two randoms (default: [0, 1.0])
    r1 = random.uniform(0, r_max)
    r2 = random.uniform(0, r_max)

    for i in range(len(x)):
        new_v[i] = w * v[i] + c1 * r1 * (p_i[i] - x[i]) + c2 * r2 * (p_g[i] - x[i])

    return new_v


def main(vel_data, U):
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = "cpu"

    # set seeds
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # remove NaN
    data_drop_na = pd.DataFrame(vel_data).dropna()
    x = torch.tensor(data_drop_na[[0]].values, device=device)
    y = torch.tensor(data_drop_na[[1]].values, device=device)
    vel = torch.tensor(data_drop_na[[2, 3]].values, device=device)

    # set params
    n = 10000             # particles
    dim = 3               # dimensions
    generation = 10       # max generations
    m_range = [0, 3000]   # m range
    x0_range = [-40, 40]  # x0 range
    y0_range = [-40, 40]  # y0 range

    # initialize particle position
    xs = torch.zeros(n, dim, device=device)
    vs = torch.zeros_like(xs, device=device)

    print("\n[initialize]")
    for i in tqdm(range(n)):
        xs[i][0] = random.uniform(*m_range)   # m init
        xs[i][1] = random.uniform(*x0_range)  # x0 init
        xs[i][2] = random.uniform(*y0_range)  # y0 init

    # set vars for evaluation
    print("\n[set vars]")
    p_i = xs                                                                    # best position of the i-th particle
    best_scores = torch.tensor([eval_func(vel, x, y, U, i) for i in tqdm(xs)],  # eval personal best
                               device=device)
    best_particle = torch.argmin(best_scores)                                   # particle index in minimum evals
    p_g = p_i[best_particle.item()]                                             # global best
    judge = 100
    tmp_p_g = None
    flag = 0

    # generations loop
    print("\n[generate]")
    for t in range(generation):
        file = open("../../data/pso/pso" + str(t + 1) + ".txt", "w")

        for i in tqdm(range(n), desc=f"\n[Gen. {t+1} / {generation}]"):
            # write file
            file.write(str(xs[i][0]) + " " + str(xs[i][1]) + " " + str(xs[i][2]) + "\n")

            # update velocity
            vs[i] = update_v(device, xs[i], vs[i], p_i[i], p_g, r_max=0.5)

            # update position
            xs[i] = update_x(xs[i], vs[i])

            # calc personal best
            score = eval_func(vel, x, y, U, xs[i])
            if score.item() < best_scores[i].item():
                best_scores[i] = score
                p_i[i] = xs[i]

        # save particles
        file.close()

        # update global best
        if torch.min(best_scores) < judge:
            flag = 0
            judge = torch.min(best_scores)
            best_particle = torch.argmin(best_scores)
            tmp_p_g = p_i[best_particle.item()].clone().detach()

        else:
            flag += 1

        if flag > 2:
            break

        best_particle = torch.argmin(best_scores)
        p_g = p_i[best_particle.item()]

    print(f"\n{tmp_p_g}")
    print(f"{judge}")


if __name__ == '__main__':
    print("start program!")

    data = np.loadtxt("../../data/sample_cp.csv", delimiter=",")
    main(data, U=200)

    print("\nPSO for gpu fin.")
