import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import loss_function as lf


def adam(vel_data, U, m, x0, y0):
    """
    Optimize (= Minimize) loss function by params: [m, x0, y0]

    :param: data: nd-array[x, y, u, v], shape -> (~, 4)

    :return: optimized params: [m, x0, y0]
    """

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # set seeds
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # remove NaN
    data_drop_na = pd.DataFrame(vel_data).dropna()
    x = torch.tensor(data_drop_na[[0]].values, device=device)
    y = torch.tensor(data_drop_na[[1]].values, device=device)
    vel = torch.tensor(data_drop_na[[2, 3]].values, device=device)

    # set params and optimizer
    m = torch.tensor(m, dtype=torch.float32, requires_grad=True)
    x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    y0 = torch.tensor(y0, dtype=torch.float32, requires_grad=True)
    params = [m, x0, y0]

    lr = 0.01  # learning rate
    loss_func = lf.global_vector_norm  # loss function
    optimizer = torch.optim.Adam(params, lr)

    # optimization
    m_list = []
    x0_list = []
    y0_list = []
    f_list = []
    i_list = []

    for i in tqdm(range(100000)):
        optimizer.zero_grad()
        outputs = loss_func(vel, x, y, U, params)
        outputs.backward()
        optimizer.step()
        m_list.append(m.item())
        x0_list.append(x0.item())
        y0_list.append(y0.item())
        f_list.append(outputs.item())
        i_list.append(i)

        if outputs < 1e-4:
            break

    # save results
    columns = ["m", "x0", "y0", "error"]
    df = pd.concat([pd.Series(m_list), pd.Series(x0_list), pd.Series(y0_list), pd.Series(f_list)], axis=1)
    df.columns = columns

    return df
