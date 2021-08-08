import numpy as np
import pandas as pd
import torch


def loss_func(vel, x, y, U, m, x0, y0):
    # calc velocity
    _u = m * (x - x0) / ((x - x0) ** 2 + (y - y0) ** 2) \
         + m * (x + x0) / ((x + x0) ** 2 + (y - y0) ** 2)
    _v = - U + m * (y - y0) / ((x - x0) ** 2 + (y - y0) ** 2) \
         + m * (y - y0) / ((x + x0) ** 2 + (y - y0) ** 2)

    # concatenate velocity
    _vel_pred = torch.cat([_u, _v], axis=1)

    # calc error
    error = (vel - _vel_pred).norm() / vel.norm()

    return error


def test_func(vel, x, y, U, m, x0, y0):
    # calc velocity
    _u = m * (x - x0) / ((x - x0) ** 2 + (y - y0) ** 2)
    _v = - U + m * (y - y0) / ((x - x0) ** 2 + (y - y0) ** 2)
    _u_inv = ((x - x0) ** 2 + (y - y0) ** 2) / m / (x - x0)
    _v_inv = ((x - x0) ** 2 + (y - y0) ** 2) / (m * (y - y0) - U * ((x - x0) ** 2 + (y - y0) ** 2))

    # concatenate velocity
    _vel_pred = torch.cat([_u, _v], axis=1)
    _vel_inv_pred = torch.cat([_u_inv, _v_inv], axis=1)
    a = np.count_nonzero(np.isnan(_vel_pred.detach().numpy().copy()))

    # calc error
    error = (vel - _vel_pred).norm() / vel.norm()
    # error = (vel - _vel_inv_pred).norm() / vel.norm()

    d = vel.norm()
    dd = (vel - _vel_inv_pred).detach().numpy().copy()
    ddd = (vel - _vel_inv_pred).norm()
    dddd = np.count_nonzero(np.isnan(dd))

    return error


def adam(data, U):
    """
    Optimize (= Minimize) loss function by params: [m, x0, y0]
    Loss function: Vector norm2 error (rel)

    :param: data: nd-array[x, y, u, v], shape -> (~, 4)

    :return: optimized params: [m, x0, y0]
    """

    # remove NaN
    data_drop_na = pd.DataFrame(data).dropna()
    x = torch.tensor(data_drop_na[[0]].values)
    y = torch.tensor(data_drop_na[[1]].values)
    vel = torch.tensor(data_drop_na[[2, 3]].values)
    vel_inv = torch.tensor(data_drop_na[[4, 5]].values)

    # set params and optimizer
    m = torch.tensor(2020.7799, requires_grad=True)
    x0 = torch.tensor(10.0186, requires_grad=True)
    y0 = torch.tensor(-5.0152, requires_grad=True)
    params = [m, x0, y0]

    lr = 0.01  # learning rate
    optimizer = torch.optim.Adam(params, lr)

    # optimization
    outputs = None
    m_list = []
    x0_list = []
    y0_list = []
    f_list = []
    i_list = []

    for i in range(100000):
        print(f"Step: {i}")
        optimizer.zero_grad()
        outputs = test_func(vel, x, y, U, *params)
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
    df.to_csv('../data/adam_result.csv')

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(15, 10))
    plt.scatter(i_list, f_list)
    plt.yscale('log')
    plt.ylim(0, 1)
    fig.show()

    return outputs
