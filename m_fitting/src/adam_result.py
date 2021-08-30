from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def scatter_graph(x, y):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('iteration', fontsize=28)
    plt.ylabel('error', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.yscale('log')

    plt.scatter(x, y)

    fig.show()


if __name__ == '__main__':
    print("start program")

    path = "../data/result_adam_175.csv"
    df = pd.read_csv(path)
    scatter_graph(x=df.index.values[1:], y=df.values[1:, -1].astype(np.float64))

    print("program fin.")
