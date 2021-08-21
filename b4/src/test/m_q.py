import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    q_m_data = np.loadtxt("../../data/Q30.csv", delimiter=",", skiprows=1)
    u = [175, 200, 225, 250]
    c = ["red", "blue", "green", "orange"]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([0, 7.5])
    ax.set_ylim([0, 3500])

    x = [3.58, 5.07, 7.15]
    v = ["3.58", "5.07", "7.15"]
    m = ["o", "v", "s", "d"]

    plt.xticks(x, v, fontsize=4)
    plt.yticks(fontsize=4)

    for line in q_m_data:
        color = c[u.index(line[1])]
        marker = m[u.index(line[1])]
        plt.scatter(line[0], line[2], label=line[1], c=color, s=500, marker=marker, zorder=10)
    plt.grid()
    plt.show()
    plt.savefig("../../data/m_Q30.png", dpi=300)
