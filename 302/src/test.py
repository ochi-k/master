import numpy as np

from modules import calc


super_dir = "../data/"

U = [175, 200, 225]

start = 0
end = 1000


if __name__ == '__main__':
    for u in U:
        ave_dir = super_dir + f"{u}_302/ave/"
        data = np.loadtxt(ave_dir + "ave.csv", delimiter=",")
        error = calc.my_fitting(data=data, U=u)
