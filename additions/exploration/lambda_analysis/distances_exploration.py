import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../.."))

from misc import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

sources_path = path + "/../../experiments/rooms/sources-"
tasks = ["linear", "periodic-no-rep", "polynomial", "sin"]

for t in tasks:
    file_name = t + "-2r"
    sources = utils.load_object(sources_path + file_name)

    weights = [[sources[i][j][1] for j in range(5)] for i in range(10)]

    x = [i for i in range(9) for j in range(25)]
    y = [distance.euclidean(weights[i][j], weights[i+1][k]) for i in range(9) for j in range(5) for k in range(5)]

    x_avg = list(range(9))
    y_avg = [np.average(y[i:i+25]) for i in range(0,9*25,25)]

    avg = np.average(y)

    fig, ax = plt.subplots(figsize=(16,9))

    ax.plot(x, y, 'b.', label="Euclidean distance")
    ax.plot(x_avg, y_avg, 'ro', label="Distances average")
    ax.axhline(avg, label="Overall average")

    ax.set_title("Euclidean distances between solutions of subsequent timesteps - " + file_name, fontsize=20)
    ax.set_xlabel("Timesteps", fontsize = 15)
    ax.set_ylabel("Q-functions euclidean distances", fontsize = 15)
    plt.xticks(x_avg, [str(i) + " - " + str(i+1) for i in x_avg])
    plt.ylim(bottom = 0, top = 8)
    ax.grid()
    ax.legend()

    plt.savefig("additions/exploration/lambda_analysis/euc-dist_" + file_name + ".png", format = "png")

    plt.show()
