import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

from misc import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

sources_path = path + "/../experiments/rooms/sources-"
file_name = "sin-2r"
sources = utils.load_object(sources_path + file_name)

weights = []
door_positions = []

for i in range(10):
    for j in range(5):
        weights.append(sources[i][j][1])
        door_positions.append(sources[i][j][0][1])
weights = np.array(weights)

# Normalize
with_std = True
scaler = StandardScaler(with_std=with_std)
weights = StandardScaler().fit_transform(weights)

# co-distances matrix
x = []
y = []
for i in range(50):
    for j in range(i, 50):
        door_dist = np.abs(door_positions[i] - door_positions[j])
        dist = distance.euclidean(weights[i], weights[j])
        x.append(door_dist)
        y.append(dist)
x = np.array(x)
y = np.array(y)

# Fit with polyfit
b, m = polyfit(x, y, 1)

fig, ax = plt.subplots(figsize=(16,9))
ax.plot(x, y, '.')
ax.plot(x, b + m * x, '-')

ax.set_title("Tasks solutions distances by door distances", fontsize = 20)
ax.set_xlabel('Door distance', fontsize = 15)
ax.set_ylabel('Solution euclidean distance', fontsize = 15)
ax.grid()

plt.savefig("additions/exploration/tasks-distances_" + file_name + ".pdf", format = 'pdf')

plt.show()