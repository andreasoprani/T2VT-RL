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
cdm = np.zeros((50, 50))
for i in range(50):
    for j in range(i, 50):
        cd = distance.euclidean(weights[i], weights[j])
        cdm[i,j] = cd
        cdm[j,i] = cd

cluster_distances = np.zeros((10, 10))
for i in range(10):
    for j in range(i, 10):
        cds = cdm[5*i : 5*(i+1), 5*j : 5*(j+1)]
        cluster_distances[i,j] = np.mean(cds)
        cluster_distances[j,i] = np.mean(cds)

# door means copied from sin
no_door_zone = 0
gw_size = 10
doors_std = 0.2
timesteps = 10
d_means = np.sin((2 * np.pi) * np.linspace(0, 1, timesteps + 1))
d_means = d_means * ((gw_size - 2 * no_door_zone - 1 - 2 * doors_std) / 2) + (gw_size / 2)
d_means = d_means[:-1]
d_means = np.around(d_means, decimals = 2)

x = []
y = []
for i in range(10):
    for j in range(i, 10):
        door_dist = np.abs(d_means[i] - d_means[j])
        dist = cluster_distances[i,j]
        x.append(door_dist)
        y.append(dist)
x = np.array(x)
y = np.array(y)
# Fit with polyfit
b, m = polyfit(x, y, 1)
        
cluster_distances = np.around(cluster_distances, decimals = 2)

fig, ax = plt.subplots(figsize=(9,9))
im = ax.imshow(cluster_distances)

ax.set_xticks(np.arange(len(d_means)))
ax.set_yticks(np.arange(len(d_means)))
ax.set_xticklabels(d_means)
ax.set_yticklabels(d_means)

#plt.setp(ax.get_xticklabels(), rotation=0 ha="right",
#         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, cluster_distances[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Clusters solutions mean distances", fontsize = 20)

plt.savefig("additions/exploration/clusters-distances-heatmap_" + file_name + ".pdf", format = 'pdf')

plt.show()

fig, ax = plt.subplots(figsize=(16,9))
ax.plot(x, y, '.')
ax.plot(x, b + m * x, '-')

ax.set_title("Cluster solutions mean distances by door distances", fontsize = 20)
ax.set_xlabel('Door distance', fontsize = 15)
ax.set_ylabel('Mean euclidean distance between clusters', fontsize = 15)
ax.grid()

plt.savefig("additions/exploration/cluster-distances-scatter_" + file_name + ".pdf", format = 'pdf')

plt.show()