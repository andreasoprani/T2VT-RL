import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

from misc import utils
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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

# apply t_SNE
weights = TSNE(n_components=2).fit_transform(weights)

# door means copied from sin
no_door_zone = 0
gw_size = 10
doors_std = 0.2
timesteps = 10
d_means = np.sin((2 * np.pi) * np.linspace(0, 1, timesteps + 1))
d_means = d_means * ((gw_size - 2 * no_door_zone - 1 - 2 * doors_std) / 2) + (gw_size / 2)

# Plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('t-SNE - ' + file_name, 
             fontsize = 20)
targets = [str(np.around(d, decimals = 2)) for d in d_means]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'] 
for i in range(10):
    pc1 = weights[5*i : 5*(i+1),0]
    pc2 = weights[5*i : 5*(i+1), 1]
    ax.scatter(pc1, pc2, c = colors[i], s = 50)
# annotations, set add_dp = false to disable them.
add_dp = True
x_offset = 0.2
y_offset = 0.2
if add_dp:
    for i, dp in enumerate(door_positions):
        x = weights[i,0]
        y = weights[i,1]
        dp = str(np.around(dp, decimals = 2))
        ax.annotate(dp, (x + x_offset, y + y_offset))
ax.legend(targets)
ax.grid()

plt.savefig("additions/exploration/" + "t-SNE_" + file_name + ".pdf", format = 'pdf')

plt.show()