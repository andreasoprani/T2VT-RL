import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(path + "/.."))

from misc import utils
import numpy as np 

timesteps = 10
doors_std = 0.2
gw_size = 10

a = np.random.uniform(low = np.pi / 4, high = np.pi / 2)
d_means = np.sin (np.linspace(a, a + np.pi, timesteps + 1))
d_means = d_means * ((gw_size - 1 - 2 * doors_std) / 2) + (gw_size/2)

print(a*180/np.pi)
print(d_means)