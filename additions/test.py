import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(path + "/.."))

from misc import utils
import numpy as np 

timesteps = 100
doors_std = 0.2
gw_size = 10
no_door_zone = 2

d_means = np.sin((2 * np.pi) * np.linspace(0, 1, timesteps + 1))
d_means = d_means * ((gw_size - no_door_zone - 1 - 2 * doors_std) / 2) + ((gw_size - no_door_zone) / 2)

print(d_means)