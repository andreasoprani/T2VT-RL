import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/.."))

from misc import utils
from additions.temporal_kernel import temporal_weights_calculator
import numpy as np
import datetime

#sys.setrecursionlimit(3000)

for s in range(1):
    np.random.seed(s)
    h = 1/np.sqrt(4 * np.pi)
    weights_path = "additions/experiments/rooms/sources-polynomial-2r"
    weights = utils.load_object(weights_path)

    timesteps = 10

    ws = []
    for i in range(timesteps):
        t_ws = weights[i]
        np.random.shuffle(t_ws)
        ws.append(t_ws[0][1])

    print(s)

    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  

    print(temporal_weights_calculator(ws, timesteps, "crossval", h=h))

    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
