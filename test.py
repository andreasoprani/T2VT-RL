import sys
import os
from misc import utils

path = os.path.dirname(os.path.realpath(__file__)) + "/experiments/rooms/sources-2r"

results = utils.load_object(path)

for i in range(len(results)):
    for j in range(len(results[i])):
        print(str(i) + ", " + str(j) + ": " + str(results[i][j][0]))