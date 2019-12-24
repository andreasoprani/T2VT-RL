import sys
import os
import datetime

path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/.."))

from misc import utils
import numpy as np

print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))