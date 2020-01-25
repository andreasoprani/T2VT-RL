import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(path + "/.."))

from misc import utils

source_file = "additions/experiments/rooms/sources-3r"

sources = utils.load_object(source_file)

print(sources[8][2][0])