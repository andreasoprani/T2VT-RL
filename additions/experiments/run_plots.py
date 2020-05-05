import subprocess
import datetime
import os
import sys
import argparse

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder

result_paths = [
    "results/two-room-gw/linear/",
    "results/two-room-gw/periodic-no-rep/",
    "results/two-room-gw/polynomial/",
    "results/two-room-gw/sin/lambda=0.3333/",
    "results/two-room-gw/sin/lambda=1.0/",
    "results/three-room-gw/linear/",
    "results/three-room-gw/periodic-no-rep/",
    "results/three-room-gw/polynomial/",
    "results/three-room-gw/sin/lambda=0.3333/",
    "results/three-room-gw/sin/lambda=1.0/",
    "results/mountaincar/linear/",
    #"results/mountaincar/periodic-no-rep/",
    #"results/mountaincar/polynomial/",
    #"results/mountaincar/sin/lambda=0.3333/",
    #"results/mountaincar/sin/lambda=1.0/"
]

for result_path in result_paths:
    print(result_path)
    f = "plot_results.py --path=" + result_path
    f += " --show=False"
    subprocess.call(python + " " + path + f, shell=True)