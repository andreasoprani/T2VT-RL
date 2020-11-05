import subprocess
import datetime
import os
import sys
import argparse

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder

results_paths = [
    #"results/two-room-gw/linear/",
    #"results/two-room-gw/polynomial/",
    #"results/two-room-gw/sin/lambda=0.3333/",
    #"results/two-room-gw/sin/lambda=1.0/",
    #"results/three-room-gw/linear/",
    #"results/three-room-gw/polynomial/",
    #"results/three-room-gw/sin/lambda=0.3333/",
    #"results/three-room-gw/sin/lambda=1.0/",
    #"results/mountaincar/linear/",
    #"results/mountaincar/polynomial/",
    #"results/mountaincar/sin/lambda=0.3333/",
    #"results/mountaincar/sin/lambda=1.0/",
    "results/lake/",
    #"results/two-room-gw/linear/lambda_test/pc=1/ --lambda_test=True",
    #"results/two-room-gw/linear/lambda_test/pc=3/ --lambda_test=True",
    #"results/two-room-gw/polynomial/lambda_test/pc=1/ --lambda_test=True",
    #"results/two-room-gw/polynomial/lambda_test/pc=3/ --lambda_test=True",
    #"results/two-room-gw/sin/lambda_test/pc=1/ --lambda_test=True",
    #"results/two-room-gw/sin/lambda_test/pc=3/ --lambda_test=True",
    #"results/three-room-gw/linear/lambda_test/pc=1/ --lambda_test=True",
    #"results/three-room-gw/linear/lambda_test/pc=3/ --lambda_test=True",
    #"results/three-room-gw/polynomial/lambda_test/pc=1/ --lambda_test=True",
    #"results/three-room-gw/polynomial/lambda_test/pc=3/ --lambda_test=True",
    #"results/three-room-gw/sin/lambda_test/pc=1/ --lambda_test=True",
    #"results/three-room-gw/sin/lambda_test/pc=3/ --lambda_test=True",
    #"results/mountaincar/linear/lambda_test/pc=1/ --lambda_test=True",
    #"results/mountaincar/linear/lambda_test/pc=3/ --lambda_test=True",
    #"results/mountaincar/polynomial/lambda_test/pc=1/ --lambda_test=True",
    #"results/mountaincar/polynomial/lambda_test/pc=3/ --lambda_test=True",
    #"results/mountaincar/sin/lambda_test/pc=1/ --lambda_test=True",
    #"results/mountaincar/sin/lambda_test/pc=3/ --lambda_test=True"
]

show = False

for results_path in results_paths:
    print(results_path)
    f = "plot_results.py --path=" + results_path
    f += " --show=" + str(show)
    subprocess.call(python + " " + path + f, shell=True)