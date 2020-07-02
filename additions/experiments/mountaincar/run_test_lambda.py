import subprocess
import datetime
import os
import sys
import argparse
import numpy as np
import scipy

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder

parser = argparse.ArgumentParser()
parser.add_argument("--exp_type", default="") # "" for all exp_types
parser.add_argument("--max_iter", default=75000) # select 3000 for two-room and 15000 for three-room
parser.add_argument("--post_components", default=3)
parser.add_argument("--n_runs", default=50)
parser.add_argument("--load_results", default=False) # load previously found results and extend them
parser.add_argument("--n_jobs", default=5)

to_bool = lambda x : x in [True, "True", "true", "1"]

args = parser.parse_args()
exp_type = str(args.exp_type)
max_iter = int(args.max_iter)
post_components = int(args.post_components)
n_runs = int(args.n_runs)
load_results = to_bool(args.load_results)
n_jobs = int(args.n_jobs)

lambdas = np.linspace(0.1, 1, 10)
lambda_presets = [
    "likelihood"
    ]

exps = [
    "linear",
    "polynomial",
    "sin"
]

if exp_type != "":
    exps = [exp_type]

for i, e in enumerate(exps):

    task = "run_t2vt.py"
    task += " --post_components=" + str(post_components)
    task += " --testing_lambda=True"
    task += " --experiment_type=" + e
    task += " --max_iter=" + str(max_iter)
    task += " --n_runs=" + str(n_runs)
    task += " --load_results=" + str(load_results)
    task += " --n_jobs=" + str(n_jobs)

    for l in lambdas:
        f = task
        f += " --temporal_bandwidth=" + str(l)
        print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        subprocess.call(python + " " + path + f, shell=True)

    for p in lambda_presets:
        f = task
        f += " --lambda_preset=" + p
        print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        subprocess.call(python + " " + path + f, shell=True)