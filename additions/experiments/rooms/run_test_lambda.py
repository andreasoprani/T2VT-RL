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
parser.add_argument("--max_iter", default=3000)
parser.add_argument("--load_results", default=False)
parser.add_argument("--n_runs", default=20)
parser.add_argument("--n_jobs", default=5)

args = parser.parse_args()
exp_type = str(args.exp_type)
max_iter = int(args.max_iter)
load_results = bool(args.load_results)
n_runs = int(args.n_runs)
n_jobs = int(args.n_jobs)

env = "two-room-gw"

lambdas = []#np.linspace(0.1, 1, 10)
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
    task += " --post_components=1"
    task += " --testing_lambda=True"
    task += " --experiment_type=" + e
    task += " --max_iter=" + str(max_iter)
    task += " --n_runs=" + str(n_runs)
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