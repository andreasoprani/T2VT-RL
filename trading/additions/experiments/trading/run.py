import subprocess
import datetime
import os
import sys
import argparse

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder

parser = argparse.ArgumentParser()
parser.add_argument("--gen_samples", default=False)
parser.add_argument("--mgvt_1", default=True)
parser.add_argument("--mgvt_3", default=False)
parser.add_argument("--t2vt_1", default=True)
parser.add_argument("--t2vt_3", default=False)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--max_iter", default=30000000)

args = parser.parse_args()
gen = bool(args.gen_samples)
mgvt_1 = bool(args.mgvt_1)
mgvt_3 = bool(args.mgvt_3)
t2vt_1 = bool(args.t2vt_1)
t2vt_3 = bool(args.t2vt_3)
n_runs = int(args.n_runs)
max_iter = int(args.max_iter)

tasks = {
         "run_mgvt.py --post_components=1": mgvt_1,
         "run_mgvt.py --post_components=3": mgvt_3,
         "run_t2vt.py --post_components=1": t2vt_1,
         "run_t2vt.py --post_components=3": t2vt_3
        }

gen_samples = "gen_samples.py"
timesteps = [2014, 
             2015, 
             2016, 
             2017]

if gen:
    for t in timesteps:
        f = gen_samples
        f += " --just_one_timestep=" + str(t)
        print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        subprocess.call(python + " " + path + f, shell=True)

for k, v in tasks.items():
    if not v: 
        continue
    f = k
    f += " --max_iter=" + str(max_iter)
    f += " --n_runs=" + str(n_runs)
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))