import subprocess
import datetime
import os
import sys
import argparse

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder

parser = argparse.ArgumentParser()
parser.add_argument("--exp_type", default="linear")
parser.add_argument("--gen_samples", default=True)
parser.add_argument("--max_iter_gen", default=100000)
parser.add_argument("--mgvt_1", default=True)
parser.add_argument("--mgvt_3", default=True)
parser.add_argument("--rtde_1", default=True)
parser.add_argument("--rtde_3", default=True)
parser.add_argument("--max_iter", default=-1) # -1 for default
parser.add_argument("--temporal_bandwidth", default=-1) # -1 for default

args = parser.parse_args()
exp_type = str(args.exp_type)
gen = bool(args.gen_samples)
max_iter_gen = int(args.max_iter_gen)
mgvt_1 = bool(args.mgvt_1)
mgvt_3 = bool(args.mgvt_3)
rtde_1 = bool(args.rtde_1)
rtde_3 = bool(args.rtde_3)
max_iter = int(args.max_iter)
temporal_bandwidth = float(args.temporal_bandwidth)

env = "two-room-gw"

tasks = {
         "run_mgvt.py --post_components=1": mgvt_1,
         "run_mgvt.py --post_components=3": mgvt_3,
         "run_rtde.py --post_components=1": rtde_1,
         "run_rtde.py --post_components=3": rtde_3
        }

if max_iter == -1:
    if exp_type == "periodic-no-rep":
        max_iter = 10000
    else: # linear, sin and polynomial
        max_iter = 3000
        
gen_samples = "gen_samples.py"

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

for k, v in tasks.items():
    if not v: 
        continue
    f = k
    f += " --max_iter=" + str(max_iter)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    if f.find("rtde") and temporal_bandwidth >= 0:
        f += " --temporal_bandwidth=" + str(temporal_bandwidth)
        
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))