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
parser.add_argument("--max_iter_gen", default=1500000) 
parser.add_argument("--no_door_zone", default=2.0)
parser.add_argument("--mgvt_1", default=True)
parser.add_argument("--mgvt_3", default=True)
parser.add_argument("--t2vt_1", default=True)
parser.add_argument("--t2vt_3", default=True)
parser.add_argument("--max_iter", default=15000)
parser.add_argument("--temporal_bandwidth", default=-1) # -1 for default
parser.add_argument("--load_results", default=False)
parser.add_argument("--n_jobs", default=1)

to_bool = lambda x : x in [True, "True", "true", "1"]

args = parser.parse_args()
exp_type = str(args.exp_type)
gen = to_bool(args.gen_samples)
max_iter_gen = int(args.max_iter_gen)
no_door_zone = float(args.no_door_zone) 
mgvt_1 = to_bool(args.mgvt_1)
mgvt_3 = to_bool(args.mgvt_3)
t2vt_1 = to_bool(args.t2vt_1)
t2vt_3 = to_bool(args.t2vt_3)
max_iter = int(args.max_iter)
temporal_bandwidth = float(args.temporal_bandwidth)
load_results = to_bool(args.load_results)
n_jobs = int(args.n_jobs)

env = "three-room-gw"

tasks = {
         "run_mgvt.py --post_components=1": mgvt_1,
         "run_mgvt.py --post_components=3": mgvt_3,
         "run_t2vt.py --post_components=1": t2vt_1,
         "run_t2vt.py --post_components=3": t2vt_3
        }

gen_samples = "gen_samples.py"

# timesteps = 10

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    f += " --no_door_zone=" + str(no_door_zone)
    if n_jobs > 1:
        f += " --n_jobs=" + str(n_jobs)

    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

#    for t in range(timesteps):
#        f = gen_samples
#        f += " --max_iter=" + str(max_iter_gen)
#        f += " --env=" + env
#        f += " --experiment_type=" + exp_type
#        f += " --no_door_zone=" + str(no_door_zone)
#        f += " --just_one_timestep=" + str(t)
#
#        print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
#        subprocess.call(python + " " + path + f, shell=True)

for k, v in tasks.items():
    if not v: 
        continue
    f = k
    f += " --max_iter=" + str(max_iter)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    if f.find("t2vt") and temporal_bandwidth >= 0:
        f += " --temporal_bandwidth=" + str(temporal_bandwidth)
    f += " --load_results=" + str(load_results)
    if n_jobs > 1:
        f += " --n_jobs=" + str(n_jobs)
        
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))