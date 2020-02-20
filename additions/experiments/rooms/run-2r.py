import subprocess
import datetime
import os
import sys

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder
files = [
         #"run_mgvt.py --post_components=1",
         #"run_mgvt.py --post_components=3",
         "run_rtde.py --post_components=1",
         #"run_rtde.py --post_components=3"
        ]

env = "two-room-gw"

exp_type = "sin"

if exp_type == "periodic-no-rep":
    max_iter = 10000
else: # linear, sin and polynomial
    max_iter = 3000

_lambda = 1

gen = False
gen_samples = "gen_samples.py"
max_iter_gen = 100000

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

for f in files:
    f += " --max_iter=" + str(max_iter)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    f += " --temporal_bandwidth=" + str(_lambda) 

    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))