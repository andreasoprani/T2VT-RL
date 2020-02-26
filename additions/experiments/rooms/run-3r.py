import subprocess
import datetime
import os
import sys

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder
files = [
         #"run_mgvt.py --post_components=1",
         "run_mgvt.py --post_components=3",
         #"run_rtde.py --post_components=1",
         "run_rtde.py --post_components=3"
        ]

env = "three-room-gw"

gen = False
gen_samples = "gen_samples.py"
max_iter_gen = 1000000
exp_type = "linear"
no_door_zone = 2.0
# timesteps = 10

max_iter = 15000

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    f += " --no_door_zone=" + str(no_door_zone)

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

for f in files:
    f += " --max_iter=" + str(max_iter)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type

    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))