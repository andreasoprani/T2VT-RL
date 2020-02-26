import subprocess
import datetime
import os
import sys
import argparse

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder

parser = argparse.ArgumentParser()
parser.add_argument("--gen_samples", default=True)
parser.add_argument("--max_iter_gen", default=1000000)
parser.add_argument("--mgvt_1", default=True)
parser.add_argument("--mgvt_3", default=True)
parser.add_argument("--rtde_1", default=True)
parser.add_argument("--rtde_3", default=True)
parser.add_argument("--max_iter", default=75000)

args = parser.parse_args()
gen = bool(args.gen_samples)
max_iter_gen = int(args.max_iter_gen)
mgvt_1 = bool(args.mgvt_1)
mgvt_3 = bool(args.mgvt_3)
rtde_1 = bool(args.rtde_1)
rtde_3 = bool(args.rtde_3)
max_iter = int(args.max_iter)

tasks = {
         "run_mgvt.py --post_components=1": mgvt_1,
         "run_mgvt.py --post_components=3": mgvt_3,
         "run_rtde.py --post_components=1": rtde_1,
         "run_rtde.py --post_components=3": rtde_3
        }

gen_samples = "gen_samples.py"
# timesteps = 10

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)

    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

#    for t in range(timesteps):
#        f = gen_samples
#        f += " --max_iter=" + str(max_iter_gen)
#        f += " --just_one_timestep=" + str(t)
#
#        print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
#        subprocess.call(python + " " + path + f, shell=True)

for k, v in tasks.items():
    if not v: 
        continue
    f = k
    f += " --max_iter=" + str(max_iter)
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))