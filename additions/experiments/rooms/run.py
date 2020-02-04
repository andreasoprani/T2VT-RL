import subprocess
import datetime

python = "C:/Users/andre/Anaconda3/python.exe"
path = "d:/Documenti/GitHub/Thesis/additions/experiments/rooms/"
files = [
         #"run_mgvt.py --post_components=1",
         #"run_mgvt.py --post_components=3",
         #"run_rtde.py --post_components=1",
         #"run_rtde.py --post_components=3"
        ]

env = "three-room-gw"

gen = True
gen_samples = "gen_samples.py"
max_iter_gen = "20000000"
timesteps = range(10)
threshold_learn = "True"
eval_threshold = "0.5"
eval_consistency = "5"

max_iter = "50000"

if gen:
    for t in timesteps:
        f = gen_samples
        f += " --max_iter=" + max_iter_gen
        f += " --just_one_timestep=" + str(t)
        f += " --env=" + env
        f += " --threshold_learn=" + threshold_learn
        f += " --eval_threshold=" + eval_threshold
        f += " --eval_consistency=" + eval_consistency

        print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        subprocess.call(python + " " + path + f)

for f in files:
    f += " --max_iter=" + max_iter
    f += " --env=" + env

    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))