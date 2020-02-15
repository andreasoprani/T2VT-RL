import subprocess
import datetime

python = "C:/Users/andre/Anaconda3/python.exe"
path = "d:/Documenti/GitHub/Thesis/additions/experiments/rooms/"
files = [
         "run_mgvt.py --post_components=1",
         "run_mgvt.py --post_components=3",
         "run_rtde.py --post_components=1",
         "run_rtde.py --post_components=3"
        ]

env = "two-room-gw"

exp_type = "periodic-no-rep"

if exp_type == "periodic-no-rep":
    max_iter = 10000
else: # linear and sin
    max_iter = 3000

gen = False
gen_samples = "gen_samples.py"
max_iter_gen = 100000

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type
    
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f)

for f in files:
    f += " --max_iter=" + str(max_iter)
    f += " --env=" + env
    f += " --experiment_type=" + exp_type

    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))