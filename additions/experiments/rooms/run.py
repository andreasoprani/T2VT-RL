import subprocess
import datetime

python = "C:/Users/andre/Anaconda3/python.exe"
path = "d:/Documenti/GitHub/Thesis/additions/experiments/rooms/"
files = ["gen_samples.py",
        #"run_gvt.py",
        #"run_mgvt.py --post_components=1",
        #"run_mgvt.py --post_components=3",
        #"run_rtde.py --post_components=1",
        #"run_rtde.py --post_components=3"
        ]

env = "three-room-gw"
max_iter = "12000" # 2000 for two-room, 12000 for three-room
max_iter_gen = "1000000" # 100000 for two-room, 500000 for three-room
exploration_gen = "0.7"
just_one_timestep = "8"

for f in files:
    if f == "gen_samples.py":
        f += " --max_iter=" + max_iter_gen
        f += " --just_one_timestep=" + just_one_timestep
        f += " --exploration_fraction=" + exploration_gen
    else:
        f += " --max_iter=" + max_iter

    f += " --env=" + env
    
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))