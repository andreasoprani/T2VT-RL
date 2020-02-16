import subprocess
import datetime

python = "C:/Users/andre/Anaconda3/python.exe"
path = "d:/Documenti/GitHub/Thesis/additions/experiments/mountaincar/"
files = [
         #"run_mgvt.py --post_components=1",
         "run_mgvt.py --post_components=3",
         #"run_rtde.py --post_components=1",
         "run_rtde.py --post_components=3"
        ]

gen = False
gen_samples = "gen_samples.py"
max_iter_gen = 1000000
# timesteps = 10

max_iter = 75000

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)

    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f)

#    for t in range(timesteps):
#        f = gen_samples
#        f += " --max_iter=" + str(max_iter_gen)
#        f += " --just_one_timestep=" + str(t)
#
#        print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
#        subprocess.call(python + " " + path + f)

for f in files:
    f += " --max_iter=" + str(max_iter)
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))