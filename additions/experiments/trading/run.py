import subprocess
import datetime

python = sys.executable # path to current python executable
path = os.path.dirname(os.path.realpath(__file__)) + "/" # path to this folder
files = [
         "run_mgvt.py --post_components=1",
         #"run_mgvt.py --post_components=3",
         "run_rtde.py --post_components=1",
         #"run_rtde.py --post_components=3"
        ]

gen = True
gen_samples = "gen_samples.py"
max_iter_gen = 10000000

max_iter = 10000
n_runs = 5

if gen:
    f = gen_samples
    f += " --max_iter=" + str(max_iter_gen)
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

for f in files:
    f += " --max_iter=" + str(max_iter)
    f += " --n_runs=" + str(n_runs)
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f, shell=True)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))