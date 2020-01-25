import subprocess
import datetime

python = "C:/Users/andre/Anaconda3/python.exe"
path = "d:/Documenti/GitHub/Thesis/additions/experiments/mountaincar/"
files = ["gen_samples.py",
         "run_mgvt.py --post_components=1",
         #"run_mgvt.py --post_components=3",
         "run_rtde.py --post_components=1",
         #"run_rtde.py --post_components=3"
        ]

for f in files:
    print(f + " - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    subprocess.call(python + " " + path + f)

print("EXECUTION COMPLETE - " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))