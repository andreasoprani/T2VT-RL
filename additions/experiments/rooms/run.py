import subprocess

python = "C:/Users/andre/Anaconda3/python.exe"
path = "d:/Documenti/GitHub/Thesis/additions/experiments/rooms/"
files = [# "d:/Documenti/GitHub/Thesis/additions/experiments/rooms/gen_samples.py",
        "run_gvt.py",
        "run_mgvt.py --post_components=1",
        "run_mgvt.py --post_components=3",
        "run_rtde.py --post_components=1",
        "run_rtde.py --post_components=3"]

env = "three-room-gw"

for f in files:
    print(f)
    subprocess.call(python + " " + path + f + " --env=" + env)

print("EXECUTION COMPLETE")