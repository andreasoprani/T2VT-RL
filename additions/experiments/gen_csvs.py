import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

from misc import utils
import numpy as np
import glob
import csv

experiments = {
    "1-MGVT": "mgvt_1c",
    "3-MGVT": "mgvt_3c",
    "1-T2VT": "rtde_1c",
    "3-T2VT": "rtde_3c"
}

data_index = 3

def gen_csv(results_path):
    
    out = {
        "i": []
    }
    
    for name, file in experiments.items():
        fs = glob.glob(results_path + file + "*.pkl")
        if len(fs) == 0:
            continue
        r = utils.load_object(fs[0][:-4])
        
        if not out["i"]:
            out["i"] = r[0][2][0]
            
        data = [r[i][2][data_index] for i in range(len(r))]
        
        mean = np.mean(data, axis = 0)
        std = 2 * np.std(data, axis = 0) / np.sqrt(np.array(data).shape[0])
        
        out["mean-" + name] = mean
        out["std-" + name] = std
        
    keys = list(out.keys())
    
    with open(results_path + "results.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(keys)
        writer.writerows(zip(*[out[key] for key in keys]))
      
paths = [
    "results/two-room-gw/linear/",
    "results/two-room-gw/periodic-no-rep/",
    "results/two-room-gw/polynomial/",
    "results/two-room-gw/sin/lambda=0.3333/",
    "results/two-room-gw/sin/lambda=1.0/",
    "results/three-room-gw/linear/",
    "results/three-room-gw/periodic-no-rep/",
    "results/three-room-gw/polynomial/",
    "results/three-room-gw/sin/lambda=0.3333/",
    "results/three-room-gw/sin/lambda=1.0/",
    "results/mountaincar/linear/",
    #"results/mountaincar/periodic-no-rep/",
    #"results/mountaincar/polynomial/",
    #"results/mountaincar/sin/lambda=0.3333/",
    #"results/mountaincar/sin/lambda=1.0/"
]

for p in paths:
    gen_csv(p)