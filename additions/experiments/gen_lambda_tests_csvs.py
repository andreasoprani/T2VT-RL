import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

from misc import utils
import numpy as np
import glob
import csv

experiments = {
    "lambda = 0.1": "rtde_1c_l=0.1",
    "lambda = 0.2": "rtde_1c_l=0.2",
    "lambda = 0.3": "rtde_1c_l=0.3",
    "lambda = 0.4": "rtde_1c_l=0.4",
    "lambda = 0.5": "rtde_1c_l=0.5",
    "lambda = 0.6": "rtde_1c_l=0.6",
    "lambda = 0.7": "rtde_1c_l=0.7",
    "lambda = 0.8": "rtde_1c_l=0.8",
    "lambda = 0.9": "rtde_1c_l=0.9",
    "lambda = 1.0": "rtde_1c_l=1.0"
}

data_index = 3

def gen_csv(results_path):
    
    print(results_path)
    
    out = {
        "i": []
    }
    
    for name, file in experiments.items():
        fs = glob.glob(results_path + file + "*.pkl")
        if len(fs) == 0:
            continue
        r = utils.load_object(fs[0][:-4])
        
        if not out["i"]:
            out["i"] = r[0][2][0][1:]
            
        data = [r[i][2][data_index][1:] for i in range(len(r))]
        
        mean = np.mean(data, axis = 0)
        std = 2 * np.std(data, axis = 0, ddof = 1) / np.sqrt(np.array(data).shape[0])
        
        out["mean-" + name] = mean
        out["std-" + name] = std
        
    keys = list(out.keys())
    
    with open(results_path + "results.csv", "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(keys)
        writer.writerows(zip(*[out[key] for key in keys]))
     
paths = [
    "results/two-room-gw/linear/lambda_test/",
    "results/two-room-gw/periodic-no-rep/lambda_test/",
    "results/two-room-gw/polynomial/lambda_test/",
    "results/two-room-gw/sin/lambda_test/"
]

for p in paths:
    gen_csv(p)