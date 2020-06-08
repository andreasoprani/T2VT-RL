import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

from misc import utils
import numpy as np
import glob
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="results/two-room-gw/linear/")
parser.add_argument("--lambda_test", default=False)

to_bool = lambda x: x in [True, "True", "true", "1"]

args = parser.parse_args()
results_path = str(args.path)
lambda_test = to_bool(args.lambda_test)

if not lambda_test:
    experiments = {
        "1-MGVT": "mgvt_1c",
        "3-MGVT": "mgvt_3c",
        "1-T2VT": "t2vt_1c",
        "3-T2VT": "t2vt_3c"
    }
else:
    experiments = {
        "1c-lambda-0.1": "t2vt_1c_l=0.1",
        "1c-lambda-0.2": "t2vt_1c_l=0.2",
        "1c-lambda-0.3": "t2vt_1c_l=0.3",
        "1c-lambda-0.4": "t2vt_1c_l=0.4",
        "1c-lambda-0.5": "t2vt_1c_l=0.5",
        "1c-lambda-0.6": "t2vt_1c_l=0.6",
        "1c-lambda-0.7": "t2vt_1c_l=0.7",
        "1c-lambda-0.8": "t2vt_1c_l=0.8",
        "1c-lambda-0.9": "t2vt_1c_l=0.9",
        "1c-lambda-1.0": "t2vt_1c_l=1.0",
        "1c-likelihood": "t2vt_1c_l=likelihood",
        "3c-lambda-0.1": "t2vt_3c_l=0.1",
        "3c-lambda-0.2": "t2vt_3c_l=0.2",
        "3c-lambda-0.3": "t2vt_3c_l=0.3",
        "3c-lambda-0.4": "t2vt_3c_l=0.4",
        "3c-lambda-0.5": "t2vt_3c_l=0.5",
        "3c-lambda-0.6": "t2vt_3c_l=0.6",
        "3c-lambda-0.7": "t2vt_3c_l=0.7",
        "3c-lambda-0.8": "t2vt_3c_l=0.8",
        "3c-lambda-0.9": "t2vt_3c_l=0.9",
        "3c-lambda-1.0": "t2vt_3c_l=1.0",
        "3c-likelihood": "t2vt_3c_l=likelihood"
    }

data_index = 3
    
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