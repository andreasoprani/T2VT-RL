import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../trading")) # add trading
sys.path.append(os.path.abspath(path + "/../../..")) # add main folder

from joblib import Parallel, delayed
import numpy as np
from misc import utils
import gym
from ags_trading.unpadded_trading_env.derivatives import TradingDerivatives

n_jobs = 4

save_dataset_path = "additions/experiments/trading/dataset-"
save_actions_path = "visualize-actions/"

etr_path = "additions/trading/extra-tree-regressors/"

etrs = {
    "2015": {
        "policy": "2015_ms2854_it10_seed2_iter4",
        "task": gym.make("TradingDer2015-v2")
    },
    "2016": {
        "policy": "2016_ms2854_seed1_iter4",
        "task": gym.make("TradingDer2016-v2")
    },
    "2017": {
        "policy": "2017_ms2854_it10_seed4_iter3",
        "task": gym.make("TradingDer-v3")
    },
    "2018": {
        "policy": "2018_ms2854_seed0_iter6",
        "task": gym.make("TradingDer2018-v2")
    }
}

def year_pass(k, v):
    
    Q = utils.load_object(etr_path + v["policy"])
    task = v["task"]
    
    task.starting_day_index = 0
    task.reset()

    days = []
    rewards = np.zeros((task.n_days, len(task.prices) - task.time_lag))
    actions = np.zeros((task.n_days, len(task.prices) - task.time_lag))

    state_value_list = []

    for di in range(task.n_days):
    
        task.starting_day_index = di
        s = task.reset()

        days.append(task.selected_day)

        done = False
        while not done:
            
            a_list = Q._q_values([s])
            
            state_value_list.append([s, a_list])

            a = np.argmax(a_list)
            s, r, done, _ = task.step(a)

            actions[di, task.current_timestep] = a - 1 # [0, 2] -> [-1, 1] 
            rewards[di, task.current_timestep] = r
        
        print("{0:s} - Day: {1:4d}, Cumulative reward: {2:8.6f}".format(k, di, np.sum(rewards)))
    
    utils.save_object(state_value_list, save_dataset_path + k)
    utils.save_object([days, actions, rewards], save_actions_path + k)

    return k

if n_jobs == 1:
    output = [year_pass(k, v) for (k,v) in etrs.items()]
elif n_jobs > 1:
    output = Parallel(n_jobs=n_jobs)(delayed(year_pass)(k, v) for (k,v) in etrs.items())

print(output)