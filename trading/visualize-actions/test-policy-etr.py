import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/.."))
sys.path.append(os.path.abspath(path + "/../additions/trading"))

import numpy as np
from misc import utils
import gym
from ags_trading.unpadded_trading_env.derivatives import TradingDerivatives

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

def year_pass(Q, task):
    
    days = []
    rewards = np.zeros((task.n_days, len(task.prices) - task.time_lag))
    actions = np.zeros((task.n_days, len(task.prices) - task.time_lag))
    
    for di in range(task.n_days):
    
        task.starting_day_index = di
        s = task.reset()
        s = [s]
        
        print("Day index:", di)

        days.append(task.selected_day)

        done = False
        while not done:

            a = np.argmax(Q._q_values(s))
            s, r, done, _ = task.step(a)
            s = [s]

            actions[di, task.current_timestep] = a - 1 # [0, 2] -> [-1, 1] 
            rewards[di, task.current_timestep] = r
        
        print("Cumulative reward:", np.sum(rewards))
            
    return [days, actions, rewards]

for k, v in etrs.items():
    print(k)
    Q = utils.load_object(etr_path + v["policy"])
    task = v["task"]
    task.starting_day_index = 0
    task.reset()
    output = year_pass(Q, task)
    utils.save_object(output, "visualize-actions/" + k)