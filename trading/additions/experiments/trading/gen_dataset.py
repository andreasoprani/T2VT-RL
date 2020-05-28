import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../trading")) # add trading
sys.path.append(os.path.abspath(path + "/../../..")) # add main folder

from joblib import Parallel, delayed
import numpy as np
from misc import utils
import gym
from ags_trading.vectorized_trading_env.prices import VecTradingPrices

n_jobs = 10

save_dataset_path = "additions/experiments/trading/dataset-"
save_actions_path = "visualize-actions/"

etr_path = "additions/trading/extra-tree-regressors/"

etrs = {
    "2015": {
        "policy": "2015_ms2854_it10_seed2_iter4",
        "task": gym.make("VecTradingPrices2015-v2")
    },
    "2016": {
        "policy": "2016_ms2854_seed1_iter4",
        "task": gym.make("VecTradingPrices2016-v2")
    },
    "2017": {
        "policy": "2017_ms2854_it10_seed4_iter3",
        "task": gym.make("VecTradingPrices-v3")
    },
    "2018": {
        "policy": "2018_ms2854_seed0_iter6",
        "task": gym.make("VecTradingPrices2018-v2")
    }
}

def day_pass(k, v, d):

    Q = utils.load_object(etr_path + v["policy"])
    task = v["task"]

    task.starting_day_index = d
    s = task.reset()

    rewards = np.zeros((len(task.prices[0])))
    actions = np.zeros((len(task.prices[0])))

    state_value_list = []

    done = False

    while not done:
            
        a_list = Q._q_values(s)
        
        state_value_list.append([s[0], a_list])
        a = np.argmax(a_list)
        s, r, done, _ = task.step([a])
        
        r = r[0]
        done = done[0]

        actions[task.current_timestep] = a - 1 # [0, 2] -> [-1, 1] 
        rewards[task.current_timestep] = r
    
    print("{0:s} - Day: {1:4d}, Cumulative reward: {2:8.6f}".format(k, d, np.sum(rewards)))
    
    return (d, rewards, actions, state_value_list)

def year_pass(k, v):
    
    Q = utils.load_object(etr_path + v["policy"])
    task = v["task"]
    
    task.starting_day_index = 0
    task.reset()
    num_days = task.n_days

    if n_jobs == 1:
        outputs = [day_pass(k, v, d) for d in range(num_days)]
    elif n_jobs > 1:
        outputs = Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(day_pass)(k, v, d) for d in range(num_days))
    
    days = []
    actions = np.zeros((num_days, len(task.prices[0])))
    rewards = np.zeros((num_days, len(task.prices[0])))
    state_value_list = []

    for (d, r, a, svl) in outputs:
        
        days.append(d)
        rewards[d, :] = r
        actions[d, :] = a

        state_value_list.extend(svl)

    print("Days:", len(days))
    print("Rewards sum:", np.sum(rewards))
    print("State values list length:", len(state_value_list))

    utils.save_object(state_value_list, save_dataset_path + k)
    utils.save_object([days, actions, rewards], save_actions_path + k)

for k, v in etrs.items():
    year_pass(k,v)