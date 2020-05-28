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
from additions.approximators.mlp_torch import MLPQFunction
from misc.policies import EpsilonGreedy

save_actions_path = "visualize-actions/"

# Q params
l1 = 32
l2 = 32
layers = [l1,l2]

w_dict = {
    "source_2015": {
        "task": gym.make("VecTradingPrices2015-v2"),
        "filepath": "additions/experiments/trading/",
        "filename": "sources",
        "source_index": 0
    },
    "source_2016": {
        "task": gym.make("VecTradingPrices2016-v2"),
        "filepath": "additions/experiments/trading/",
        "filename": "sources",
        "source_index": 1
    },
    "source_2017": {
        "task": gym.make("VecTradingPrices-v3"),
        "filepath": "additions/experiments/trading/",
        "filename": "sources",
        "source_index": 2
    }
}

for k, v in w_dict.items():
    if "source_index" in v.keys():
        weights = utils.load_object(v["filepath"] + v["filename"])
        weights = weights[v["source_index"]][0][1]
        v["weights"] = weights
    else:
        weights = utils.load_object(v["filepath"] + v["filename"])
        weights = weights[0][1]
        v["weights"] = weights

def year_pass(Q, task):
    
    days = []
    rewards = np.zeros((task.n_days, len(task.prices[0])))
    actions = np.zeros((task.n_days, len(task.prices[0])))
    state_value_list = []
    
    for di in range(task.n_days):
    
        task.starting_day_index = di
        s = task.reset()

        days.append(di)

        done = False
        while not done:
            a_list = Q.value_actions(s)
            state_value_list.append([s[0], a_list])
            a = np.argmax(a_list)
            s, r, done, _ = task.step([a])
        
            r = r[0]
            done = done[0]

            actions[di, task.current_timestep] = a - 1 # [0, 2] -> [-1, 1] 
            rewards[di, task.current_timestep] = r
        
        print("{0:s} - Day: {1:4d}, Cumulative reward: {2:8.6f}".format(k, di, np.sum(rewards)))

    return [days, actions, rewards, state_value_list]

def make_Q(weights, task):
    
    # task params
    state_dim = task.state_dim
    action_dim = 1
    n_actions = task.action_space.n
    
    return MLPQFunction(state_dim, n_actions, layers=layers, initial_params=weights)

for k, v in w_dict.items():
    print(k)
    Q = make_Q(v["weights"], v["task"])
    v["task"].starting_day_index = 0
    v["task"].reset()
    output = year_pass(Q, v["task"])
    print(len(output))
    utils.save_object(output, save_actions_path + k)