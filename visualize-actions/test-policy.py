import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/.."))
sys.path.append(os.path.abspath(path + "/../additions/trading"))

import numpy as np
from misc import utils
import gym
from gym import spaces
from ags_trading.unpadded_trading_env.derivatives import TradingDerivatives
from additions.approximators.mlp_torch import MLPQFunction
from misc.policies import EpsilonGreedy

w_dict = {
    "mgvt_1c": {
        "task": gym.make('TradingDer2018-v2'),
        "filepath": "results/trading/",
        "filename": "mgvt_1c_2020-03-30_11-21-20"
    },
    "rtde_1c": {
        "task": gym.make('TradingDer2018-v2'),
        "filepath": "results/trading/",
        "filename": "rtde_1c_2020-04-02_14-21-18"
    },
    "source_2014": {
        "task": gym.make('TradingDer2014-v2'),
        "filepath": "additions/experiments/trading/",
        "filename": "sources",
        "source_index": 0
    },
    "source_2015": {
        "task": gym.make('TradingDer2015-v2'),
        "filepath": "additions/experiments/trading/",
        "filename": "sources",
        "source_index": 1
    },
    "source_2016": {
        "task": gym.make('TradingDer2016-v2'),
        "filepath": "additions/experiments/trading/",
        "filename": "sources",
        "source_index": 2
    },
    "source_2017": {
        "task": gym.make('TradingDer-v3'),
        "filepath": "additions/experiments/trading/",
        "filename": "sources",
        "source_index": 3
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
    rewards = np.zeros((task.n_days, len(task.prices) - task.time_lag))
    actions = np.zeros((task.n_days, len(task.prices) - task.time_lag))
    
    for di in range(task.n_days):
    
        task.starting_day_index = di
        s = task.reset()
        
        print("Day index:", di)

        days.append(task.selected_day)

        done = False
        while not done:

            a = np.argmax(Q.value_actions(s))
            s, r, done, _ = task.step(a)

            actions[di, task.current_timestep] = a - 1 # [0, 2] -> [-1, 1] 
            rewards[di, task.current_timestep] = r
            
    return [days, actions, rewards]

def make_Q(weights, task):
    
    # Q params
    l1 = 32
    l2 = 32
    layers = [l1,l2]
    
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
    utils.save_object(output, "visualize-actions/" + k)