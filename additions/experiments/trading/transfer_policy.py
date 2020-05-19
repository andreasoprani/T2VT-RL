import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../trading")) # add trading
sys.path.append(os.path.abspath(path + "/../../..")) # add main folder

from joblib import Parallel, delayed
import numpy as np
from misc import utils
import gym
from additions.approximators.mlp_torch import MLPQFunction
from ags_trading.vectorized_trading_env.prices import VecTradingPrices

layers = [32, 32]

dataset = {
    '2015': {
        'data': 'additions/experiments/trading/dataset-2015',
        'task': gym.make('VecTradingPrices2015-v2')
    },
    '2016': {
        'data': 'additions/experiments/trading/dataset-2016',
        'task': gym.make('VecTradingPrices2016-v2')
    },
    '2017': {
        'data': 'additions/experiments/trading/dataset-2017',
        'task': gym.make('VecTradingPrices-v3')
    },
    '2018': {
        'data': 'additions/experiments/trading/dataset-2018',
        'task': gym.make('VecTradingPrices2018-v2')
    }
} 

y = 2015

data = dataset[str(y)]['data']
data = utils.load_object(data)

task = dataset[str(y)]['task']
state_dim = task.state_dim
n_actions = task.action_space.n 

Q = MLPQFunction(state_dim, n_actions, layers=layers)

np.random.seed(0)
Q.init_weights()

s = np.array(data[0][0])

print(Q._nn.forward(s))
print()

