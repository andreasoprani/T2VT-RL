import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../../trading")) # add trading
sys.path.append(os.path.abspath(path + "/../../..")) # add main folder

import gym
from ags_trading.unpadded_trading_env.derivatives import TradingDerivatives

import numpy as np
from additions.approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from additions.algorithms.nt import learn
from misc import utils
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms.dqn import DQN

# Global parameters
verbose = False

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--max_iter", default=1000000)
parser.add_argument("--buffer_size", default=50000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--exploration_fraction", default=0.5)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.01)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=100)
parser.add_argument("--mean_episodes", default=20)
parser.add_argument("--l1", default=64)
parser.add_argument("--l2", default=0)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--dqn", default=False)

parser.add_argument("--just_one_timestep", default=-1) # Used to re-train for just one timestep. -1 = False, 2014 -> 2017 = timestep to train
parser.add_argument("--sources_file_name", default=path + "/sources")
parser.add_argument("--tasks_file_name", default=path + "/tasks")

# Read arguments
args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
batch_size = int(args.batch_size)
max_iter = int(args.max_iter)
buffer_size = int(args.buffer_size)
random_episodes = int(args.random_episodes)
exploration_fraction = float(args.exploration_fraction)
eps_start = float(args.eps_start)
eps_end = float(args.eps_end)
train_freq = int(args.train_freq)
eval_freq = int(args.eval_freq)
mean_episodes = int(args.mean_episodes)
l1 = int(args.l1)
l2 = int(args.l2)
alpha = float(args.alpha)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
dqn = bool(args.dqn)

just_one_timestep = int(args.just_one_timestep)
sources_file_name = str(args.sources_file_name)
tasks_file_name = str(args.tasks_file_name)

# Seed to get reproducible results
seed = 1
np.random.seed(seed)

# All environments
envs = {
    2014: 'TradingDer2014-v2',
    2015: 'TradingDer2015-v2',
    2016: 'TradingDer2016-v2',
    2017: 'TradingDer-v3'
    }
mdps = [gym.make(envs[k]) for k in envs.keys()]

n_eval_episodes = 5

state_dim = mdps[0].state_dim
action_dim = 1
n_actions = mdps[0].action_space.n

layers = [l1]
if l2 > 0:
    layers.append(l2)

if not dqn:
    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, mdps[0].gamma, state_dim, action_dim)
    # Create Q Function
    Q = MLPQFunction(state_dim, n_actions, layers=layers)
else:
    Q, operator = DQN(state_dim, action_dim, n_actions, mdps[0].gamma, layers=layers)

def run(mdp, seed=None):
    return learn(mdp,
                 Q,
                 operator,
                 max_iter=max_iter,
                 buffer_size=buffer_size,
                 batch_size=batch_size,
                 alpha=alpha,
                 train_freq=train_freq,
                 eval_freq=eval_freq,
                 eps_start=eps_start,
                 eps_end=eps_end,
                 exploration_fraction=exploration_fraction,
                 random_episodes=random_episodes,
                 eval_episodes=n_eval_episodes,
                 mean_episodes=mean_episodes,
                 seed=seed,
                 verbose=verbose)

last_rewards = 5

results = []

if just_one_timestep in envs.keys():
    results = utils.load_object(sources_file_name)
    index = list(envs.keys()).index(just_one_timestep)
    mdp = mdps[index]
    print(mdp.get_info())
    results[index] = [run(mdp, seed)]
    print("Last learning rewards:", np.around(results[index][0][2][3][-last_rewards:], decimals = 5))
    utils.save_object(results, sources_file_name)
else:
    for mdp in mdps:
        print(mdp.get_info())
        results.append([run(mdp, seed)])
        print("Last learning rewards:", np.around(results[-1][0][2][3][-last_rewards:], decimals = 5))
        utils.save_object(results, sources_file_name)

tasks = [gym.make('TradingDer2018-v2')]
utils.save_object(tasks, tasks_file_name)